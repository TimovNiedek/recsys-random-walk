import argparse
import json
import numpy as np
from tqdm import tqdm
from recsysrw import shared
from recsysrw.random_walk import MultipleRandomWalk
import datetime
from recsysrw.title_processing import TitleIndexer, expand_on_title
from recsysrw.playlist_model import HistogramPlaylist


def challenge_playlist_to_nodes(playlist):
    query_tracks = []
    for track in playlist['tracks']:
        track_uri = track['track_uri']
        if track_uri not in shared.graph.track_node_ids:
            print("Track id was not already in MPD: {}. Skipping this track".format(track_uri))
            continue
        node_id = shared.graph.track_node_ids[track_uri]
        query_tracks.append(node_id)
    return query_tracks


def read_challenge_set(challenge_file):
    with open(challenge_file, 'r') as f:
        challenge_set = json.load(f)

    pids = []
    playlists = []
    titles = []
    for playlist in tqdm(challenge_set['playlists']):
        pid = playlist['pid']
        query_tracks = challenge_playlist_to_nodes(playlist)
        pids.append(pid)
        playlists.append(np.array(query_tracks))
        if 'name' in playlist:
            titles.append(playlist['name'])
        else:
            titles.append(None)
    return pids, playlists, titles


def main(challenge_file, graph_file, title_index, alpha=0.99, n=100000, n_p=2000, n_v=4, n_jobs=0, seed=None,
         smoothing=0.0, prune_distance=0, n_distance_prune=False, degree_prune_factor=1.0,
         switch_degree_prune=False):
    shared.init(graph_file,
                seed,
                n_jobs)

    title_indexer = TitleIndexer(title_index)

    graph = shared.graph  # type: MPDGraph

    # make the degree pruning masks (takes a while)
    if degree_prune_factor < 1.0 and not switch_degree_prune:
        prune_mask = graph.make_degree_prune_mask(degree_prune_factor)
        before_prune_edges = np.sum(graph._enabled_edges) / 2
        graph.disable_mask(prune_mask)
        after_prune_edges = np.sum(graph._enabled_edges) / 2
        print('Pruned {} edges (ratio of {})'.format(before_prune_edges - after_prune_edges,
                                                     (before_prune_edges - after_prune_edges) / before_prune_edges))
    elif switch_degree_prune:
        prune_mask_075 = graph.make_degree_prune_mask(0.75)
        prune_mask_09 = graph.make_degree_prune_mask(0.75)
        prune_mask_095 = graph.make_degree_prune_mask(0.75)

    # Get shared array without copying data
    enabled_np = np.frombuffer(shared.arr_enabled, dtype='bool')
    # Update the shared array
    print("TOTAL ENABLED EDGES: ", np.sum(graph._enabled_edges))
    enabled_np[:] = graph._enabled_edges

    pids, playlists, titles = read_challenge_set(challenge_file)
    now = datetime.datetime.now()

    t = tqdm(list(zip(pids, playlists, titles)), total=len(pids), desc='...')
    submission = "team_info,Team Radboud,creative,timovanniedek@gmail.com\r\n"
    rw = MultipleRandomWalk(alpha=alpha, n=n, n_p=n_p, n_v=n_v, n_jobs=n_jobs, use_boosting=False,
                            dynamic_allocation=False, pixie_weighting=False)

    for pid, playlist_tracks, title in t:
        t.set_description("{} ({}, n={})".format(title, pid, len(playlist_tracks)))

        if switch_degree_prune:
            graph.enable_all()
            num_q = len(playlist_tracks)
            if num_q >= 100:
                graph.disable_mask(prune_mask_075)
                print('Using delta = 0.75')
            elif num_q >= 25:
                graph.disable_mask(prune_mask_09)
                print('Using delta = 0.9')
            elif num_q >= 5:
                graph.disable_mask(prune_mask_095)
                print('Using delta = 0.95')
            else:
                print('Not pruning')


        # Feature-based pruning
        if len(playlist_tracks) > 1 and (prune_distance > 0 or n_distance_prune > 0):
            playlist_features = graph.features[playlist_tracks]
            playlist_model = HistogramPlaylist(playlist_features, pid, smoothing=smoothing,
                                               background_model=graph.background_model)
            if n_distance_prune > 0:
                graph.histogram_distance_prune_top(playlist_model, n_distance_prune)
            else:
                graph.histogram_distance_prune(playlist_model, prune_distance)

        enabled_np = np.frombuffer(shared.arr_enabled, dtype='bool')
        # Update the shared array
        print("TOTAL ENABLED EDGES: ", np.sum(graph._enabled_edges))
        enabled_np[:] = graph._enabled_edges

        if len(playlist_tracks) > 0:
            visit_count, early_stop, num_walks = rw.run_random_walk(query_tracks=playlist_tracks)
        else:
            expansion_nodes, frequencies = expand_on_title(title, title_indexer, frequency_threshold=0.075, max_expand=100)
            if len(expansion_nodes) == 0:
                expansion_nodes, frequencies = expand_on_title(title, title_indexer, frequency_threshold=0.0,
                                                               max_expand=100)
                if len(expansion_nodes) == 0:
                    # Only playlist for which this is a problem is 'universal stereo'
                    print(
                        "Name not shared with any playlist in MPD: {}, mapping to 'universal'".format(title))
                    expansion_nodes, frequencies = expand_on_title('universal', title_indexer,
                                                                   frequency_threshold=0.075, max_expand=100)
            visit_count, early_stop, num_walks = rw.run_random_walk(query_tracks=expansion_nodes)

        recommendations = [shared.graph.node_track_ids[key] for (key, _) in
                           sorted(iter(visit_count.items()), key=lambda k_v: (k_v[1], k_v[0]))[::-1]
                           if key not in playlist_tracks][:500]

        sline = "{},".format(pid)
        sline += ",".join(recommendations) + '\r\n'
        submission += sline

    if n_jobs > 0:
        print("Closing pool")
        shared.pool.close()
        shared.pool.join()

    fname = 'submission_{}_{}_{}_{}_{}_{}.csv'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(fname, 'w+') as f:
        f.write(submission)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate recommendations for the MPD challenge set')
    parser.add_argument('challenge_file', type=str, help='Path to challenge set file')
    parser.add_argument('graph', type=str, help='Path to graph file.')
    parser.add_argument('title_index', type=str, help='Path of title index (or desired path, if the index has not been'
                                                      'built yet).')
    parser.add_argument('-alpha', dest='alpha', type=float, help='Restart probability.', default=0.99)
    parser.add_argument('-N', dest='N', type=int, help='Number of steps per walk', default=100000)
    parser.add_argument('-n_p', dest='n_p', type=int,
                        help='Early stopping parameter (threshold for number of visits).', default=2000)
    parser.add_argument('-n_v', dest='n_v', type=int,
                        help='Early stopping parameter '
                             '(threshold for number of nodes that reached visits threshold)',
                        default=4)
    parser.add_argument('-n_jobs', dest='n_jobs', type=int, help='Number of parallel processes spawned. Set to 0 for '
                                                                 'no parallelism.', default=0)
    parser.add_argument('-seed', dest='seed', type=int, help='Seed for random train/test split selection and '
                                                             'random walks.', default=None)
    parser.add_argument('-prune_dist', dest='prune_dist', type=float, default=0.0,
                        help='L1 Distance threshold for pruning (not used when -prune_num specified)')
    parser.add_argument('-smoothing', dest='smoothing', type=float, default=0.25,
                        help='Amount of Jelinek-Mercer smoothing to apply before computing L1 distance.')
    parser.add_argument('-prune_num', dest='prune_num', type=int, default=0,
                        help='Number of playlists to keep when pruning (not used when -prune_dist specified)')
    parser.add_argument('-d_prune_delta', dest='degree_prune_factor', type=float, default=1.0,
                        help='Prune factor delta, determines amount of high degree pruning. Default = 1.0 (no pruning)')
    parser.add_argument('-switch_d_prune', dest='switch_degree_prune', action='store_true', default=True,
                        help='Switch between pruning method based on number of query tracks')


    args = parser.parse_args()
    print(args)
    main(challenge_file=args.challenge_file, graph_file=args.graph, title_index=args.title_index, alpha=args.alpha,
         n=args.N, n_p=args.n_p, n_v=args.n_v, n_jobs=args.n_jobs, seed=args.seed, smoothing=args.smoothing,
         prune_distance=args.prune_dist, n_distance_prune=args.prune_num, degree_prune_factor=args.degree_prune_factor,
         switch_degree_prune=args.switch_degree_prune)
