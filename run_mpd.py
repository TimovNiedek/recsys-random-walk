import argparse
import numpy as np
from tqdm import tqdm
from recsysrw.playlist_model import HistogramPlaylist
from recsysrw import shared, metrics
from recsysrw.random_walk import MultipleRandomWalk, normalize_visit_counts, sum_visit_counts
import json
import datetime
from recsysrw.title_processing import TitleIndexer, expand_on_title, normalize_title
from recsysrw.graph import MPDGraph
import time


def main(graph_file, index_path, n_test=10000, alpha=0.99, n=50000, n_p=1000, n_v=4, n_jobs=0, seed=None,
         smoothing=0.25, name_expansion=False, prune_distance=0, n_distance_prune=0, degree_prune_factor=1.0,
         switch_degree_prune=False, title_prune=False):
    now = datetime.datetime.now()

    # Load the graph
    shared.init(graph_file,
                seed,
                n_jobs)
    graph = shared.graph  # type: MPDGraph

    # make the degree pruning masks (takes a while)
    if degree_prune_factor < 1.0 and not switch_degree_prune:
        prune_mask = graph.make_degree_prune_mask(degree_prune_factor)
    elif switch_degree_prune:
        prune_mask_075 = graph.make_degree_prune_mask(0.75)
        prune_mask_09 = graph.make_degree_prune_mask(0.75)
        prune_mask_095 = graph.make_degree_prune_mask(0.75)


    title_indexer = TitleIndexer(index_path)
    test_playlists, ground_truth, titles = graph.split_train_test_spotify(n_test=n_test, seed=seed)

    rps = []
    ndcgs = []
    rss = []

    rps_expansion = []
    ndcgs_expansion = []
    rss_expansion = []

    rw = MultipleRandomWalk(alpha, n, n_p, n_v, n_jobs,
                            use_boosting=False,
                            dynamic_allocation=False,
                            pixie_weighting=False)

    early_stops = 0.0
    num_walks_total = 0

    # Dict like:
    # { <playlist_id>: {<metadatas>:<values>, <metric>:<score>}  }
    performance_stats = {}

    speeds = {}

    t = tqdm(list(zip(test_playlists, ground_truth, titles)), total=len(test_playlists), desc='pid = ...')
    for pid, gt_tracks, title in t:
        start = time.time()
        graph.disable_test_connections(pid, gt_tracks)
        playlist_tracks = graph.neighbors(pid, only_enabled=True, unique=True)
        graph.disable_node(pid)
        num_q = len(playlist_tracks)
        t.set_description("{} ({}, n={})".format(title, pid, num_q))

        if len(playlist_tracks) == 0 and not name_expansion:
            rps.append(0)
            ndcgs.append(0)
            rss.append(51)
            graph.enable_all()
            continue

        num_pruned_nodes = 0
        before_prune_edges = np.sum(graph._enabled_edges) / 2

        # High-degree pruning
        if degree_prune_factor < 1.0 and not switch_degree_prune:
            graph.disable_mask(prune_mask)
        elif switch_degree_prune:
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

        playlist_features = graph.features[playlist_tracks - 1000000]

        # Feature-based pruning
        if len(playlist_tracks) > 1 and (prune_distance > 0 or n_distance_prune > 0):
            playlist_model = HistogramPlaylist(playlist_features, pid, smoothing=smoothing,
                                               background_model=graph.background_model)
            if n_distance_prune > 0:
                num_pruned_nodes = graph.histogram_distance_prune_top(playlist_model, n_distance_prune)
            else:
                num_pruned_nodes = graph.histogram_distance_prune(playlist_model, prune_distance)

        # Title-based pruning
        if title_prune and title is not None:
            # Keep playlists with similar titles
            to_keep = title_indexer.find_top_similar(title, threshold=0)
            if len(to_keep) > 150:
                print("Keeping {} nodes based on title".format(len(to_keep)))
                to_remove = np.setdiff1d(graph.playlist_nodes, to_keep, assume_unique=True)
                graph.disable_nodes(to_remove)

        # Get shared array without copying data
        enabled_np = np.frombuffer(shared.arr_enabled, dtype='bool')
        # Update the shared array
        print("TOTAL ENABLED EDGES: ", np.sum(graph._enabled_edges))
        enabled_np[:] = graph._enabled_edges
        after_prune_edges = np.sum(graph._enabled_edges) / 2
        num_pruned_edges = before_prune_edges - after_prune_edges
        print('Pruned {} edges (ratio of {})'.format(num_pruned_edges, (num_pruned_edges) / before_prune_edges))

        if len(playlist_tracks) > 0:
            visit_count, early_stop, num_walks = rw.run_random_walk(query_tracks=playlist_tracks, pid=pid)
            visit_count = normalize_visit_counts(visit_count)
        else:
            visit_count = {}
            early_stop = 0
            num_walks = 0

        if name_expansion and title is not None and len(playlist_tracks) == 0:
            expansion_nodes, frequencies = expand_on_title(title, title_indexer, frequency_threshold=0.15,
                                                           max_expand=100, disallowed_playlist=pid)
            if len(expansion_nodes) > 0:
                visit_counts_expansions, _, _ = rw.run_random_walk(query_tracks=expansion_nodes, counts_per_node=True)
                visit_count = normalize_visit_counts(sum_visit_counts(visit_counts_expansions))

            else:
                expansion_nodes, frequencies = expand_on_title(title, title_indexer, frequency_threshold=0.0,
                                                               max_expand=100, disallowed_playlist=pid)
                if len(expansion_nodes) > 0:
                    visit_counts_expansions, _, _ = rw.run_random_walk(query_tracks=expansion_nodes,
                                                                       counts_per_node=True)
                    visit_count = normalize_visit_counts(sum_visit_counts(visit_counts_expansions))
                else:
                    visit_count = {}

        early_stops += early_stop
        num_walks_total += num_walks

        top_500 = [(key, value) for (key, value) in
                   sorted(iter(visit_count.items()), key=lambda k_v: (k_v[1], k_v[0]))[::-1] if
                   key not in playlist_tracks][:500]

        recommendations = np.array([key for (key, _) in top_500])

        rp = metrics.r_precision(gt_tracks, recommendations)
        ndcg_score = metrics.ndcg(gt_tracks, recommendations, 500)
        rs_clicks = metrics.playlist_extender_clicks(gt_tracks, recommendations)
        rps.append(rp)
        rss.append(rs_clicks)
        ndcgs.append(ndcg_score)

        median_degree_query = np.median([graph.degree(p_track) for p_track in playlist_tracks])
        median_degree_heldout = np.median([graph.degree(p_track) for p_track in gt_tracks])
        median_degree_recc = np.median([graph.degree(p_track) for p_track in recommendations])
        mean_features_query = np.mean(playlist_features, axis=0)
        mean_features_heldout = np.mean(graph.features[np.array(gt_tracks) - 1000000], axis=0)
        mean_features_recommendations = np.mean(graph.features[recommendations - 1000000], axis=0) if len(
            recommendations) > 0 else None
        num_query = len(playlist_tracks)
        num_heldout = len(gt_tracks)
        performance_stats[str(pid)] = {
            'playlist_title': title,
            'normalized_title': normalize_title(title) if title is not None else None,
            'num_pruned_nodes': int(num_pruned_nodes),
            'num_pruned_edges': int(num_pruned_edges),
            'median_degree_query': median_degree_query,
            'median_degree_heldout': median_degree_heldout,
            'median_degree_recommendations': median_degree_recc,
            'num_query_tracks': num_query,
            'num_heldout_tracks': num_heldout,
            'r_precision': rp,
            'ndcg': ndcg_score,
            'rs_clicks': rs_clicks
        }

        if name_expansion:
            rp_expansion = rp
            ndcg_expansion = ndcg_score
            rs_clicks_expansion = rs_clicks
            rps_expansion.append(rp_expansion)
            ndcgs_expansion.append(ndcg_expansion)
            rss_expansion.append(rs_clicks_expansion)
            performance_stats[str(pid)]['r_precision_expanded'] = rp_expansion
            performance_stats[str(pid)]['ndcg_expanded'] = ndcg_expansion
            performance_stats[str(pid)]['rs_clicks_expanded'] = rs_clicks_expansion

        for i, val in enumerate(mean_features_query):
            performance_stats[str(pid)]['feature_{}_query'.format(i)] = val
        for i, val in enumerate(mean_features_heldout):
            performance_stats[str(pid)]['feature_{}_heldout'.format(i)] = val
        if mean_features_recommendations is not None:
            for i, val in enumerate(mean_features_recommendations):
                performance_stats[str(pid)]['feature_{}_recommendations'.format(i)] = val
        graph.enable_all()

        end = time.time()
        diff = end - start
        if num_query not in speeds:
            speeds[num_query] = [diff]
        else:
            speeds[num_query].append(diff)

    for n, times in speeds.items():
        print('num query: {} \t avg time: {} \t num samples: {}'.format(n, np.mean(times), len(times)))

    if n_jobs > 0:
        print("Closing pool")
        shared.pool.close()
        shared.pool.join()
    try:
        with open('../Experiment_results/{}_{}_{}_{}_{}_{}.json'.format(now.year, now.month, now.day, now.hour,
                                                                        now.minute,
                                                                        now.second), 'w') as f:
            json.dump(performance_stats, f, indent=4, separators=(',', ':'))
    except Exception as e:
        print("Failed to save results due to exception:")
        print(e)

    print("Results for run started at: ", now.strftime("%Y-%m-%d %H:%M"))

    print("Avg R-precision: %.4f, median: %.4f" % (np.mean(rps), np.median(rps)))
    print("Avg NDCG: %.4f, median: %.4f" % (np.mean(ndcgs), np.median(ndcgs)))
    print("Avg RS Clicks: %.4f, median: %.4f" % (np.mean(rss), np.median(rss)))
    print()
    if name_expansion:
        print("Expanded:")
        print("Avg R-precision: %.4f, median: %.4f" % (np.mean(rps_expansion), np.median(rps_expansion)))
        print("Avg NDCG: %.4f, median: %.4f" % (np.mean(ndcgs_expansion), np.median(ndcgs_expansion)))
        print("Avg RS Clicks: %.4f, median: %.4f" % (np.mean(rss_expansion), np.median(rss_expansion)))

    print("Percentage of walks that stopped early: %.4f" % (early_stops * 100 / float(num_walks_total)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a Pixie Random Walk experiment')
    parser.add_argument('graph', type=str, help='Path to graph file.')
    parser.add_argument('title_index', type=str, help='Path of title index (or desired path, if the index has not been'
                                                      'built yet).')
    parser.add_argument('-n_test', dest='n_test', type=int, help='Number of test samples to use.', default=5000)
    parser.add_argument('-alpha', dest='alpha', type=float, help='Restart probability.', default=0.99)
    parser.add_argument('-N', dest='N', type=int, help='Number of steps per walk', default=200000)
    parser.add_argument('-n_p', dest='n_p', type=int,
                        help='Early stopping parameter (threshold for number of visits).', default=1000)
    parser.add_argument('-n_v', dest='n_v', type=int,
                        help='Early stopping parameter '
                             '(threshold for number of nodes that reached visits threshold)',
                        default=4)
    parser.add_argument('-n_jobs', dest='n_jobs', type=int, help='Number of parallel processes spawned. Set to 0 for '
                                                                 'no parallelism.', default=0)
    parser.add_argument('-seed', dest='seed', type=int, help='Seed for random train/test split selection and '
                                                             'random walks.', default=None)
    parser.add_argument('-smoothing', dest='smoothing', type=float, default=0.25,
                        help='Amount of uniform smoothing to apply to transition probabilities.')
    parser.add_argument('-name_retrieval', dest='name_expansion', action='store_true', default=True,
                        help='Use playlist name retrieval for playlists with title only.')
    parser.add_argument('-prune_dist', dest='prune_dist', type=float, default=0.0,
                        help='L1 Distance threshold for pruning (not used when -prune_num specified)')
    parser.add_argument('-prune_num', dest='prune_num', type=int, default=0,
                        help='Number of playlists to keep when pruning (not used when -prune_dist specified)')
    parser.add_argument('-d_prune_delta', dest='degree_prune_factor', type=float, default=1.0,
                        help='Prune factor delta, determines amount of high degree pruning. Default = 1.0 (no pruning)')
    parser.add_argument('-switch_d_prune', dest='switch_degree_prune', action='store_true', default=False,
                        help='Switch between pruning method based on number of query tracks')
    parser.add_argument('-t_prune', dest='title_prune', action='store_true', default=False,
                        help='Prune nodes with different titles.')

    args = parser.parse_args()
    print(args)

    main(graph_file=args.graph, index_path=args.title_index, n_test=args.n_test, alpha=args.alpha, n=args.N,
         n_p=args.n_p, n_v=args.n_v, n_jobs=args.n_jobs, seed=args.seed, smoothing=args.smoothing,
         name_expansion=args.name_expansion, prune_distance=args.prune_dist, n_distance_prune=args.prune_num,
         degree_prune_factor=args.degree_prune_factor, switch_degree_prune=args.switch_degree_prune,
         title_prune=args.title_prune)
