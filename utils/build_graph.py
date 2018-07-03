import json
import os
import numpy as np
import argparse
from tqdm import tqdm, trange


def process_mpd(path, num_nodes, track_node_ids, quick=False, max_files_for_quick_processing=1):
    count = 0
    filenames = os.listdir(path)
    adjacency = [list() for _ in range(num_nodes)]

    name_index = {}
    names = []
    t = tqdm(sorted(filenames), desc='Processing ')
    for filename in t:
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            t.set_description("Processing {}".format(filename))
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            for playlist in mpd_slice['playlists']:
                pid = playlist['pid']
                name = playlist['name']
                names.append(name)

                if name in name_index:
                    name_index[name].append(pid)
                else:
                    name_index[name] = [pid]

                for track in playlist['tracks']:
                    track_uri = track['track_uri']
                    track_node_id = track_node_ids[track_uri]
                    adjacency[pid].append(track_node_id)
                    adjacency[track_node_id].append(pid)
            count += 1

            if quick and count >= max_files_for_quick_processing:
                break
    return adjacency, name_index, np.asarray(names)


def flatten_adjacency(adjacency):
    offsets = [0]
    prev_offset = 0
    for i, adj in enumerate(adjacency[1:]):
        cur_offset = prev_offset + len(adjacency[i])
        offsets.append(cur_offset)
        prev_offset = cur_offset
    adj_flat = [node for adjacent in adjacency for node in adjacent]
    return np.array(adj_flat), np.array(offsets)


def release_date_to_feature(datestring):
    if isinstance(datestring, str):
        year = int(datestring[:4])
        if 1920 <= year <= 2020:
            return (year - 1920) / 100
    return np.nan


def metadata_to_features(metadata, track_node_ids, num_tracks):
    selected_metadata = ['acousticness', 'danceability', 'energy', 'explicit', 'instrumentalness', 'liveness',
                         'speechiness', 'tempo', 'valence', 'count', 'release_date']

    track_metadata_list = [list() for _ in range(num_tracks)]
    for track_uri, node_id in tqdm(track_node_ids.items()):
        for key in selected_metadata:
            if track_uri in metadata and key in metadata[track_uri]:
                if key == 'tempo':
                    value = np.clip((metadata[track_uri][key] - 50) / 130.0, 0.0, 1.0)
                elif key == 'count':
                    value = np.log(metadata[track_uri][key]) / np.log(46574)
                elif key == 'release_date':
                    value = release_date_to_feature(metadata[track_uri][key])
                else:
                    value = metadata[track_uri][key]
                if value is None:
                    value = np.nan
                track_metadata_list[node_id - 1000000].append(value)
            else:
                track_metadata_list[node_id - 1000000].append(np.nan)
    return np.array(track_metadata_list)


def build_edge_inverse_list(offsets, adjacency_flat):
    inverse_list = np.zeros(len(adjacency_flat), dtype=int) - 1
    for p_node in trange(1000000):
        playlist_offset = offsets[p_node]
        try:
            p_neighbors = adjacency_flat[playlist_offset: offsets[p_node + 1]]
        except IndexError:
            p_neighbors = adjacency_flat[playlist_offset:]
        for t_node in p_neighbors:
            track_offset = offsets[t_node]
            try:
                t_neighbors = adjacency_flat[track_offset:offsets[t_node + 1]]
            except IndexError:
                t_neighbors = adjacency_flat[track_offset:]

            track_playlist_offsets = np.where(t_neighbors == p_node)[0]
            playlist_track_offsets = np.where(p_neighbors == t_node)[0]
            assert len(track_playlist_offsets) == len(playlist_track_offsets), "Not all edges have an opposite"
            inverse_list[playlist_track_offsets + playlist_offset] = track_playlist_offsets + track_offset
            inverse_list[track_playlist_offsets + track_offset] = playlist_track_offsets + playlist_offset
    return inverse_list


def neighbors(node, adjacency, offsets, unique=False):
    if node < len(offsets) - 1:
        neighbors = adjacency[offsets[node]: offsets[node + 1]]
    else:
        neighbors = adjacency[offsets[node]: len(adjacency)]
    return np.unique(neighbors) if unique else neighbors


def compute_histogram(features, bins=10):
    p_model = np.zeros((features.shape[1], bins))
    for i in range(features.shape[1]):
        f = features[:, i]
        hist, bin_edges = np.histogram(f[~np.isnan(f)], range=(0.0, 1.0), bins=bins)
        p_model[i] = hist
    return p_model / np.sum(p_model)


def make_background_model(playlist_nodes, features, adjacency, offsets):
    concatenated_playlists = []
    for pid in tqdm(playlist_nodes):
        p_tracks = neighbors(pid, adjacency, offsets)
        concatenated_playlists.extend(p_tracks)
    concatenated_playlists = np.asarray(concatenated_playlists)
    background_histograms = compute_histogram(features[concatenated_playlists - 1000000], bins=10)
    return background_histograms


def make_histogram_models(features, adjacency, offsets, nbins=10):
    N = features.shape[0]
    nfeatures = features.shape[1]

    models = np.zeros((N, nfeatures, nbins), dtype=np.float)
    for p in trange(N):
        p_tracks = neighbors(p, adjacency, offsets, unique=True)
        p_features = features[p_tracks - 1000000]
        models[p] = compute_histogram(p_features, bins=nbins)
    return models


def main(metadata_file, mpd_path, target_file, quick=False):
    print("Loading metadata")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    playlist_ids = np.arange(1000000)

    track_node_ids = {}
    for i, track_id in enumerate(metadata.keys()):
        track_node_ids[track_id] = len(playlist_ids) + i

    num_tracks = len(metadata.keys())
    num_nodes = 1000000 + num_tracks

    print("Computing features")
    features = metadata_to_features(metadata, track_node_ids, num_tracks)
    print("Computing adjacency lists")
    adjacency, name_index, playlist_titles = process_mpd(mpd_path, num_nodes, track_node_ids, quick=quick)
    # This is the insertion point for adding other edges (eg. based on playlist title)
    # next step will flatten the lists
    print("Flattening list")
    adjacency, offsets = flatten_adjacency(adjacency)
    # Build the inverse list (takes a while)
    print("Building inverse list")
    inverse_list = build_edge_inverse_list(offsets, adjacency)

    # Build the background histogram model
    print("Building background model")
    background_model = make_background_model(playlist_ids, features, adjacency, offsets)

    # Build the playlist histogram models (takes a while)
    print("Building playlist models")
    playlist_histograms = make_histogram_models(features, adjacency, offsets)


    # Save it

    # adjacency           = flat adjacency list
    # inverse_list        = list containing at every index the index of the opposite direction in the adjacency list;
    #                       if adjacency[0] corresponds to a -> b, adjacency[inverse_list[0]] corresponds to b -> a
    # offsets             = list of offsets where to find neighbors of a given node
    # features            = metadata converted to features for playlist model
    # track_node_ids      = dictionary mapping track uris to node ids
    # name_index          = dictionary mapping playlist titles to a list of node ids
    # playlist_titles     = list of playlist titles
    # playlist_histograms = np.ndarray containing histograms of features per playlist
    # background_model    = np.ndarray containing histograms of features over entire collection
    print("Saving")
    np.savez(target_file, adjacency=adjacency, inverse_list=inverse_list, offsets=offsets, features=features,
             track_node_ids=track_node_ids, name_index=name_index, playlist_titles=playlist_titles,
             playlist_models=playlist_histograms, background_model=background_model)
    print("Save successful.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create the MPD graph from the metadata file.")
    parser.add_argument('metadata', type=str, help='Path to the metadata json file.')
    parser.add_argument('mpd_path', type=str, help='Path to the MPD.')
    parser.add_argument('target_file', type=str, help='Target filename where graph will be written.')
    parser.add_argument('-quick', action='store_true', default=False, help='Limits the number of MPD files '
                                                                           'read for quick testing.')
    args = parser.parse_args()
    main(args.metadata, args.mpd_path, args.target_file, args.quick)
