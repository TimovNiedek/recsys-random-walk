import numpy as np
from sklearn.model_selection import train_test_split
from random import choices
import random
from recsysrw.playlist_model import HistogramPlaylist, l1_distance_multiple
from tqdm import trange, tqdm
import math


class MPDGraph(object):
    def __init__(self, filename, seed=None):
        self._seed = seed
        if seed is not None and seed >= 0:
            np.random.seed(seed)
            random.seed(seed)
        with np.load(filename) as graph:
            self._adjacency = graph['adjacency']
            self._offsets = graph['offsets']
            self._offsets = np.append(self._offsets, [len(self._adjacency)])
            self._features = graph['features']
            self._track_node_ids = graph['track_node_ids'].item()
            self._node_track_ids = {node_id: track_uri for track_uri, node_id in self._track_node_ids.items()}
            self._name_index = graph['name_index'].item()
            self._playlist_titles = graph['playlist_titles']
            self._playlist_histograms = graph['playlist_models']
            self._inverse_list = graph['inverse_list']
            self._background_model = graph['background_model']
        self._enabled_edges = np.ones(len(self._adjacency), dtype=bool)
        self._max_track_degree = np.max([self.degree(n) for n in self.track_nodes])
        print("Max track degree: {}".format(self._max_track_degree))

    def check_inverse_list(self):
        for node_index in trange(self.num_nodes):
            start = self._offsets[node_index]
            end = self._offsets[node_index + 1] if node_index + 1 < self.num_nodes else len(self._adjacency)
            for off in range(start, end):
                off_other = self.get_opposite_edges(off)
                neighbor = self._adjacency[off]
                max_off = self._offsets[neighbor + 1] if neighbor + 1 < self.num_nodes else len(self._adjacency)
                # Other offset is within adjacency list of neighbor
                assert self._offsets[neighbor] <= off_other < max_off, "Error at offset {}, inverse {}".format(off,
                                                                                                               self.get_opposite_edges(
                                                                                                                   off))
                # Other offset points exactly towards the current node in the adjacency list
                assert self._adjacency[off_other] == node_index, "Error at offset {}, inverse {}".format(off,
                                                                                                         self.get_opposite_edges(
                                                                                                             off))

    def build_edge_inverse_list(self):
        inverse_list = np.zeros(len(self._adjacency), dtype=int) - 1
        for p_node in trange(1000000):
            playlist_offset = self._offsets[p_node]
            try:
                p_neighbors = self._adjacency[playlist_offset: self._offsets[p_node + 1]]
            except IndexError:
                p_neighbors = self._adjacency[playlist_offset:]
            for t_node in p_neighbors:
                track_offset = self._offsets[t_node]
                try:
                    t_neighbors = self._adjacency[track_offset:self._offsets[t_node + 1]]
                except IndexError:
                    t_neighbors = self._adjacency[track_offset:]

                track_playlist_offsets = np.where(t_neighbors == p_node)[0]
                playlist_track_offsets = np.where(p_neighbors == t_node)[0]
                assert len(track_playlist_offsets) == len(playlist_track_offsets), "Not all edges have an opposite?"
                inverse_list[playlist_track_offsets + playlist_offset] = track_playlist_offsets + track_offset
                inverse_list[track_playlist_offsets + track_offset] = playlist_track_offsets + playlist_offset

        self._inverse_list = inverse_list
        np.savez('../graph_withinverse2.npz',
                 adjacency=self._adjacency,
                 inverse_list=inverse_list,
                 offsets=self._offsets,
                 features=self._features,
                 track_node_ids=self._track_node_ids,
                 name_index=self._name_index,
                 playlist_titles=self._playlist_titles,
                 playlist_models=self._playlist_histograms)

    def get_opposite_edges(self, edge_index):
        return self._inverse_list[edge_index]

    def sample_playlists(self, playlist_test_ratio, stratify=False, random_state=None):
        if stratify:
            playlist_lenghts = np.asarray([self.degree(pid) for pid in self.playlist_nodes])
            unique, counts = np.unique(playlist_lenghts, return_counts=True)

            idx = np.argmin(counts)

            # Map samples with unique length to a lower count to allow stratification
            for un_idx in np.where(counts == 1)[0]:
                unique_length = unique[un_idx]
                for i in np.where(playlist_lenghts == unique_length)[0]:
                    playlist_lenghts[i] = playlist_lenghts[i - 1]
            train_playlists, test_playlists = train_test_split(np.arange(len(self.playlist_nodes)),
                                                               test_size=playlist_test_ratio,
                                                               stratify=playlist_lenghts,
                                                               random_state=random_state)
        else:
            train_playlists, test_playlists = train_test_split(np.arange(len(self.playlist_nodes)),
                                                               test_size=playlist_test_ratio,
                                                               random_state=random_state)

        return train_playlists, test_playlists

    def split_train_test(self, playlist_test_ratio=0.1,
                         stratify=False,
                         track_test_ratio=0.5,
                         track_mode='random',
                         seed=None,
                         test_playlists=None):
        if track_mode not in {'random', 'first'}:
            raise ValueError("Supported sample modes are 'random' and 'first'")

        if test_playlists is None:
            # Training playlists are kept 100% intact
            # Test playlists are split based on the sampling method & provided track ratio
            _, test_playlists = self.sample_playlists(playlist_test_ratio, stratify, random_state=seed)

        # Keep a list of test tracks per test playlist
        ground_truth = [list() for _ in range(len(test_playlists))]

        # Loop through the test playlists
        for i, p_node in enumerate(test_playlists):
            p_tracks = self.neighbors(p_node)
            n_test_tracks = np.clip(int(len(p_tracks) * track_test_ratio), 1, len(p_tracks) - 1)
            if track_mode == 'random':
                test_tracks = random.sample(p_tracks.tolist(), n_test_tracks)
            else:
                test_tracks = p_tracks[-n_test_tracks:]

            ground_truth[i] = test_tracks

        return test_playlists, ground_truth

    def split_playlist(self, p_node, n_samples, mode='first'):
        if mode not in {'random', 'first'}:
            raise ValueError("Supported sample modes are 'random' and 'first'")

        if self.degree(p_node) <= n_samples:
            raise ValueError("Number of samples is greater than or equal to total number of tracks"
                             " in the playlist")

        p_tracks = self.neighbors(p_node)
        p_unique = np.unique(p_tracks)

        if mode == 'first':
            query_tracks = p_unique[:n_samples]
            ground_truth = p_unique[n_samples:]
        else:
            query_tracks = random.sample(p_unique.tolist(), n_samples)
            ground_truth = random.sample(p_unique.tolist(), len(p_unique) - n_samples)

        assert len(query_tracks) == n_samples, "Sampling incorrect"
        return ground_truth

    def split_train_test_spotify(self, n_test=10000, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        playlist_lengths = np.asarray([len(self.neighbors(p_node, unique=True)) for p_node in self.playlist_nodes])

        test_playlists = []
        ground_truth = []
        titles = []

        n_samples_category = int(n_test / 10)

        # Category: just title
        candidates = np.argwhere(playlist_lengths >= 10).squeeze()
        candidates = list(set(candidates.tolist()) - set(test_playlists))
        just_title = np.random.choice(candidates, size=n_samples_category, replace=False)
        test_playlists.extend(just_title)
        ground_truth.extend([self.split_playlist(p_node, 0, mode='first') for p_node in just_title])
        titles.extend(self.playlist_titles[just_title])

        # Categories: title + 100 tracks
        candidates = np.argwhere(playlist_lengths >= 150).squeeze()
        title_100 = np.random.choice(candidates, size=n_samples_category * 2, replace=False)

        title_random_100 = title_100[:n_samples_category]
        gt_random_100 = [self.split_playlist(p_node, 100, mode='random') for p_node in title_random_100]
        ground_truth.extend(gt_random_100)
        test_playlists.extend(title_random_100)
        titles.extend(self.playlist_titles[title_random_100])

        title_first_100 = title_100[n_samples_category:]
        gt_first_100 = [self.split_playlist(p_node, 100, mode='first') for p_node in title_first_100]
        ground_truth.extend(gt_first_100)
        test_playlists.extend(title_first_100)
        titles.extend(self.playlist_titles[title_first_100])

        # Categories: title + 25 tracks
        candidates = np.argwhere(playlist_lengths >= 38).squeeze()
        candidates = list(set(candidates.tolist()) - set(test_playlists))
        title_25 = np.random.choice(candidates, size=n_samples_category * 2, replace=False)

        title_random_25 = title_25[:n_samples_category]
        gt_random_25 = [self.split_playlist(p_node, 25, mode='random') for p_node in title_random_25]
        ground_truth.extend(gt_random_25)
        test_playlists.extend(title_random_25)
        titles.extend(self.playlist_titles[title_random_25])

        title_first_25 = title_25[n_samples_category:]
        gt_first_25 = [self.split_playlist(p_node, 25, mode='first') for p_node in title_first_25]
        ground_truth.extend(gt_first_25)
        test_playlists.extend(title_first_25)
        titles.extend(self.playlist_titles[title_first_25])

        # Categories: 1st 10 tracks (with/without title)
        candidates = np.argwhere(playlist_lengths >= 20).squeeze()
        candidates = list(set(candidates.tolist()) - set(test_playlists))
        first_10 = np.random.choice(candidates, size=n_samples_category * 2, replace=False)

        title_first_10 = first_10[:n_samples_category]
        ground_truth.extend([self.split_playlist(p_node, 10, mode='first') for p_node in title_first_10])
        test_playlists.extend(title_first_10)
        titles.extend(self.playlist_titles[title_first_10])

        no_title_first_10 = first_10[n_samples_category:]
        ground_truth.extend([self.split_playlist(p_node, 10, mode='first') for p_node in no_title_first_10])
        test_playlists.extend(no_title_first_10)
        titles.extend([None] * len(no_title_first_10))

        # Categories: 1st 5 tracks (with/without title)
        candidates = np.argwhere(playlist_lengths >= 10).squeeze()
        candidates = list(set(candidates.tolist()) - set(test_playlists))
        first_5 = np.random.choice(candidates, size=n_samples_category * 2, replace=False)

        title_first_5 = first_5[:n_samples_category]
        ground_truth.extend([self.split_playlist(p_node, 5, mode='first') for p_node in title_first_5])
        test_playlists.extend(title_first_5)
        titles.extend(self.playlist_titles[title_first_5])

        no_title_first_5 = first_5[n_samples_category:]
        ground_truth.extend([self.split_playlist(p_node, 5, mode='first') for p_node in no_title_first_5])
        test_playlists.extend(no_title_first_5)
        titles.extend([None] * len(no_title_first_5))

        # Category: title + 1st track
        candidates = np.argwhere(playlist_lengths >= 10).squeeze()
        candidates = list(set(candidates.tolist()) - set(test_playlists))
        title_first_1 = np.random.choice(candidates, size=n_samples_category, replace=False)
        test_playlists.extend(title_first_1)
        ground_truth.extend([self.split_playlist(p_node, 1, mode='first') for p_node in title_first_1])
        titles.extend(self.playlist_titles[title_first_1])

        return test_playlists, ground_truth, titles

    def split_train_test_title_only(self, n_test=10000, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        playlist_lengths = np.asarray([self.degree(p_node) for p_node in self.playlist_nodes])

        test_playlists = []
        ground_truth = []
        titles = []

        # Category: just title
        candidates = np.argwhere(playlist_lengths >= 10).squeeze()
        candidates = list(set(candidates.tolist()) - set(test_playlists))
        just_title = np.random.choice(candidates, size=n_test, replace=False)
        test_playlists.extend(just_title)
        ground_truth.extend([self.split_playlist(p_node, 0, mode='first') for p_node in just_title])
        titles.extend(self.playlist_titles[just_title])

        return test_playlists, ground_truth, titles

    def edge_indices(self, n_from, a_to):
        off = self._offsets[n_from]
        neighbors = self._adjacency[self._offsets[n_from]:self._offsets[n_from + 1]]
        return off + np.nonzero(np.isin(neighbors, a_to))

    def disable_test_connections(self, p_node, t_nodes):
        for t_node in t_nodes:
            self.set_enabled(p_node, t_node, False)
            self.set_enabled(t_node, p_node, False)

    def enable_all(self):
        self._enabled_edges = np.ones(len(self._adjacency), dtype=bool)

    def disable_node(self, node):
        neighbors = self.neighbors(node)
        for n in neighbors:
            self.set_enabled(n, node, False)
            self.set_enabled(node, n, False)

    def disable_nodes(self, nodes):
        for i, (start, end) in enumerate(zip(self._offsets[nodes], self._offsets[nodes + 1])):
            self._enabled_edges[start:end] = False
        self._enabled_edges[self.get_opposite_edges(np.nonzero(~self._enabled_edges)[0])] = False

    def enable_node(self, node):
        neighbors = self.neighbors(node)
        for n in neighbors:
            self.set_enabled(n, node, True)
            self.set_enabled(node, n, False)

    def set_enabled(self, a, b, enabled=True):
        """
        Set the enabled status of the connection from node a to node b
        :param a: node id a
        :param b: node id a
        :param enabled: bool, True for enabled, False for disabled
        :return: New status of the connection if there is an edge, None if there is no edge between the nodes
        """
        a_offset = self._offsets[a]
        a_neighbors = self.neighbors(a)
        try:
            en_index = a_offset + np.where(a_neighbors == b)[0]
            self._enabled_edges[en_index] = enabled
            return enabled
        except ValueError:
            return None

    def disable_mask(self, mask):
        self._enabled_edges = self._enabled_edges * mask

    def is_enabled(self, a, b, edge_mask=None):
        """
        Get the enabled/disabled status of the connection from node a to node b
        :param a: node id a
        :param b: node id b
        :param edge_mask: boolean array with same length of adjacency list, mask of enabled edges
        :return: True if enabled, False if disabled, None if there is no edge from a to b
        """
        a_offset = self._offsets[a]
        a_neighbors = self.neighbors(a, only_enabled=False)
        if edge_mask is None:
            edge_mask = self._enabled_edges
        try:
            en_index = a_offset + np.where(a_neighbors == b)[0][0]
            return edge_mask[en_index]
        except ValueError:
            return None

    def check_enabled_symmetry(self):
        assert (self._enabled_edges[
                    self.get_opposite_edges(np.nonzero(self._enabled_edges)[0])] == True).all(), "Positive not matching"
        assert (self._enabled_edges[self.get_opposite_edges(
            np.nonzero(~self._enabled_edges)[0])] == False).all(), "Negative not matching"

    def histogram_distance_prune(self, model_in: HistogramPlaylist, threshold: float):
        distances = l1_distance_multiple(model_in.histograms, self._playlist_histograms)
        disable_mask = (distances > threshold)
        num_kept = np.sum(~disable_mask)
        print("Keeping {} playlists.".format(num_kept))
        # Only prune when there will be at least 1000 playlists left
        if num_kept >= 1000:
            to_disable = np.nonzero(disable_mask)[0]
            self.disable_nodes(to_disable)
        return 1000000 - num_kept if num_kept >= 1000 else 0

    def histogram_distance_prune_top(self, model_in: HistogramPlaylist, num_keep: int = 100000):
        distances = l1_distance_multiple(model_in.histograms, self._playlist_histograms)
        d_sort = np.argsort(distances)
        to_disable = d_sort[num_keep:]

        print("Keeping {} playlists.".format(num_keep))
        self.disable_nodes(to_disable)
        return 1000000 - num_keep

    def degree(self, node, only_enabled=False, edge_mask=None):
        """
        Get the degree (number of neighbors) of a node
        :param node: int, node id
        :param only_enabled: bool, only count enabled edges
        :return: int, degree
        """
        if only_enabled:
            if edge_mask is None:
                edge_mask = self._enabled_edges
            if node < self.num_nodes - 1:
                indices = np.arange(self._offsets[node], self._offsets[node + 1])
                return np.sum(edge_mask[indices])
            else:
                indices = np.arange(self._offsets[node], len(self._adjacency))
                return np.sum(edge_mask[indices])
        else:
            if node < self.num_nodes - 1:
                return self._offsets[node + 1] - self._offsets[node]
            else:
                return len(self._adjacency) - self._offsets[node]

    def sample_neighbor_fast(self, node, edge_mask):
        min_id = self._offsets[node]
        try:
            max_id = self._offsets[node + 1]
        except IndexError:
            max_id = len(self._adjacency)
        n = self._adjacency[min_id:max_id][edge_mask[min_id: max_id]]
        try:
            return n[int(random.random() * len(n))]
        except IndexError:
            return None

    def neighbors_fast(self, node, edge_mask):
        min_id = self._offsets[node]
        try:
            max_id = self._offsets[node + 1]
        except IndexError:
            max_id = len(self._adjacency)
        mask = edge_mask[min_id: max_id]
        return self._adjacency[min_id:max_id][mask]

    def neighbors_flat(self, nodes, edge_mask):
        min_ids = self._offsets[nodes]
        max_ids = self._offsets[nodes + 1]
        counts = max_ids - min_ids
        counts_csum = counts.cumsum()
        id_arr = np.ones(counts_csum[-1], dtype=int)
        id_arr[0] = min_ids[0]
        id_arr[counts_csum[:-1]] = min_ids[1:] - max_ids[:-1] + 1
        id_arr = id_arr.cumsum()

        return self._adjacency[id_arr[edge_mask[id_arr]]]

    def neighbors_flat_nomask(self, nodes):
        min_ids = self._offsets[nodes]
        max_ids = self._offsets[nodes + 1]
        counts = max_ids - min_ids
        counts_csum = counts.cumsum()
        id_arr = np.ones(counts_csum[-1], dtype=int)
        id_arr[0] = min_ids[0]
        id_arr[counts_csum[:-1]] = min_ids[1:] - max_ids[:-1] + 1

        return self._adjacency[id_arr.cumsum()]

    def neighbors(self, node, only_enabled=False, unique=False, edge_mask=None):
        """
        Get the neigbors of a given node
        :param node: int, node id
        :param only_enabled: bool, if True only sample from neighbors connected with an enabled edge
        :return: np.ndarray of type int, node ids of neighbors
        """
        if edge_mask is None:
            edge_mask = self._enabled_edges

        if node < self.num_nodes - 1:
            neighbors = self._adjacency[self._offsets[node]: self._offsets[node + 1]]
            if only_enabled:
                neighbors = neighbors[edge_mask[self._offsets[node]: self._offsets[node + 1]]]
        else:
            neighbors = self._adjacency[self._offsets[node]: len(self._adjacency)]
            if only_enabled:
                neighbors = neighbors[edge_mask[self._offsets[node]: len(self._adjacency)]]

        if unique:
            return np.unique(neighbors)
        else:
            return neighbors

    def sample_neighbors(self, node, n=1, only_enabled=False, edge_mask=None):
        """
        Sample a random set of neighbors of a given node
        :param node: int, node id
        :param n: int, number of samples
        :param only_enabled: bool, if True only sample from neighbors connected with an enabled edge
        :return: Optional[np.ndarray] of type int, node ids of sampled neighbors. None if the number of sampled
                    neighbors is zero.
        """

        neighbors = self.neighbors(node, only_enabled=only_enabled, edge_mask=edge_mask)
        if len(neighbors) == 0:
            return None
        if n == 1:
            return choices(neighbors, k=n)[0]
        else:
            return choices(neighbors, k=n)

    def sample_neighbor(self, node, only_enabled=False, edge_mask=None):
        if edge_mask is None:
            edge_mask = self._enabled_edges
        if node < self.num_nodes - 1:
            neighbors = self._adjacency[self._offsets[node]: self._offsets[node + 1]]
            if only_enabled:
                neighbors = neighbors[edge_mask[self._offsets[node]: self._offsets[node + 1]]]
        else:
            neighbors = self._adjacency[self._offsets[node]: len(self._adjacency)]
            if only_enabled:
                neighbors = neighbors[edge_mask[self._offsets[node]: len(self._adjacency)]]
        if len(neighbors) > 0:
            return neighbors[int(random.random() * len(neighbors))]
        else:
            return None

    def add_edge(self, node_a, node_b):
        if node_a >= self.num_nodes or node_b >= self.num_nodes:
            raise ValueError("Cannot add edge between two nodes that do not exist")
        # Add B to A's adjacency list
        offset_a = self._offsets[node_a]
        self._adjacency = np.insert(self._adjacency, offset_a, node_b)
        self._offsets[node_a + 1:] += 1
        # Add A to B's adjacency list
        offset_b = self._offsets[node_b]
        self._adjacency = np.insert(self._adjacency, offset_b, node_a)
        self._offsets[node_b + 1:] += 1

    def is_valid_graph(self):
        for p_node in self.playlist_nodes:
            for t_node in self.neighbors(p_node):
                assert p_node in self.neighbors(t_node), \
                    "Track %i appears as neighbor in playlist %i, but not vice versa." % (t_node, p_node)

        for t_node in self.track_nodes:
            for p_node in self.neighbors(t_node):
                assert t_node in self.neighbors(p_node), \
                    "Playlist %i appears as neighbor for track %i, but nog vice versa." % (p_node, t_node)
        return True

    def make_degree_prune_mask(self, prune_factor=0.91):
        """
        Build a boolean mask that disables edges with high-degree pruning
        :param prune_factor: parameter controlling amount of pruning (also called delta)
        :return: boolean np.ndarray, prune mask
        """
        bins = np.linspace(0.0, 1.0, 11, endpoint=True)
        prune_mask = np.ones(len(self._adjacency), dtype=bool)

        t = tqdm(self.track_nodes, desc='d = . pruning .')
        for track in t:
            t_degree = self.degree(track)
            if t_degree > 2:
                new_degree = math.ceil(math.pow(t_degree, prune_factor))
                if new_degree < t_degree:
                    n_prune = t_degree - new_degree
                    t.set_description('d = {} pruning {}'.format(t_degree, n_prune))
                    t_features = self.features[track - 1000000]
                    nan_mask = np.isfinite(t_features)
                    binned_features = np.clip(np.digitize(t_features, bins) - 1, a_min=None, a_max=9)

                    t_playlists = self.neighbors(track)
                    p_models = self.playlist_histograms[t_playlists]  # Degree * Features * Bins
                    p_models = p_models[:, nan_mask, :]
                    likelihoods = p_models[:, np.arange(p_models.shape[1]), binned_features[nan_mask]]
                    p_scores = np.sum(np.log(likelihoods), axis=1)
                    score_sort = np.argsort(p_scores)[:n_prune]
                    to_disable = t_playlists[score_sort]
                    e_idxs = self.edge_indices(track, to_disable)
                    prune_mask[e_idxs] = False
                else:
                    t.set_description('d = {} pruning 0'.format(t_degree))
            else:
                t.set_description('d = {} pruning 0'.format(t_degree))
        prune_mask[self.get_opposite_edges(np.nonzero(~prune_mask)[0])] = False
        return prune_mask

    @property
    def playlist_nodes(self):
        """
        Get the node ids for all playlists
        :return: np.ndarray of type int, node ids
        """
        # First 1.000.000 nodes are playlists, corresponds exactly with playlist ids in MPD
        return np.arange(0, 1000000)

    @property
    def track_nodes(self):
        """
        Get the node ids for all tracks
        :return: np.ndarray of type int, node ids
        """
        # Nodes starting with id 1.000.000 are tracks
        return np.arange(1000000, self.num_nodes)

    @property
    def max_track_degree(self):
        return self._max_track_degree

    @property
    def num_nodes(self):
        return len(self._offsets) - 1

    @property
    def features(self):
        return self._features

    @property
    def track_node_ids(self):
        return self._track_node_ids

    @property
    def node_track_ids(self):
        return self._node_track_ids

    @property
    def name_index(self):
        return self._name_index

    @property
    def playlist_titles(self):
        return self._playlist_titles

    @property
    def edge_mask(self):
        return self._enabled_edges

    @property
    def playlist_histograms(self):
        return self._playlist_histograms

    @property
    def background_model(self):
        return self._background_model
