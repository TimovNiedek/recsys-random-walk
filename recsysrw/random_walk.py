import numpy as np
import random
from recsysrw import shared
from cachetools import LRUCache
import functools
import cachetools
import operator
from multiprocessing import Manager
from collections import Counter


def scaling_factor(query_track, pixie_weighting=False):
    track_degree = shared.graph.degree(query_track, only_enabled=True)
    if track_degree == 0:
        return 0
    else:
        if pixie_weighting:
            return track_degree * (shared.graph.max_track_degree - np.log(track_degree))
        else:
            return track_degree * (1 - np.log(track_degree) + np.log(shared.graph.max_track_degree))


def sum_visit_counts(visit_counts_list):
    result = {}
    for v in visit_counts_list:
        for track, visit_count in v.items():
            if track in result:
                result[track] += visit_count
            else:
                result[track] = visit_count
    return result


def multiply_visit_counts(visit_counts, multiplier):
    return {k: v * multiplier for (k, v) in visit_counts.items()}


def visit_counts_to_recommendations(visit_count, query_tracks):
    top_500 = [(key, value) for (key, value) in
               sorted(iter(visit_count.items()), key=lambda k_v: (k_v[1], k_v[0]))[::-1] if
               key not in query_tracks][:500]

    return np.array([key for (key, _) in top_500])


def normalize_visit_counts(vc):
    total = np.sum([c for (_, c) in vc.items()])
    if total > 0:
        normalized = {}
        for track, count in vc.items():
            normalized[track] = count / total
        return normalized
    else:
        return vc


class MultipleRandomWalk(object):
    cache_misses = 0
    total_steps = 0

    def __init__(self, alpha=0.1, n=100000, n_p=500, n_v=4, n_jobs=0,
                  use_boosting=False, dynamic_allocation=False, pixie_weighting=False,
                  cache_maxsize=2048):
        self._alpha = alpha
        self._n = n
        self._n_p = n_p
        self._n_v = n_v
        self._n_jobs = n_jobs
        self._use_boosting = use_boosting
        self._dynamic_allocation = dynamic_allocation
        self._pixie_weighting = pixie_weighting
        self._cache_maxsize = cache_maxsize
        self._playlist_model = None
        self._cache = Manager().dict()
        self._query_pid = None
        self._edge_mask = None
        self.cache = LRUCache(maxsize=cache_maxsize)

    def run_random_walk(self, query_tracks, playlist_model=None, pid=None, counts_per_node=False):
        # Clear the cache, we have a new model
        self._edge_mask = None
        self.cache.clear()
        self.total_steps = 0
        self.cache_misses = 0
        self._query_pid = pid

        self._playlist_model = playlist_model
        if self._n_jobs <= 0:
            return self.pixie_random_walk_multiple(query_tracks, counts_per_node=counts_per_node)
        else:
            return self.pixie_random_walk_multiple_parallel(query_tracks, counts_per_node=counts_per_node)

    @cachetools.cachedmethod(operator.attrgetter('cache'))
    def get_neighborhood(self, node):
        self.cache_misses += 1
        return shared.graph.neighbors_fast(node, self._edge_mask)

    @cachetools.cachedmethod(operator.attrgetter('cache'))
    def get_transition_probabilities(self, playlist, query_pid):
        self.cache_misses += 1
        neighbors = shared.graph.neighbors(playlist, only_enabled=True)
        if len(neighbors) > 0:
            track_features = shared.graph.features[neighbors - 1000000]
            scores = self._playlist_model.score(track_features)
            return scores, neighbors
        else:
            return None, None

    def get_transition_probabilities_nocache(self, playlist, query_pid):
        self.cache_misses += 1
        neighbors = shared.graph.neighbors(playlist, only_enabled=True)
        if len(neighbors) > 0:
            track_features = shared.graph.features[neighbors - 1000000]
            scores = self._playlist_model.score(track_features)
            return scores, neighbors
        else:
            return None, None

    def pixie_random_walk(self, q, edge_mask, n=None):
        self._edge_mask = edge_mask
        if n is None:
            n = self._n
        visit_count = Counter()
        total_steps = 0
        n_high_visited = 0

        while total_steps < n and n_high_visited < self._n_p:
            # Restart to the query track
            curr_track = q
            sample_steps = np.random.geometric(self._alpha)
            for i in range(sample_steps):
                curr_playlist = shared.graph.sample_neighbor_fast(curr_track, edge_mask)
                # Reached a dead end
                if curr_playlist is None:
                    if total_steps == 0:
                        # The query node does not have any connections
                        return Counter(), False
                    else:
                        break

                if self._playlist_model is None:
                    curr_track = shared.graph.sample_neighbor_fast(curr_playlist, edge_mask)
                else:
                    p, neighbors = self.get_transition_probabilities(curr_playlist, self._query_pid)
                    if neighbors is not None and len(neighbors) > 0:
                        curr_track = random.choices(neighbors, weights=p)[0]
                    else:
                        curr_track = None
                # Reached a dead end
                if curr_track is None:
                    break

                visit_count[curr_track] += 1

                if visit_count[curr_track] == self._n_v:
                    n_high_visited += 1

                total_steps += 1
                self.total_steps += 2
                if total_steps >= self._n or n_high_visited >= self._n_p:
                    break
        return visit_count, n_high_visited >= self._n_p

    def pixie_random_walk_multiple(self, query_tracks, counts_per_node=False):
        scaling_factors = [scaling_factor(q, self._pixie_weighting) for q in query_tracks]
        summed_scaling_factors = np.sum(scaling_factors)
        visit_counts = []
        boosted_visit_counts = {}
        total_random_walks = len(query_tracks)
        early_stopped_count = 0.0

        for q, s in zip(query_tracks, scaling_factors):
            n_q = self._n * s / summed_scaling_factors if self._dynamic_allocation else int(self._n / len(query_tracks))
            visit_q, early_stopped = self.pixie_random_walk(q, shared.graph.edge_mask, n=n_q)
            if early_stopped:
                early_stopped_count += 1.0
            visit_counts.append(visit_q)

        print("misses = {}, hits = {}, total steps = {}".format(self.cache_misses,
                                                                self.total_steps - self.cache_misses,
                                                                self.total_steps))

        if counts_per_node:
            return visit_counts, early_stopped_count, total_random_walks

        for v in visit_counts:
            for track, visit_count in v.items():
                if self._use_boosting:
                    if track in boosted_visit_counts:
                        boosted_visit_counts[track] += np.sqrt(visit_count)
                    else:
                        boosted_visit_counts[track] = np.sqrt(visit_count)
                else:
                    if track in boosted_visit_counts:
                        boosted_visit_counts[track] += visit_count
                    else:
                        boosted_visit_counts[track] = visit_count

        if self._use_boosting:
            boosted_visit_counts = {k: v ** 2 for k, v in boosted_visit_counts.items()}

        return boosted_visit_counts, early_stopped_count, total_random_walks

    def worker_random_walk(self, q, summed_scaling_factors, n, pid):
        self.cache.clear()
        if self._dynamic_allocation:
            n_q = n * scaling_factor(q, self._pixie_weighting) / summed_scaling_factors
        else:
            n_q = n
        edge_mask = np.frombuffer(shared.arr_enabled, dtype='bool')
        return self.pixie_random_walk(q, edge_mask, n=n_q)

    def pixie_random_walk_multiple_parallel(self, query_tracks, counts_per_node=False):
        scaling_factors = [scaling_factor(q, self._pixie_weighting) for q in query_tracks]
        summed_scaling_factors = np.sum(scaling_factors)
        boosted_visit_counts = {}
        total_random_walks = len(query_tracks)

        if self._dynamic_allocation:
            n_q = self._n
        else:
            n_q = int(self._n / len(query_tracks))

        self.cache.clear()
        results = shared.pool.map(functools.partial(self.worker_random_walk,
                                                    summed_scaling_factors=summed_scaling_factors,
                                                    n=n_q,
                                                    pid=self._query_pid), query_tracks)

        early_stopped_count = np.sum([int(e) for _, e in results])
        if counts_per_node:
            return [v for v, _ in results], early_stopped_count, total_random_walks

        for v, early_stopped in results:
            for track, visit_count in v.items():
                if self._use_boosting:
                    if track in boosted_visit_counts:
                        boosted_visit_counts[track] += np.sqrt(visit_count)
                    else:
                        boosted_visit_counts[track] = np.sqrt(visit_count)
                else:
                    if track in boosted_visit_counts:
                        boosted_visit_counts[track] += visit_count
                    else:
                        boosted_visit_counts[track] = visit_count

        if self._use_boosting:
            boosted_visit_counts = {k: v ** 2 for k, v in boosted_visit_counts.items()}

        return boosted_visit_counts, early_stopped_count, total_random_walks

