import numpy as np
from enum import IntEnum


class Feature(IntEnum):
    acousticness = 0
    danceability = 1
    energy = 2
    explicit = 3
    instrumentalness = 4
    liveness = 5
    speechiness = 6
    tempo = 7
    valence = 8
    count = 9
    year = 10


def kl_divergence(p, q):
    """
    Kullback-Leibler divergence D(P || Q) for discrete distributions

        Parameters
        ----------
        p, q : array-like, dtype=float, shape=k
            Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where((p != 0) & (q != 0), p * np.log(p / q), 0))


def kl_divergence_multiple(p, Q):
    """
    Kullback-Leibler divergence D(p || q) for discrete distributions between p and all q in Q.

        Parameters
        ----------
        p: array-like, dtype=float, shape=k
            Discrete probability distribution.

        q : array-like, dtype=float, shape=(n, k)
            Discrete probability distributions.
    """
    print(p.shape, Q.shape)
    return np.sum(p*np.log(p / Q), axis=(1, 2))


def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def jensen_shannon_divergence_multiple(p, Q):
    m = 0.5 * (p + Q)
    print(m.shape)
    return 0.5 * (kl_divergence_multiple(p, m) + kl_divergence_multiple(Q, m))


def l1_distance_multiple(p, Q):
    return np.sum(np.abs(p - Q), axis=(1, 2))


class HistogramPlaylist(object):
    def __init__(self, features, playlist_id, bins=10, smoothing=0.01, background_model=None):
        self._bins = np.linspace(0.0, 1.0, bins+1, endpoint=True)
        self._n_bins = bins
        self._playlist_id = playlist_id
        self._smoothing = smoothing
        self._background_model = background_model
        self._histograms = self.fit(features)

    def fit(self, features):
        _histograms = np.zeros((features.shape[1], self._n_bins), dtype=np.float)
        for i in range(features.shape[1]):
            f = features[:, i]
            hist, bin_edges = np.histogram(f[~np.isnan(f)], range=(0.0, 1.0), bins=self._n_bins)
            _histograms[i] = hist
        if self._smoothing > 0:
            _histograms += np.ones((features.shape[1], self._n_bins)) * self._smoothing
        _histograms = _histograms / np.sum(_histograms, axis=1, keepdims=True)
        if self._background_model is not None:
            _histograms = _histograms * (1-self._smoothing) + self._background_model * self._smoothing

        return _histograms


    def score(self, X):
        """
        :param X: np.ndarray
                  shape (N, F), N = number of tracks to score, F = number of features
        :return:
        """
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(col_mean))
        col_mean[inds] = 0
        inds = np.where(np.isnan(X))
        X_nonan = np.copy(X)
        X_nonan[inds] = np.take(col_mean, inds[1])
        X_bins = np.clip((X_nonan * self._n_bins).astype(int), a_min=0, a_max=self._n_bins-1)  # shape: (N, F)
        bin_scores = np.take(self._histograms.T, X_bins)  # shape: (N, F)
        # bin_scores might contain zeros if we don't smooth
        track_scores = np.prod(bin_scores, axis=1)  # shape: N
        return track_scores / np.sum(track_scores)  # shape: N


    @property
    def histograms(self):
        return self._histograms

    @property
    def playlist_id(self):
        return self._playlist_id
