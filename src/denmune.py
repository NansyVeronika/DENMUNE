"""DenMune (Abbas, El-Zoghabi, Shoukry - Pattern Recognition 2021), Algorithms 1-3.

Pipeline: KNN lists -> reverse lists (votes) -> MNN = KNN-> ∩ KNN<- ->
 noise type-1 (|MNN|=0) -> strong/seed (|KNN<-| >= K) vs weak ->
 Phase I  : seeds sharing seed-MNNs are merged into the cluster skeleton
 Phase II : weak points, one by one in canonical order, join the cluster with which
            they share the most MNN members; the rest is noise type-2.
labels_ == -1 means noise. Only parameter: K."""
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

STRONG, WEAK, NOISE1, NOISE2 = 0, 1, 2, 3
TYPE_NAMES = {STRONG: "Strong point", WEAK: "Weak point (merged)",
              NOISE1: "Noise type-1", NOISE2: "Noise type-2"}


def knn_lists(X, K):
    """(n, K) array: row i = K nearest neighbours of i (self excluded), sorted by distance."""
    n = len(X)
    _, idx = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors(X)
    is_self = idx == np.arange(n)[:, None]
    is_self[~is_self.any(axis=1), -1] = True          # duplicates: drop last column instead
    return idx[~is_self].reshape(n, K)


class DenMune:
    def __init__(self, k, track=False, snapshot_fracs=(), weak_passes=1):
        self.k = k
        self.track = track                      # record iteration history (Fig. 4 / Fig. 5)
        self.snapshot_fracs = snapshot_fracs
        self.weak_passes = weak_passes          # paper: a single pass

    # ------------------------------------------------------------------ fit
    def fit(self, X):
        t0 = time.perf_counter()
        X = np.asarray(X, dtype=float)
        n = len(X)
        K = int(max(1, min(self.k, n - 1)))

        t1 = time.perf_counter()
        idx = knn_lists(X, K)                                   # KNN_{p->}
        self.knn_time_ = time.perf_counter() - t1

        votes = np.bincount(idx.ravel(), minlength=n)           # |KNN_{p<-}|
        src = np.repeat(np.arange(n, dtype=np.int64), K)
        dst = idx.ravel().astype(np.int64)
        mutual = np.isin(dst * n + src, src * n + dst)          # j in KNN(i) and i in KNN(j)
        msrc, mdst = src[mutual], dst[mutual]
        mnn_count = np.bincount(msrc, minlength=n)
        indptr = np.concatenate(([0], np.cumsum(mnn_count)))    # CSR of the MNN lists

        noise1 = mnn_count == 0
        strong = ~noise1 & (votes >= K)                         # r >= 1
        weak = ~noise1 & ~strong                                # r < 1
        order = np.argsort(-votes, kind="stable")               # canonical ordering
        seed_order = order[strong[order]]

        # ---- Phase I: skeleton from seeds
        if self.track:
            labels1, self.history_, self.snapshots_ = self._skeleton_tracked(
                n, seed_order, indptr, mdst, strong)
        else:
            labels1 = self._skeleton_fast(n, strong, msrc, mdst)
            self.history_, self.snapshots_ = None, {}

        # ---- Phase II: weak points join clusters one by one
                # ---- Phase II: each weak point joins the cluster with which it shares the
        # most MNN-*seeds* (Phase I labels only); no seed in its MNN list -> noise type-2
        labels = labels1.copy()
        weak_order = order[weak[order]]
        for q in weak_order:
            l = labels1[mdst[indptr[q]:indptr[q + 1]]]       # seed labels only
            l = l[l >= 0]
            if l.size:
                labels[q] = np.bincount(l).argmax()

        ptype = np.full(n, STRONG)
        ptype[weak] = WEAK
        ptype[noise1] = NOISE1
        ptype[weak & (labels < 0)] = NOISE2

        self.labels_ = labels
        self.phase1_labels_ = labels1
        self.point_type_ = ptype
        self.n_clusters_ = int(labels.max() + 1) if (labels >= 0).any() else 0
        cnt = np.bincount(ptype, minlength=4)
        self.counts_ = {"strong": int(cnt[STRONG]), "weak": int(cnt[WEAK]),
                        "noise1": int(cnt[NOISE1]), "noise2": int(cnt[NOISE2])}
        self.fit_time_ = time.perf_counter() - t0
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_

    # ------------------------------------------------------- Phase I variants
    @staticmethod
    def _skeleton_fast(n, strong, msrc, mdst):
        keep = strong[msrc] & strong[mdst]                      # MNN restricted to seeds
        g = csr_matrix((np.ones(keep.sum()), (msrc[keep], mdst[keep])), shape=(n, n))
        _, comp = connected_components(g, directed=False)
        labels = np.full(n, -1)
        sid = np.flatnonzero(strong)
        labels[sid] = np.unique(comp[sid], return_inverse=True)[1]
        return labels

    def _skeleton_tracked(self, n, seed_order, indptr, mdst, strong):
        """Algorithm 2 literally: seeds processed in canonical order, each seed together with
        its seed-MNNs is merged with every existing cluster it intersects (union-find)."""
        parent = np.arange(n)
        touched = np.zeros(n, bool)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def labels_now():
            t = np.flatnonzero(touched)
            roots = np.array([find(v) for v in t])
            lab = np.full(n, -1)
            lab[t] = np.unique(roots, return_inverse=True)[1]
            return lab

        marks = {max(1, int(np.ceil(f * len(seed_order)))) for f in self.snapshot_fracs}
        n_clu = n_pts = 0
        hist_c, hist_p, snaps = [], [], {}
        for it, s in enumerate(seed_order, 1):
            nb = mdst[indptr[s]:indptr[s + 1]]
            nb = nb[strong[nb]]
            for v in np.concatenate(([s], nb)):
                if not touched[v]:
                    touched[v] = True
                    n_pts += 1
                    n_clu += 1
            rs = find(s)
            for v in nb:
                rv = find(v)
                if rv != rs:
                    parent[rv] = rs
                    n_clu -= 1
            hist_c.append(n_clu)
            hist_p.append(n_pts)
            if it in marks:
                snaps[it] = labels_now()
        hist = {"iterations": np.arange(1, len(seed_order) + 1),
                "n_clusters": np.array(hist_c), "n_clustered": np.array(hist_p)}
        return labels_now(), hist, snaps