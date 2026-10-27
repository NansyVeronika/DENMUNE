import warnings

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, identity, diags, triu
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu, eigsh
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

def kmeanspp(X, k, runs, seed):
    for r in range(runs):
        yield KMeans(k, init="k-means++", n_init=1, random_state=seed + r).fit_predict(X)


def _maxmin(X, k, rng):
    c = [int(rng.integers(len(X)))]
    d = ((X - X[c[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        j = int(d.argmax())
        c.append(j)
        d = np.minimum(d, ((X - X[j]) ** 2).sum(1))
    return X[c]


def cbkm(X, k, runs, seed):
    rng = np.random.default_rng(seed)
    for _ in range(runs):
        yield KMeans(k, init=_maxmin(X, k, rng), n_init=1).fit_predict(X)


def random_swap(X, k, runs, seed, swaps=100):
    rng = np.random.default_rng(seed)
    n = len(X)
    for _ in range(runs):
        km = KMeans(k, init=X[rng.choice(n, k, replace=False)], n_init=1, max_iter=10).fit(X)
        C, best = km.cluster_centers_, km.inertia_
        for _ in range(swaps):
            C2 = C.copy()
            C2[rng.integers(k)] = X[rng.integers(n)]
            km2 = KMeans(k, init=C2, n_init=1, max_iter=2).fit(X)
            if km2.inertia_ < best:
                C, best = km2.cluster_centers_, km2.inertia_
        yield KMeans(k, init=C, n_init=1, max_iter=50).fit_predict(X)

def spectral(X, k, runs, seed):
    for r in range(min(runs, 5)):
        yield SpectralClustering(n_clusters=k, random_state=seed + r).fit_predict(X)


def hdbscan(X, k, runs, seed):
    from sklearn.cluster import HDBSCAN
    for m in (2, 3, 5, 8, 10, 15, 20, 30, 50, 100):      # min_cluster_size in [2, 100]
        if m < len(X):
            yield HDBSCAN(min_cluster_size=m).fit_predict(X)


def _finch_partitions(X):
    n, d = X.shape
    cur, lab, parts = X.copy(), np.arange(n), []
    while len(cur) > 2:
        m = len(cur)
        first = NearestNeighbors(n_neighbors=2).fit(cur).kneighbors(cur)[1][:, 1]
        A = csr_matrix((np.ones(m), (np.arange(m), first)), shape=(m, m))
        G = A + A.T + A @ A.T            # j=k(i) or i=k(j) or k(i)=k(j)
        nc, c = connected_components(G, directed=False)
        if nc == m:
            break
        lab = c[lab]
        parts.append(lab.copy())
        if nc == 1:
            break
        cur = np.zeros((nc, d))
        np.add.at(cur, lab, X)
        cur /= np.bincount(lab, minlength=nc)[:, None]
    return parts


def finch(X, k, runs, seed):
    parts = _finch_partitions(X)
    if not parts:
        yield np.zeros(len(X), int)
        return
    ncs = np.array([p.max() + 1 for p in parts])
    yield parts[int(np.abs(ncs - k).argmin())]          


def _fastdp(X, k, K):
    n = len(X)
    d, _ = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors(X)
    rho = 1.0 / (d[:, 1:].mean(1) + 1e-12)
    order = np.argsort(-rho, kind="stable")
    rank = np.empty(n, int)
    rank[order] = np.arange(n)
    delta = np.zeros(n)
    nh = np.full(n, -1)
    for s in range(0, n, 500):
        e = min(n, s + 500)
        D = cdist(X[s:e], X)
        D[rank[None, :] >= rank[s:e, None]] = np.inf    
        j = D.argmin(1)
        delta[s:e] = D[np.arange(e - s), j]
        nh[s:e] = j
    fin = np.isfinite(delta)
    top = order[0]
    delta[~fin] = delta[fin].max() if fin.any() else 1.0
    nh[~fin] = -1
    gamma = (rho / rho.max()) * (delta / max(delta.max(), 1e-12))
    gamma[top] = np.inf
    centers = np.argsort(-gamma)[:k]
    labels = np.full(n, -1)
    labels[centers] = np.arange(k)
    for i in order:
        if labels[i] < 0:
            labels[i] = labels[nh[i]] if nh[i] >= 0 else 0
    return labels


def fastdp(X, k, runs, seed):
    for K in (5, 10, 20, 40, 80):
        if K < len(X):
            yield _fastdp(X, k, K)


def npir(X, k, runs, seed, iters=10, knn=10):
    rng = np.random.default_rng(seed)
    n = len(X)
    knn = min(knn, n - 1)
    nbr = NearestNeighbors(n_neighbors=knn + 1).fit(X).kneighbors(X)[1][:, 1:]
    eye = np.eye(k)
    for ir in (0.01, 0.05, 0.10, 0.15, 0.20):
        for r in range(max(1, runs // 5)):
            lab = KMeans(k, init="k-means++", n_init=1, random_state=seed + r).fit_predict(X)
            for _ in range(iters):
                sel = rng.choice(n, max(1, int(ir * n)), replace=False)
                lab[sel] = eye[lab[nbr[sel]]].sum(1).argmax(1)
            yield lab

def _rcc(X, knn):
    n = len(X)
    idx = NearestNeighbors(n_neighbors=knn + 1).fit(X).kneighbors(X)[1][:, 1:]
    A = csr_matrix((np.ones(n * knn), (np.repeat(np.arange(n), knn), idx.ravel())), shape=(n, n))
    A = A.minimum(A.T)                                  
    Au = triu(A, 1).tocoo()
    i, j = Au.row, Au.col
    if len(i) == 0:
        return np.zeros(n, int)
    deg = np.asarray(A.sum(1)).ravel()
    w = deg.mean() / np.sqrt(np.maximum(deg[i] * deg[j], 1))
    eps2 = ((X[i] - X[j]) ** 2).sum(1)
    delta = np.sqrt(eps2).mean()
    W = coo_matrix((w, (i, j)), shape=(n, n))
    W = W + W.T
    L0 = diags(np.asarray(W.sum(1)).ravel()) - W
    lam = np.linalg.norm(X, 2) / max(eigsh(L0.tocsc(), k=1, which="LA")[0][0], 1e-12)
    U, mu, I = X.copy(), 3 * eps2.max(), identity(n, format="csc")
    for it in range(100):                                
        l = (mu / (mu + ((U[i] - U[j]) ** 2).sum(1))) ** 2
        Wl = coo_matrix((w * l, (i, j)), shape=(n, n))
        Wl = Wl + Wl.T
        Lap = diags(np.asarray(Wl.sum(1)).ravel()) - Wl
        U = splu((I + lam * Lap).tocsc()).solve(X)
        if (it + 1) % 4 == 0:
            mu = max(mu / 2, delta ** 2 / 2)
    keep = np.sqrt(((U[i] - U[j]) ** 2).sum(1)) < delta
    G = coo_matrix((np.ones(keep.sum()), (i[keep], j[keep])), shape=(n, n))
    return connected_components(G, directed=False)[1]


def rcc(X, k, runs, seed):
    for knn in (5, 10, 20):
        if knn < len(X):
            yield _rcc(X, knn)


ALGORITHMS = {            
    "DenMune": None,
    "NPIR": npir, "CBKM": cbkm, "RS": random_swap, "FastDP": fastdp, "FINCH": finch,
    "HDBSCAN": hdbscan, "RCC": rcc, "Spectral": spectral, "KM+": kmeanspp,
}