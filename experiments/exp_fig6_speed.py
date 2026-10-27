import numpy as np

from src.config import DS7_K, SEED
from src.data_loader import ds7_or_largest
from src.denmune import DenMune
from src.utils import save_raw


def _time(X, k, repeats):
    core, tot = [], []
    for _ in range(repeats):
        m = DenMune(k).fit(X)
        core.append(m.fit_time_ - m.knn_time_)
        tot.append(m.fit_time_)
    return float(np.median(core)), float(np.median(tot))


def run(repeats=3):
    d = ds7_or_largest()
    X = d["X"][np.random.default_rng(SEED).permutation(len(d["X"]))]
    n = len(X)
    sizes = list(range(1000, n + 1, 1000)) if n >= 2000 else list(np.linspace(n // 10, n, 10, dtype=int))
    if sizes[-1] != n:
        sizes.append(n)
    a = [_time(X[:m], min(DS7_K, m - 1), repeats) for m in sizes]
    ks = [k for k in [1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200] if k < n]
    b = [_time(X, k, repeats) for k in ks]
    print(f"[Fig 6] timed on {d['title']} (n={n})")
    save_raw("fig6_speed", dict(title=d["title"], sizes=sizes, size_core=[t[0] for t in a],
                                size_total=[t[1] for t in a], ks=ks,
                                k_core=[t[0] for t in b], k_total=[t[1] for t in b]))


if __name__ == "__main__":
    run()