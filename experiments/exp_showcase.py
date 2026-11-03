"""EXP 1 (where DenMune excels): ground truth vs DenMune vs DBSCAN / K-Means++ / HDBSCAN / Spectral / FINCH
on natively 2-D datasets.  Also stores the strong / weak / noise anatomy (EXP D) for each dataset."""
import numpy as np
import pandas as pd

from src.baselines import ALGORITHMS, EXTRA_ALGORITHMS
from src.config import TABLES, N_RUNS, SEED, SHOWCASE
from src.data_loader import available_datasets
from src.denmune import DenMune
from src.metrics import evaluate
from src.utils import save_raw, best_k, n_clusters, noise_pct
from .exp_comparison import _best

SHOW_ALGOS = ["DenMune", "KM+", "DBSCAN", "HDBSCAN", "Spectral", "FINCH"]
FUNCS = {**ALGORITHMS, **EXTRA_ALGORITHMS}


def run(n_runs=N_RUNS):
    rows, store = [], {}
    for d in available_datasets(labeled_only=True):
        if d["X_raw"].shape[1] != 2:
            continue
        name, X, y = d["name"], d["X"], d["y"]
        k_true = len(np.unique(y))
        print(f"[Showcase] {d['title']} (n={len(X)}, true clusters={k_true})")
        preds, anatomy = {}, None
        for algo in SHOW_ALGOS:
            if algo == "DenMune":
                m = DenMune(best_k(name)).fit(X)
                lab, anatomy = m.labels_, dict(ptype=m.point_type_, counts=m.counts_, phase1=m.phase1_labels_,
                                              k=m.k)
                met = evaluate(y, lab)
            else:
                try:
                    met, lab = _best(y, FUNCS[algo](X, k_true, n_runs, SEED))
                except Exception as e:
                    print(f"   {algo} failed: {e}")
                    continue
            preds[algo] = lab
            rows.append(dict(dataset=d["title"], algo=algo, n_clusters=n_clusters(lab), true_clusters=k_true,
                             noise_pct=noise_pct(lab), **met))
            print(f"   {algo:9s} F1={met['F1']:.2f} NMI={met['NMI']:.2f} ARI={met['ARI']:.2f} "
                  f"clusters={n_clusters(lab)} noise={noise_pct(lab):.1f}%")
        store[name] = dict(title=d["title"], X=X, y=y, preds=preds, anatomy=anatomy)
    if rows:
        pd.DataFrame(rows).to_csv(TABLES / "exp1_showcase_scores.csv", index=False)
    save_raw("showcase", store)


if __name__ == "__main__":
    run()
