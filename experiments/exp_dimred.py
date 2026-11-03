"""EXP 3 (high-dimensional data): what does DenMune do on RAW features vs PCA / FA / NMF / t-SNE (2-D)?
A K-Means++ control on the SAME representations separates 'the reduction helps' from 'DenMune helps'."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis, NMF

from src.config import TABLES, SEED
from src.data_loader import available_datasets
from src.denmune import DenMune
from src.metrics import evaluate
from src.paper_reference import PAPER_DR
from src.utils import save_raw, n_clusters, noise_pct

K_SEARCH = [5, 10, 15, 20, 30, 40, 50]


def _reps(d):
    X, Z = d["X_raw"], d["X"]                       # Z = t-SNE 2-D cached by data_loader
    reps = {"Raw": X, "PCA-2": PCA(2, random_state=SEED).fit_transform(X)}
    try:
        reps["FA-2"] = FactorAnalysis(2, random_state=SEED).fit_transform(X)
    except Exception as e:
        print(f"   FA failed: {e}")
    try:
        reps["NMF-2"] = NMF(2, random_state=SEED, max_iter=500, init="nndsvda").fit_transform(X - X.min(0))
    except Exception as e:
        print(f"   NMF failed: {e}")
    reps["t-SNE-2"] = Z
    return reps


def run():
    rows, store = [], {}
    for d in available_datasets(labeled_only=True):
        if d["X_raw"].shape[1] <= 2:
            continue
        y, k_true, n = d["y"], len(np.unique(d["y"])), len(d["y"])
        print(f"[DimRed] {d['title']} (n={n}, D={d['X_raw'].shape[1]})")
        store[d["name"]] = dict(title=d["title"], y=y, emb={}, labels={})
        for rep, Z in _reps(d).items():
            best = None
            for k in [k for k in K_SEARCH if k < n]:
                lab = DenMune(k).fit(Z).labels_
                m = evaluate(y, lab)
                if best is None or m["NMI"] > best[0]["NMI"]:
                    best = (m, lab, k)
            km = max(evaluate(y, KMeans(k_true, n_init=1, random_state=SEED + r).fit_predict(Z))["NMI"]
                     for r in range(5))
            m, lab, k = best
            rows.append(dict(dataset=d["title"], representation=rep, DenMune_NMI=round(m["NMI"], 3),
                             DenMune_ARI=round(m["ARI"], 3), best_K=k, clusters=n_clusters(lab),
                             noise_pct=round(noise_pct(lab), 1), KMeansPP_NMI=round(km, 3),
                             paper_DenMune_NMI=PAPER_DR.get(d["name"], {}).get(rep.lower().split("-")[0]
                                                                           if rep != "Raw" else "raw", np.nan)))
            print(f"   {rep:8s} DenMune NMI={m['NMI']:.3f} (K={k}) | KM++ NMI={km:.3f}")
            if Z.shape[1] == 2:
                store[d["name"]]["emb"][rep], store[d["name"]]["labels"][rep] = Z, lab
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "exp3_dimred_nmi.csv", index=False)
    save_raw("dimred", dict(table=df, store=store))


if __name__ == "__main__":
    run()
