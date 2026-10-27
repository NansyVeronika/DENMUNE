import numpy as np
import pandas as pd

from src.baselines import ALGORITHMS
from src.config import TABLES, N_RUNS, SEED, DATASETS
from src.data_loader import available_datasets
from src.denmune import DenMune
from src.metrics import evaluate
from src.utils import save_raw, load_raw, raw_exists
from . import exp_k_sweep

ALGOS = list(ALGORITHMS)
SPEC = {("synthetic", "F1"): 6, ("real", "F1"): 7, ("synthetic", "NMI"): 8,
        ("real", "NMI"): 9, ("synthetic", "ARI"): 10, ("real", "ARI"): 11}


def _best(y, candidates):
    best = None
    for lab in candidates:
        m = evaluate(y, lab)
        if best is None or m["NMI"] > best[0]["NMI"]:
            best = (m, lab)
    return best


def build_tables(df):
    tables = {}
    for (kind, metric), num in SPEC.items():
        sub = df[df["kind"] == kind]
        if sub.empty:
            continue
        titles = [DATASETS[n]["title"] for n in DATASETS if DATASETS[n]["title"] in set(sub["dataset"])]
        piv = sub.pivot(index="dataset", columns="algo", values=metric)[ALGOS].reindex(titles)
        ranks = piv.rank(axis=1, ascending=False, method="min")
        total = ranks.sum()
        out = piv.round(2).astype(object)
        out.loc["Total Rank"] = total.astype(int)
        out.loc["Avg rank"] = ranks.mean().round(2)
        out.loc["Rank"] = total.rank(method="min").astype(int)
        out.index.name = "Dataset"
        name = f"table_{num:02d}_{metric}_{kind}"
        out.to_csv(TABLES / f"{name}.csv")
        tables[num] = dict(df=out, metric=metric, kind=kind)
        print(f"\nTable {num} ({metric}, {kind})\n{out.to_string()}")
    for kind, num in (("synthetic", 12), ("real", 13)):
        sub = df[(df["kind"] == kind) & (df["algo"] == "DenMune")]
        if sub.empty:
            continue
        t = sub.set_index("dataset")[["Homogeneity", "Completeness"]].T.round(2)
        t.index.name = "Score"
        t.to_csv(TABLES / f"table_{num}_homogeneity_completeness_{kind}.csv")
        tables[num] = dict(df=t, metric="Hom/Comp", kind=kind)
    return tables


def run(n_runs=N_RUNS):
    sweep = load_raw("k_sweep") if raw_exists("k_sweep") else exp_k_sweep.run()
    rows, store = [], {}
    for d in available_datasets(labeled_only=True):
        name, X, y = d["name"], d["X"], d["y"]
        k = len(np.unique(y))
        print(f"[Comparison] {d['title']} (n={len(X)}, k={k})")
        preds = {}
        for algo, fn in ALGORITHMS.items():
            try:
                if algo == "DenMune":
                    cands = [DenMune(sweep[name]["best_k"]).fit(X).labels_]
                else:
                    cands = fn(X, k, n_runs, SEED)
                m, lab = _best(y, cands)
                preds[algo] = lab
                rows.append(dict(dataset=d["title"], kind=d["kind"], algo=algo, **m))
                print(f"   {algo:9s} F1={m['F1']:.2f} NMI={m['NMI']:.2f} ARI={m['ARI']:.2f}")
            except Exception as e:
                print(f"   {algo:9s} FAILED: {e}")
                rows.append(dict(dataset=d["title"], kind=d["kind"], algo=algo, F1=np.nan, NMI=np.nan,
                                 ARI=np.nan, Homogeneity=np.nan, Completeness=np.nan))
        store[name] = dict(title=d["title"], kind=d["kind"], X=X, y=y, preds=preds)
    if not rows:
        print("[Comparison] no labeled datasets loaded - nothing to compare.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "all_scores_long.csv", index=False)
    save_raw("comparison", dict(results=df, store=store, tables=build_tables(df)))


if __name__ == "__main__":
    run()