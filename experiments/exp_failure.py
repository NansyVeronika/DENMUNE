import numpy as np
import pandas as pd

from src.config import TABLES
from src.data_loader import available_datasets
from src.denmune import DenMune
from src.metrics import evaluate
from src.paper_reference import PAPER
from src.utils import save_raw, load_raw, raw_exists, best_k, n_clusters, noise_pct
from . import exp_comparison, exp_k_sweep


def _verdict(rank, total):
    return "excels" if rank == 1 else ("competitive" if rank <= 3 else "struggles")


def _diagnose(n, d_raw, k_true, k_found, noise, weak_share, rank):
    why = []
    if n < 250:
        why.append("small sample: kNN graph too sparse (paper Sec.4.3)")
    if d_raw > 20:
        why.append("high-D: neighbourhoods blurred before t-SNE")
    if k_found > k_true * 1.5:
        why.append(f"over-segmentation ({k_found} vs {k_true})")
    elif k_found < k_true:
        why.append(f"clusters merged ({k_found} vs {k_true})")
    if noise > 15:
        why.append(f"{noise:.0f}% points rejected as noise")
    if weak_share > 40:
        why.append(f"{weak_share:.0f}% weak points: overlapping / fuzzy borders")
    return "; ".join(why) if why else ("-" if rank <= 3 else "no single cause found - classes overlap in 2-D")


def run():
    sweep = load_raw("k_sweep") if raw_exists("k_sweep") else exp_k_sweep.run()
    if not raw_exists("comparison"):
        exp_comparison.run()
    comp = load_raw("comparison")["results"]
    rows = []
    for d in available_datasets(labeled_only=True):
        name, X, y, n = d["name"], d["X"], d["y"], len(d["X"])
        k_true, K = len(np.unique(y)), best_k(name)
        m = DenMune(K).fit(X)
        met = evaluate(y, m.labels_)
        c = m.counts_
        sw = sweep[name]["df"]
        sub = comp[comp.dataset == d["title"]].dropna(subset=["NMI"])
        rank = int(sub["NMI"].rank(ascending=False, method="min")[sub.algo == "DenMune"].iloc[0]) if len(sub) else 0
        others = sub[sub.algo != "DenMune"]
        bo = others.loc[others["NMI"].idxmax()] if len(others) else None
        weak_share = 100 * (c["weak"] + c["noise2"]) / n
        ref = PAPER.get(name, {})
        rows.append(dict(
            dataset=d["title"], kind=d["kind"], n=n, D=d["X_raw"].shape[1], true_k=k_true, K=K,
            F1=round(met["F1"], 3), NMI=round(met["NMI"], 3), ARI=round(met["ARI"], 3),
            clusters_found=m.n_clusters_, noise_pct=round(noise_pct(m.labels_), 1),
            strong=c["strong"], weak=c["weak"], noise1=c["noise1"], noise2=c["noise2"],
            NMI_std_over_K=round(float(sw.loc[sw.K >= 5, "NMI"].std()), 3), runtime_s=round(m.fit_time_, 3),
            rank_NMI=rank, best_baseline=None if bo is None else bo["algo"],
            best_baseline_NMI=None if bo is None else round(bo["NMI"], 3),
            paper_NMI=ref.get("NMI", np.nan), verdict=_verdict(rank, len(sub)),
            diagnosis=_diagnose(n, d["X_raw"].shape[1], k_true, m.n_clusters_, noise_pct(m.labels_), weak_share, rank)))
        print(f"[Failure] {d['title']:14s} NMI={met['NMI']:.2f} rank={rank} -> {rows[-1]['verdict']}: {rows[-1]['diagnosis']}")
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "exp4_failure_analysis.csv", index=False)
    save_raw("failure", df)


if __name__ == "__main__":
    run()
