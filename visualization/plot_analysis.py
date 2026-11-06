import matplotlib.pyplot as plt
import numpy as np

from src.denmune import TYPE_NAMES
from src.metrics import evaluate
from src.utils import load_raw, raw_exists
from .style import (save, scatter_labels, render_table, colour, TYPE_C, DENMUNE_C, OTHER_C, PAPER_C)


def showcase():
    if not raw_exists("showcase"):
        return
    st = load_raw("showcase")
    if not st:
        return
    algos = ["Ground truth", "DenMune", "KM+", "DBSCAN", "HDBSCAN", "Spectral", "FINCH"]
    fig, ax = plt.subplots(len(st), len(algos), figsize=(2.6 * len(algos), 2.6 * len(st)), squeeze=False)
    for r, (n, s) in enumerate(st.items()):
        for c, a in enumerate(algos):
            lab = s["y"] if a == "Ground truth" else s["preds"].get(a)
            if lab is None:
                ax[r, c].axis("off"); continue
            t = a if a == "Ground truth" else f"{a}  NMI={evaluate(s['y'], lab)['NMI']:.2f}"
            scatter_labels(ax[r, c], s["X"], lab, (s["title"] + "\n" if c == 0 else "") + t, s=3)
    save(fig, "exp1_showcase_grid")
    # anatomy
    fig, ax = plt.subplots(1, len(st), figsize=(3.4 * len(st), 3.6), squeeze=False)
    for a, (n, s) in zip(ax[0], st.items()):
        pt = s["anatomy"]["ptype"]
        for t in (0, 1, 2, 3):
            a.scatter(s["X"][pt == t, 0], s["X"][pt == t, 1], s=5 if t < 2 else 14, c=TYPE_C[t],
                      marker="o" if t < 2 else "x", label=f"{TYPE_NAMES[t]} ({(pt == t).sum()})")
        a.set_title(f"{s['title']} (K={s['anatomy']['k']})"); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    save(fig, "expD_point_types")


def failure():
    if not raw_exists("failure"):
        return
    df = load_raw("failure")
    render_table(df[["dataset", "n", "D", "NMI", "paper_NMI", "clusters_found", "true_k", "noise_pct", "rank_NMI",
                     "verdict", "diagnosis"]], "exp4_failure_table", "Exp 4 - where DenMune works / struggles", fs=7, index=False)
    d = df.sort_values("rank_NMI")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(d)); w = 0.27
    ax.bar(x - w, d.NMI, w, color=DENMUNE_C, label="DenMune (ours)")
    ax.bar(x, d.best_baseline_NMI, w, color=OTHER_C, label="best of 9 baselines (ours)")
    ax.bar(x + w, d.paper_NMI, w, color=PAPER_C, label="DenMune (paper)")
    ax.set_xticks(x, [f"{t}\nrank {r}" for t, r in zip(d.dataset, d.rank_NMI)], fontsize=7)
    ax.set(ylabel="NMI", title="NMI: ours vs best baseline vs paper (sorted by our rank)", ylim=(0, 1.05)); ax.legend(fontsize=7)
    save(fig, "exp4_nmi_vs_baselines")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for t, c, lab in (("strong", TYPE_C[0], "strong"), ("weak", TYPE_C[1], "weak"), ("noise1", TYPE_C[2], "noise 1"), ("noise2", TYPE_C[3], "noise 2")):
        pass
    parts = df[["strong", "weak", "noise1", "noise2"]].div(df.n, axis=0) * 100
    bot = np.zeros(len(df))
    for col, c in zip(parts.columns, [TYPE_C[0], TYPE_C[1], TYPE_C[2], TYPE_C[3]]):
        ax.bar(df.dataset, parts[col], bottom=bot, color=c, label=col, edgecolor="white", linewidth=1); bot += parts[col].values
    ax.set(ylabel="% of points", title="Point-type mix per dataset"); ax.legend(fontsize=7, ncol=4)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "exp4_point_type_mix")


def kstab():
    if not raw_exists("k_stability"):
        return
    r = load_raw("k_stability"); g = r["grid"]
    M = np.array([g[t]["NMI"].reindex(list(g[next(iter(g))].index)).values for t in g], float)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(g) + 1.8))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(M.shape[1]), g[next(iter(g))].index); ax.set_yticks(range(len(g)), list(g)); ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > .6 else "black")
    ax.set(xlabel="K", title="NMI on the K grid"); fig.colorbar(im, label="NMI")
    save(fig, "expA_k_grid_heatmap")
    s = r["summary"]
    fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(len(s)); w = .38
    ax.bar(x - w / 2, s.best_NMI, w, color=DENMUNE_C, label="K tuned with labels (oracle)")
    ax.bar(x + w / 2, s.NMI_at_fixed_K, w, color=PAPER_C, label=f"fixed K={int(s.fixed_K.iloc[0])} (no labels)")
    ax.set_xticks(x, s.dataset, rotation=30, ha="right"); ax.set(ylabel="NMI", title="Cost of not tuning K", ylim=(0, 1.05)); ax.legend(fontsize=7)
    save(fig, "expA_oracle_vs_fixed_k")


def dimred():
    if not raw_exists("dimred"):
        return
    r = load_raw("dimred"); df = r["table"]
    for ds, g in df.groupby("dataset", sort=False):
        fig, ax = plt.subplots(figsize=(7.5, 3.6)); x = np.arange(len(g)); w = .38
        ax.bar(x - w / 2, g.DenMune_NMI, w, color=DENMUNE_C, label="DenMune")
        ax.bar(x + w / 2, g.KMeansPP_NMI, w, color=OTHER_C, label="K-Means++ (control)")
        ax.plot(x, g.paper_DenMune_NMI, "D", c=PAPER_C, label="paper (DenMune)")
        ax.set_xticks(x, g.representation); ax.set(ylabel="NMI", ylim=(0, 1.05), title=f"{ds}: input representation matters"); ax.legend(fontsize=7)
        save(fig, f"exp3_dimred_{ds.lower().replace(' ', '_')}")
    for n, s in r["store"].items():
        reps = list(s["emb"])
        fig, ax = plt.subplots(1, len(reps), figsize=(3.2 * len(reps), 3.2), squeeze=False)
        for a, rep in zip(ax[0], reps):
            scatter_labels(a, s["emb"][rep], s["labels"][rep], f"{s['title']} - {rep}", s=4)
        save(fig, f"exp3_embeddings_{n}")


def paper():
    if not raw_exists("paper_vs_ours"):
        return
    r = load_raw("paper_vs_ours"); df = r["table"]
    if df.empty:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for a, m in zip(ax, ("F1", "NMI", "ARI")):
        a.plot([0, 1], [0, 1], c="#bbbbbb", ls="--")
        a.scatter(df[f"{m}_paper"], df[f"{m}_ours"], c=DENMUNE_C, s=28)
        for _, row in df.iterrows():
            a.annotate(row.dataset, (row[f"{m}_paper"], row[f"{m}_ours"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        a.set(xlabel=f"paper {m}", ylabel=f"our {m}", title=f"{m}: our reproduction vs paper", xlim=(0, 1.05), ylim=(0, 1.05))
    save(fig, "paper_vs_ours_scatter")
    render_table(df[["dataset", "F1_ours", "F1_paper", "NMI_ours", "NMI_paper", "ARI_ours", "ARI_paper",
                     "best_other_F1_ours", "best_other_F1_paper"]], "paper_vs_ours_table", "Our results vs paper (DenMune)", index=False)


def main():
    for f in (showcase, failure, kstab, dimred, paper):
        try:
            f()
        except Exception as e:
            print(f"  ! {f.__name__} plot failed: {e}")
