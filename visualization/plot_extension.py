import matplotlib.pyplot as plt
import numpy as np

from src.utils import load_raw, raw_exists
from .plot_utils import scatter_labels, save_fig


def fig_a():
    if not raw_exists("ext_tsne_seeds"):
        return
    r = load_raw("ext_tsne_seeds")
    if not r:
        return
    fig, ax = plt.subplots(len(r), 3, figsize=(13, 3.6 * len(r)), squeeze=False)
    for row, v in zip(ax, r.values()):
        df = v["df"]
        row[0].boxplot([df.NMI, df.ARI], labels=["NMI", "ARI"])
        row[0].set_title(f"{v['title']} (K={v['k']}): quality over 10 t-SNE seeds")
        row[1].bar(df.seed, df.n_clusters, color="tab:blue")
        row[1].set(title="# clusters per seed", xlabel="t-SNE seed")
        row[2].bar(df.seed, df.noise_pct, color="tab:red")
        row[2].set(title=f"% noise per seed (mean pairwise ARI={np.mean(v['pair_ari']):.2f})", xlabel="t-SNE seed")
    fig.suptitle("Extension A - Sensitivity of the t-SNE + DenMune pipeline to the t-SNE random seed")
    fig.tight_layout()
    save_fig(fig, "ext_A_tsne_seed_sensitivity.png")

    # gallery: best and worst seed for each dataset
    fig, ax = plt.subplots(len(r), 2, figsize=(9, 4 * len(r)), squeeze=False)
    for row, v in zip(ax, r.values()):
        df = v["df"]
        for a, s, tag in ((row[0], int(df.NMI.idxmax()), "best"), (row[1], int(df.NMI.idxmin()), "worst")):
            scatter_labels(a, v["embs"][s], v["labs"][s], s=10)
            a.set_title(f"{v['title']} - {tag} seed {s} (NMI={df.NMI[s]:.2f}, {df.n_clusters[s]} clusters)")
    fig.tight_layout()
    save_fig(fig, "ext_A_best_vs_worst_seed.png")


def fig_b():
    if not raw_exists("ext_noise"):
        return
    r = load_raw("ext_noise")
    if not r:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for v in r.values():
        x = v["agg"].noise_frac * 100
        for a, col, t in zip(ax, ["ARI", "injected_detected_pct", "n_clusters"],
                             ["ARI on original points", "% injected noise flagged as noise", "# clusters found"]):
            a.errorbar(x, v["agg"][col], yerr=v["std"][col].fillna(0), marker="o", capsize=3,
                       label=f"{v['title']} (K={v['k']})")
            a.set(title=t, xlabel="% uniform noise added")
            a.grid(alpha=.3)
    ax[0].legend()
    fig.suptitle("Extension B - DenMune under increasing background noise")
    fig.tight_layout()
    save_fig(fig, "ext_B_noise_curves.png")

    for name, v in r.items():
        ex = v["examples"]
        fig, ax = plt.subplots(1, len(ex), figsize=(5 * len(ex), 4.3))
        for a, (f, (Xn, lab, n)) in zip(np.atleast_1d(ax), sorted(ex.items())):
            scatter_labels(a, Xn, lab, s=6)
            a.set_title(f"{int(f*100)}% noise added")
        fig.suptitle(f"DenMune on {v['title']} with added noise (x = detected noise)")
        save_fig(fig, f"ext_B_examples_{name}.png")


def fig_wholesale():
    if not raw_exists("ext_wholesale"):
        return
    r = load_raw("ext_wholesale")
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    scatter_labels(ax[0], r["Z_tsne"], r["lab_t"], s=14)
    ax[0].set_title(f"t-SNE + DenMune (K={r['k_t']})")
    scatter_labels(ax[1], r["Z_pca"], r["lab_p"], s=14)
    ax[1].set_title(f"PCA + DenMune (K={r['k_p']}) | ARI vs t-SNE run = {r['agree']:.2f}")
    prof = r["profile"].drop(columns="size")
    rel = np.log2(prof / prof.median().clip(lower=1e-9) .where(prof.median() > 0, 1))   # spend relative to overall median
    rel = np.log2((prof + 1) / (prof.median() + 1))
    im = ax[2].imshow(rel.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax[2].set_xticks(range(rel.shape[1]))
    ax[2].set_xticklabels(rel.columns, rotation=40, ha="right")
    ax[2].set_yticks(range(rel.shape[0]))
    ax[2].set_yticklabels([f"cluster {i} (n={int(r['profile']['size'].iloc[j])})" for j, i in enumerate(rel.index)])
    ax[2].set_title("Cluster spending profile (log2 vs median cluster)")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.suptitle("Extension - Customer segmentation on UCI Wholesale Customers")
    fig.tight_layout()
    save_fig(fig, "ext_wholesale_segmentation.png")


def main():
    fig_a()
    fig_b()
    fig_wholesale()