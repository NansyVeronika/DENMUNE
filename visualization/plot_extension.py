
import matplotlib.pyplot as plt
import numpy as np

from src.config import SEED
from src.utils import load_raw, raw_exists
from .style import save, scatter_labels, colour, DENMUNE_C


def _seeds():
    if not raw_exists("ext_tsne_seeds"):
        return
    for name, r in load_raw("ext_tsne_seeds").items():
        df = r["df"]
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].bar(df.seed, df.NMI, color=DENMUNE_C)
        ax[0].set(title=f"{r['title']}: NMI per t-SNE seed (mean {df.NMI.mean():.2f}+-{df.NMI.std():.2f})", xlabel="seed", ylim=(0, 1))
        ax[1].hist(r["pair_ari"], bins=12, color=DENMUNE_C)
        ax[1].set(title="pairwise ARI between seeds", xlabel="ARI")
        b, w = int(df.NMI.idxmax()), int(df.NMI.idxmin())
        scatter_labels(ax[2], r["embs"][b], r["labs"][b], f"best seed {b} (NMI {df.NMI[b]:.2f}) | worst {w}: {df.NMI[w]:.2f}")
        save(fig, f"ext_A_tsne_seeds_{name}")


def _noise():
    if not raw_exists("ext_noise"):
        return
    for name, r in load_raw("ext_noise").items():
        a, sd = r["agg"], r["std"]
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
        for m, c in (("ARI", colour(0)), ("NMI", colour(1)), ("F1", colour(2))):
            ax[0].errorbar(a.noise_frac * 100, a[m], sd[m].fillna(0), marker="o", c=c, label=m, capsize=2)
        ax[0].set(title=f"{r['title']}: quality (original points only)", xlabel="% injected noise", ylim=(0, 1.02)); ax[0].legend()
        ax[1].plot(a.noise_frac * 100, a.injected_detected_pct, "o-", c=colour(0), label="recall (injected flagged)")
        if "noise_precision_pct" in a:
            ax[1].plot(a.noise_frac * 100, a.noise_precision_pct, "s-", c=colour(1), label="precision")
        ax[1].set(title="noise detection", xlabel="% injected noise", ylabel="%"); ax[1].legend()
        ax[2].plot(a.noise_frac * 100, a.original_lost_pct, "o-", c=colour(5))
        ax[2].set(title="real points wrongly rejected", xlabel="% injected noise", ylabel="%")
        save(fig, f"ext_B_noise_{name}")
        ex = r["examples"]
        if ex:
            fig, axs = plt.subplots(1, len(ex), figsize=(4 * len(ex), 3.6))
            for axx, (f, (Xn, lab, n)) in zip(np.atleast_1d(axs), sorted(ex.items())):
                scatter_labels(axx, Xn, lab, f"{r['title']} + {int(f*100)}% noise", s=5)
            save(fig, f"ext_B_noise_examples_{name}")


def _wholesale():
    if not raw_exists("ext_wholesale"):
        return
    r = load_raw("ext_wholesale")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    scatter_labels(ax[0], r["Z_tsne"], r["lab_t"], f"t-SNE + DenMune (K={r['k_t']})", s=10)
    scatter_labels(ax[1], r["Z_pca"], r["lab_p"], f"PCA + DenMune (K={r['k_p']})  ARI vs t-SNE={r['agree']:.2f}", s=10)
    save(fig, "ext_wholesale_clusters")
    p = r["profile"].drop(columns="size")
    z = (p - p.mean()) / p.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(7, 0.7 * len(p) + 1.5))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(p.shape[1]), p.columns, rotation=30); ax.set_yticks(range(len(p)), [f"C{i} (n={n})" for i, n in zip(p.index, r["profile"]["size"])])
    ax.grid(False); fig.colorbar(im, label="median spend (z across clusters)")
    ax.set_title("Cluster spending profiles")
    save(fig, "ext_wholesale_profiles")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, c in (("sweep_t", colour(0)), ("sweep_p", colour(1))):
        ax[0].plot(r[k].K, r[k].silhouette, "o-", c=c, label=k[-1] + "-SNE" if k == "sweep_t" else "PCA")
        ax[1].plot(r[k].K, r[k].noise_pct, "o-", c=c)
    ax[0].set(title="label-free K choice: silhouette", xlabel="K"); ax[0].legend()
    ax[1].set(title="noise % vs K", xlabel="K")
    save(fig, "ext_wholesale_ksweep")


def main():
    _seeds(); _noise(); _wholesale()
