import matplotlib.pyplot as plt
import numpy as np

from src.denmune import STRONG, WEAK, NOISE1, NOISE2, TYPE_NAMES
from src.utils import load_raw
from .plot_utils import scatter_labels, save_fig, table_png


def main():
    r = load_raw("ds7_points")
    X = r["X"]
    table_png(r["table1"], "table_01_ds7_points.png",
              f"Strong and weak points found by DenMune ({r['title']})")
    table_png(r["table2"].set_index("K"), "table_02_points_vs_K.png",
              f"Types of points vs K ({r['title']})")

    # Fig 3: point types -> skeleton after Phase I -> final result after Phase II
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    colors = {STRONG: "tab:blue", WEAK: "tab:orange", NOISE1: "black", NOISE2: "tab:red"}
    for t in (STRONG, WEAK, NOISE1, NOISE2):
        m = r["ptype"] == t
        if m.any():
            ax[0].scatter(X[m, 0], X[m, 1], s=4, c=colors[t], label=f"{TYPE_NAMES[t]} ({m.sum()})")
    ax[0].legend(markerscale=3, fontsize=8)
    ax[0].set_title("(a) Point types")
    scatter_labels(ax[1], X, r["labels1"], noise_color="lightgray", noise_marker=".")
    ax[1].set_title("(b) Phase I: skeleton built from strong points")
    scatter_labels(ax[2], X, r["labels"])
    ax[2].set_title("(c) Phase II: weak points merged, noise removed (x)")
    for a in ax[:1]:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Phases of DenMune ({r['title']}, K={r['k']})")
    save_fig(fig, "fig03_phases.png")

    # Fig 4: propagation snapshots
    snaps = r["snapshots"]
    its = sorted(snaps)
    cols = 3
    rows = int(np.ceil(len(its) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 4.3 * rows))
    ax = np.atleast_1d(ax).ravel()
    for a, it in zip(ax, its):
        lab = snaps[it]
        scatter_labels(a, X, lab, noise_color="lightgray", noise_marker=".")
        a.set_title(f"iteration {it}  ({lab.max() + 1} clusters)")
    for a in ax[len(its):]:
        a.axis("off")
    fig.suptitle("Cluster formation and propagation in DenMune (Phase I)")
    save_fig(fig, "fig04_propagation.png")