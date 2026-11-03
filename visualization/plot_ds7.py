"""Tables 1-2 and Figs 3-4 (strong/weak/noise anatomy and propagation of the skeleton on DS7)."""
import matplotlib.pyplot as plt
import numpy as np

from src.denmune import TYPE_NAMES
from src.utils import load_raw, raw_exists
from .style import render_table, save, scatter_labels, TYPE_C


def main():
    if not raw_exists("ds7_points"):
        print("  (no ds7_points results)"); return
    r = load_raw("ds7_points")
    X, pt, title = r["X"], r["ptype"], r["title"]
    render_table(r["table1"], "table_01_ds7_points", f"Table 1 - point types on {title} (K={r['k']})", index=False)
    render_table(r["table2"], "table_02_points_vs_K", "Table 2 - point types vs K", index=False)

    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for a, (t, txt) in zip(ax[:2], [(0, "(a) strong points"), (1, "(b) weak points")]):
        a.scatter(X[pt == t, 0], X[pt == t, 1], s=3, c=TYPE_C[t]); a.set_title(txt)
    scatter_labels(ax[2], X, r["labels"], "(c) final clusters (weak merged)", s=3)
    for t in (2, 3):
        ax[3].scatter(X[pt == t, 0], X[pt == t, 1], s=6, c=TYPE_C[t], marker="x", label=TYPE_NAMES[t])
    ax[3].set_title("(d) noise points"); ax[3].legend(fontsize=7)
    for a in ax:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    save(fig, "fig03_ds7_phases")

    snaps = r["snapshots"]
    if snaps:
        its = sorted(snaps)
        fig, ax = plt.subplots(1, len(its), figsize=(3.2 * len(its), 3.2))
        for a, it in zip(np.atleast_1d(ax), its):
            scatter_labels(a, X, snaps[it], f"seed iteration {it}", s=2)
        save(fig, "fig04_ds7_propagation")
