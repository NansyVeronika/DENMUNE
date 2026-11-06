import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURES

OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
NOISE_C = "#9a9a9a"
TYPE_C = {0: "#0072B2", 1: "#E69F00", 2: "#D55E00", 3: "#CC79A7"}
TYPE_N = {0: "Strong (seed)", 1: "Weak (merged)", 2: "Noise type-1", 3: "Noise type-2"}
DENMUNE_C, OTHER_C, PAPER_C = "#0072B2", "#9a9a9a", "#E69F00"

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 10, "axes.labelsize": 9,
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})


def colour(i):
    return OKABE[i % 8] if i < 8 else plt.cm.tab20(i % 20)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{name}.png")


def scatter_labels(ax, X, lab, title="", s=6):
    lab = np.asarray(lab)
    m = lab >= 0
    if (~m).any():
        ax.scatter(X[~m, 0], X[~m, 1], s=s, c=NOISE_C, marker="x", linewidths=0.6)
    if m.any():
        ids = np.unique(lab[m])
        cols = [colour(i) for i in range(len(ids))]
        idx = np.searchsorted(ids, lab[m])
        ax.scatter(X[m, 0], X[m, 1], s=s, c=[cols[i] for i in idx])
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def render_table(df, name, title="", fs=8, index=True):
    d = df.reset_index() if index else df
    fig, ax = plt.subplots(figsize=(max(5, 0.95 * len(d.columns)), 0.34 * (len(d) + 2) + 0.5))
    ax.axis("off")
    t = ax.table(cellText=d.astype(str).values, colLabels=[str(c) for c in d.columns], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, 1.25)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor("#0072B2"); cell.set_text_props(color="white", weight="bold")
    if title:
        ax.set_title(title, fontsize=10, weight="bold")
    save(fig, name)
