import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURES

PALETTE = plt.get_cmap("tab20").colors


def scatter_labels(ax, X, labels, s=6, noise_color="black", noise_marker="x"):
    labels = np.asarray(labels)
    for i, l in enumerate(np.unique(labels[labels >= 0])):
        m = labels == l
        ax.scatter(X[m, 0], X[m, 1], s=s, color=PALETTE[i % len(PALETTE)])
    m = labels < 0
    if m.any():
        ax.scatter(X[m, 0], X[m, 1], s=s, c=noise_color, marker=noise_marker, linewidths=0.6)
    ax.set_xticks([])
    ax.set_yticks([])


def save_fig(fig, filename):
    path = FIGURES / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def table_png(df, filename, title=""):
    d = df.copy().reset_index()
    d.columns = ["" if i == 0 and c == "index" else c for i, c in enumerate(d.columns)]
    cells = [[f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
             for row in d.values]
    fig, ax = plt.subplots(figsize=(1.15 * d.shape[1] + 1, 0.42 * (len(d) + 1) + 0.9))
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold")
    t = ax.table(cellText=cells, colLabels=list(d.columns), loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.4)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8e8e8")
    save_fig(fig, filename)