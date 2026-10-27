"""Tables 6-13 as images, Figs 8-12 (algorithm visualizations), Figs 14-15 (ground truths)."""
import matplotlib.pyplot as plt
import numpy as np

from src.config import DATASETS
from src.data_loader import available_datasets
from src.utils import load_raw
from .plot_utils import scatter_labels, save_fig, table_png

ORDER = ["DenMune", "NPIR", "CBKM", "FINCH", "FastDP", "RS", "RCC", "HDBSCAN", "Spectral", "KM+"]
PRETTY = {"KM+": "KMeans++", "Spectral": "Spectral clustering"}
CAPTIONS = {n: "Table" for n in range(6, 14)}


def tables(r):
    for num, t in sorted(r["tables"].items()):
        table_png(t["df"], f"table_{num:02d}.png",
                  f"Table {num} - {t['metric']} ({t['kind']} datasets)")


def algorithm_figs(r):
    for name, s in r["store"].items():
        fig_no = DATASETS[name]["fig"]
        if fig_no is None:
            continue
        panels = [(f"{s['title']} dataset", s["y"])] + [(PRETTY.get(a, a), s["preds"][a])
                                                        for a in ORDER if a in s["preds"]]
        fig, ax = plt.subplots(3, 4, figsize=(15, 10))
        ax = ax.ravel()
        for i, (a, (title, lab)) in enumerate(zip(ax, panels)):
            scatter_labels(a, s["X"], lab, s=8)
            a.set_title(f"({chr(97 + i)}) {title}", fontsize=10)
        for a in ax[len(panels):]:
            a.axis("off")
        fig.suptitle(f"Fig. {fig_no} - Results of the ten algorithms on the {s['title']} dataset")
        save_fig(fig, f"fig{fig_no:02d}_{name}.png")


def ground_truths():
    for kind, fig_no, label in (("synthetic", 14, "synthetic"), ("real", 15, "real")):
        ds = [d for d in available_datasets(labeled_only=True) if d["kind"] == kind]
        if not ds:
            continue
        cols = 4
        rows = int(np.ceil(len(ds) / cols))
        fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows))
        ax = np.atleast_1d(ax).ravel()
        for i, (a, d) in enumerate(zip(ax, ds)):
            scatter_labels(a, d["X"], d["y"], s=6)
            a.set_title(f"({chr(97 + i)}) {d['title']}", fontsize=10)
        for a in ax[len(ds):]:
            a.axis("off")
        fig.suptitle(f"Fig. {fig_no} - Ground truths of the {label} datasets "
                     f"({'2-D' if kind == 'synthetic' else 't-SNE'} view)")
        save_fig(fig, f"fig{fig_no}_ground_truth_{label}.png")


def main():
    r = load_raw("comparison")
    tables(r)
    algorithm_figs(r)
    ground_truths()