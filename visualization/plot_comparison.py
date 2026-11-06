import matplotlib.pyplot as plt
import numpy as np

from src.config import DATASETS
from src.utils import load_raw, raw_exists
from .style import render_table, save, scatter_labels
from src.metrics import evaluate


def main():
    if not raw_exists("comparison"):
        print("  (no comparison results)"); return
    r = load_raw("comparison")
    for num, t in sorted(r["tables"].items()):
        render_table(t["df"], f"table_{num:02d}_{t['metric'].replace('/', '_')}_{t['kind']}",
                     f"Table {num} - {t['metric']} ({t['kind']} datasets)")
    for name, s in r["store"].items():
        fig_no = DATASETS[name]["fig"]
        if fig_no is None or s["X"].shape[1] != 2:
            continue
        algos = list(s["preds"])
        cols = 4
        fig, ax = plt.subplots(int(np.ceil((len(algos) + 1) / cols)), cols, figsize=(3.2 * cols, 3.0 * np.ceil((len(algos) + 1) / cols)))
        ax = ax.ravel()
        scatter_labels(ax[0], s["X"], s["y"], f"{s['title']} (ground truth)")
        for a, algo in zip(ax[1:], algos):
            m = evaluate(s["y"], s["preds"][algo])
            scatter_labels(a, s["X"], s["preds"][algo], f"{algo}  F1={m['F1']:.2f}")
        for a in ax[len(algos) + 1:]:
            a.axis("off")
        save(fig, f"fig{fig_no:02d}_{name}_all_algorithms")
    for kind, no in (("synthetic", 14), ("real", 15)):
        items = [(n, s) for n, s in r["store"].items() if s["kind"] == kind]
        if not items:
            continue
        cols = min(4, len(items)); rows = int(np.ceil(len(items) / cols))
        fig, ax = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
        for a, (n, s) in zip(ax.ravel(), items):
            scatter_labels(a, s["X"], s["y"], s["title"], s=4)
        for a in ax.ravel()[len(items):]:
            a.axis("off")
        save(fig, f"fig{no}_{kind}_ground_truth")
