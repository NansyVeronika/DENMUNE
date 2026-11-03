"""Figs 5-7: propagation curves, speed (with empirical complexity exponent), K-stability curves."""
import matplotlib.pyplot as plt
import numpy as np

from src.utils import load_raw, raw_exists
from .style import save, colour, DENMUNE_C


def main():
    if raw_exists("fig5_propagation"):
        r = load_raw("fig5_propagation"); h = r["history"]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(h["iterations"], h["n_clusters"], c=DENMUNE_C)
        ax[0].set(title=f"(a) clusters vs seed iteration - {r['title']}", xlabel="iteration", ylabel="# clusters")
        ax[1].plot(h["iterations"], h["n_clustered"], c=DENMUNE_C)
        ax[1].set(title="(b) clustered points vs iteration (Phase I)", xlabel="iteration", ylabel="# points")
        save(fig, "fig05_propagation")
    if raw_exists("fig6_speed"):
        r = load_raw("fig6_speed")
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        s, t = np.array(r["sizes"], float), np.array(r["size_total"])
        ax[0].plot(s, t, "o-", c=DENMUNE_C, label="total"); ax[0].plot(s, r["size_core"], "s--", c="#E69F00", label="without kNN search")
        ok = t > 0
        if ok.sum() > 2:
            slope = np.polyfit(np.log(s[ok]), np.log(t[ok]), 1)[0]
            ax[0].set_title(f"(a) time vs N  (empirical exponent ~ N^{slope:.2f})")
        ax[0].set(xlabel="N points", ylabel="seconds"); ax[0].legend()
        ax[1].plot(r["ks"], r["k_total"], "o-", c=DENMUNE_C)
        ax[1].set(title=f"(b) time vs K - {r['title']}", xlabel="K", ylabel="seconds")
        save(fig, "fig06_speed")
    if raw_exists("k_sweep"):
        sw = load_raw("k_sweep")
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, (n, r) in enumerate(sw.items()):
            ax.plot(r["df"].K, r["df"].NMI, c=colour(i), label=r["title"])
            ax.plot(r["best_k"], r["df"].NMI.max(), "o", c=colour(i), ms=5)
        ax.set(title="Fig 7 - NMI vs K (dot = best K)", xlabel="K", ylabel="NMI", ylim=(0, 1.02))
        ax.legend(fontsize=7, ncol=2)
        save(fig, "fig07_k_stability_nmi")
