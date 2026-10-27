import matplotlib.pyplot as plt

from src.utils import load_raw
from .plot_utils import save_fig


def fig5():
    r = load_raw("fig5_propagation")
    h = r["history"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["iterations"], h["n_clusters"], color="k")
    ax[0].set(xlabel="Iteration", ylabel="Number of clusters", title="(a) Number of clusters")
    ax[1].plot(h["iterations"], h["n_clustered"], color="k")
    ax[1].set(xlabel="Iteration", ylabel="Number of clustered points", title="(b) Clustered data points")
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle(f"DenMune cluster formation in Phase I ({r['title']}, K={r['k']}); ")
    save_fig(fig, "fig05_convergence.png")


def fig6():
    r = load_raw("fig6_speed")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(r["sizes"], r["size_core"], "o-", label="core clustering")
    ax[0].plot(r["sizes"], r["size_total"], "s--", label="incl. kNN search")
    ax[0].set(xlabel="Number of points", ylabel="Time (s)", title="(a) Speed vs data size")
    ax[1].plot(r["ks"], r["k_core"], "o-", label="core clustering")
    ax[1].plot(r["ks"], r["k_total"], "s--", label="incl. kNN search")
    ax[1].set(xlabel="K (nearest neighbours)", ylabel="Time (s)", title="(b) Speed vs K")
    for a in ax:
        a.grid(alpha=.3)
        a.legend()
    fig.suptitle(f" Speed of DenMune ({r['title']})")
    save_fig(fig, "fig06_speed.png")


def fig7():
    r = load_raw("k_sweep")
    groups = [("real", "Real datasets"), ("synthetic", "Synthetic datasets")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for a, (kind, title) in zip(ax, groups):
        for v in r.values():
            if v["kind"] == kind:
                a.plot(v["df"]["K"], v["df"]["NMI"], label=v["title"])
        a.set(xlabel="K", ylabel="NMI", title=title, ylim=(0, 1.02))
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.suptitle(" DenMune stability over changes in K (NMI)")
    save_fig(fig, "fig07_stability.png")


def main():
    fig5()
    fig6()
    fig7()