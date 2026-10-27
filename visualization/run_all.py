"""Run everything:  python -m visualization.run_all """
import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import N_RUNS, RESULTS                                       
from experiments import (exp_ds7_points, exp_fig5_propagation, exp_fig6_speed,   
                         exp_k_sweep, exp_comparison)
from visualization import plot_ds7, plot_curves, plot_comparison              


FAILED = []


def step(name, fn):
    t = time.time()
    print(f"\n{name}")
    try:
        fn()
        print(f"done in {time.time() - t:.1f}s")
    except Exception:
        FAILED.append(name)
        print(f"!!!!! {name} failed:")
        traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-experiments", action="store_true")
    ap.add_argument("--runs", type=int, default=N_RUNS)
    a = ap.parse_args()

    if not a.skip_experiments:
        step("Experiment: DS7 points (Tables 1-2)", exp_ds7_points.run)
        step("Experiment: Fig 5 propagation", exp_fig5_propagation.run)
        step("Experiment: K sweep (Fig 7)", exp_k_sweep.run)
        step("Experiment: speed (Fig 6)", exp_fig6_speed.run)
        step("Experiment: comparison (Tables 6-13)", lambda: exp_comparison.run(a.runs))

    step("Plots: DS7 (Tables 1-2, Figs 3-4)", plot_ds7.main)
    step("Plots: Figs 5-7", plot_curves.main)
    step("Plots: Tables 6-13, Figs 8-12, 14-15", plot_comparison.main)
    if FAILED:
        print("\nSTEPS THAT FAILED:")
        for n in FAILED:
            print("  -", n)
        sys.exit(1)
    print(f"\nAll outputs are in {RESULTS}")


if __name__ == "__main__":
    main()