import numpy as np
import pandas as pd

from src.config import TABLES, K_GRID, FIXED_K
from src.utils import save_raw, load_raw, raw_exists
from . import exp_k_sweep


def run():
    sweep = load_raw("k_sweep") if raw_exists("k_sweep") else exp_k_sweep.run()
    rows, grid = [], {}
    for name, r in sweep.items():
        df = r["df"].set_index("K")
        best_nmi, bk = df["NMI"].max(), r["best_k"]
        ks = [k for k in K_GRID if k in df.index]
        grid[r["title"]] = df.loc[ks, ["NMI", "ARI", "F1", "n_clusters", "noise_pct"]]
        good = df.index[df["NMI"] >= 0.9 * best_nmi]                 # K values within 90% of the best NMI
        fk = FIXED_K if FIXED_K in df.index else df.index[np.abs(df.index - FIXED_K).argmin()]
        win = df.loc[df.index >= 5, "NMI"]
        rows.append(dict(dataset=r["title"], best_K=bk, best_NMI=round(best_nmi, 3),
                         NMI_at_fixed_K=round(df.loc[fk, "NMI"], 3), fixed_K=int(fk),
                         oracle_gap=round(best_nmi - df.loc[fk, "NMI"], 3),
                         good_K_share_pct=round(100 * len(good) / len(df), 1),
                         NMI_std_K_ge_5=round(float(win.std()), 3) if len(win) > 1 else np.nan))
    summ = pd.DataFrame(rows)
    summ.to_csv(TABLES / "exp_k_stability_summary.csv", index=False)
    pd.DataFrame({t: g["NMI"] for t, g in grid.items()}).round(3).to_csv(TABLES / "exp_k_stability_nmi_grid.csv")
    print(summ.to_string(index=False))
    save_raw("k_stability", dict(summary=summ, grid=grid))


if __name__ == "__main__":
    run()
