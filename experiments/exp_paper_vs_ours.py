import numpy as np
import pandas as pd

from src.config import TABLES, DATASETS
from src.paper_reference import PAPER, PAPER_DS7
from src.utils import save_raw, load_raw, raw_exists
from . import exp_comparison


def run():
    if not raw_exists("comparison"):
        exp_comparison.run()
    res = load_raw("comparison")["results"]
    rows = []
    for name, ref in PAPER.items():
        dataset_key = "pen" if name == "pendigits" else name
        if dataset_key not in DATASETS:
            print(f"Skipping {name}: not found in DATASETS")
            continue
        t = DATASETS[dataset_key]["title"]
        sub = res[res.dataset == t]
        if sub.empty:
            continue
        dm = sub[sub.algo == "DenMune"].iloc[0]
        others = sub[sub.algo != "DenMune"]
        rows.append(dict(dataset=t, F1_ours=round(dm.F1, 2), F1_paper=ref["F1"], dF1=round(dm.F1 - ref["F1"], 2),
                         NMI_ours=round(dm.NMI, 2), NMI_paper=ref["NMI"], ARI_ours=round(dm.ARI, 2),
                         ARI_paper=ref["ARI"], best_other_F1_ours=round(others.F1.max(), 2),
                         best_other_F1_paper=ref["best_other_F1"],
                         DenMune_wins_ours=bool(dm.F1 >= others.F1.max() - 1e-9),
                         DenMune_wins_paper=bool(ref["F1"] >= ref["best_other_F1"] - 1e-9)))
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "paper_vs_ours.csv", index=False)
    print(df.to_string(index=False))
    ds7 = None
    if raw_exists("ds7_points"):
        c = load_raw("ds7_points")["table1"].iloc[0]
        ds7 = pd.DataFrame([dict(source="paper", strong=PAPER_DS7["strong"], weak=PAPER_DS7["weak"],
                                 noise1=PAPER_DS7["noise1"], noise2=PAPER_DS7["noise2"]),
                            dict(source="ours", strong=c["Strong Points"], weak=c["Weak Points"],
                                 noise1=c["Noise of type-1"], noise2=c["Noise of type-2"])])
        ds7.to_csv(TABLES / "paper_vs_ours_ds7_points.csv", index=False)
        print(ds7.to_string(index=False))
    save_raw("paper_vs_ours", dict(table=df, ds7=ds7))


if __name__ == "__main__":
    run()
