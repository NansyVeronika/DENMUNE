import pandas as pd

from src.config import DS7_K, TABLES
from src.data_loader import ds7_or_largest
from src.denmune import DenMune
from src.utils import save_raw


def run():
    d = ds7_or_largest()
    X, k = d["X"], DS7_K
    print(f"[Table 1/2] dataset: {d['title']} (n={len(X)}), K={k}")
    m = DenMune(k, track=True, snapshot_fracs=(0.05, 0.2, 0.4, 0.6, 0.8, 1.0)).fit(X)
    c = m.counts_
    table1 = pd.DataFrame([{"Algorithm": "DenMune", "Parameters": f"K={k}",
                            "Strong Points": c["strong"], "Weak Points": c["weak"],
                            "Noise of type-1": c["noise1"], "Noise of type-2": c["noise2"]}])
    rows = []
    for kk in range(1, 21):
        ck = DenMune(kk).fit(X).counts_
        rows.append({"K": kk, "Strong Points": ck["strong"], "Weak Points": ck["weak"],
                     "Noise of type-1": ck["noise1"], "Noise of type-2": ck["noise2"]})
    table2 = pd.DataFrame(rows)
    table1.to_csv(TABLES / "table_01_ds7_points.csv", index=False)
    table2.to_csv(TABLES / "table_02_points_vs_K.csv", index=False)
    print(table1.to_string(index=False))
    print(table2.to_string(index=False))
    save_raw("ds7_points", dict(title=d["title"], X=X, k=k, ptype=m.point_type_,
                                labels1=m.phase1_labels_, labels=m.labels_,
                                snapshots=m.snapshots_, table1=table1, table2=table2))


if __name__ == "__main__":
    run()