"""DenMune quality vs K on every labeled dataset (Fig. 7) and best-K selection
(K in [1,50] for small datasets, [1,200] for big ones, best NMI kept)."""
import pandas as pd

from src.config import TABLES
from src.data_loader import available_datasets
from src.denmune import DenMune
from src.metrics import evaluate
from src.utils import save_raw


def run():
    out = {}
    for d in available_datasets(labeled_only=True):
        X, y, n = d["X"], d["y"], len(d["X"])
        ks = list(range(1, min(50, n - 1) + 1)) if n < 3000 else [1] + list(range(5, 201, 5))
        rows = []
        for k in ks:
            lab = DenMune(k).fit(X).labels_
            m = evaluate(y, lab)
            m.update(K=k, n_clusters=int(lab.max() + 1) if (lab >= 0).any() else 0)
            rows.append(m)
        df = pd.DataFrame(rows)
        best = int(df.loc[df["NMI"].idxmax(), "K"])
        df.to_csv(TABLES / f"k_sweep_{d['name']}.csv", index=False)
        out[d["name"]] = dict(title=d["title"], kind=d["kind"], df=df, best_k=best)
        print(f"[K sweep] {d['title']}: best K={best} (NMI={df['NMI'].max():.3f})")
    save_raw("k_sweep", out)
    return out


if __name__ == "__main__":
    run()