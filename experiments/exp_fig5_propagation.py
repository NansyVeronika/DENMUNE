"""Fig. 5: number of clusters / number of clustered points vs iteration (Phase I)."""
from src.config import DS7_K
from src.data_loader import ds7_or_largest
from src.denmune import DenMune
from src.utils import save_raw


def run():
    d = ds7_or_largest()
    m = DenMune(DS7_K, track=True).fit(d["X"])
    h = m.history_
    print(f"[Fig 5] {d['title']}: {m.counts_['strong']} seeds -> {h['n_clusters'][-1]} clusters after Phase I")
    save_raw("fig5_propagation", dict(title=d["title"], k=DS7_K, n=len(d["X"]),
                                      history=h, n_seeds=m.counts_["strong"]))


if __name__ == "__main__":
    run()