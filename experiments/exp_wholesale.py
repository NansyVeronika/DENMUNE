"""EXTENSION - Real-world case study: UCI Wholesale Customers (no ground-truth clusters).
Pipeline: log1p -> standardise -> 2-D (t-SNE and PCA) -> DenMune. K is chosen WITHOUT labels
(silhouette on the non-noise points), so no label leakage."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import DATA_RAW, TABLES, SEED
from src.denmune import DenMune
from src.utils import save_raw

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
SPEND = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
K_GRID = list(range(5, 41, 5))


def _load():
    f = next(iter(DATA_RAW.glob("*holesale*.csv")), None)
    return pd.read_csv(f) if f else pd.read_csv(URL)


def _ncl(lab):
    return int(lab.max() + 1) if (lab >= 0).any() else 0


def _pick_k(Z, tag):
    rows, best = [], None
    for k in K_GRID:
        lab = DenMune(k).fit(Z).labels_
        keep = lab >= 0
        nc, noise = _ncl(lab), 100 * float(np.mean(~keep))
        sil = silhouette_score(Z[keep], lab[keep]) if nc >= 2 and noise < 30 else np.nan
        rows.append(dict(K=k, n_clusters=nc, noise_pct=noise, silhouette=sil))
        if not np.isnan(sil) and (best is None or sil > best[1]):
            best = (k, sil)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(TABLES / f"ext_wholesale_ksweep_{tag}.csv", index=False)
    print(f"[Wholesale/{tag}]\n{sweep.round(2).to_string(index=False)}")
    return (best[0] if best else 15), sweep


def run():
    df = _load()
    Xlog = np.log1p(df[SPEND].to_numpy(float))        # spending is heavily right-skewed
    X = StandardScaler().fit_transform(Xlog)

    Z_tsne = TSNE(n_components=2, init="pca", random_state=SEED, perplexity=30).fit_transform(X)
    Z_pca = PCA(n_components=2, random_state=SEED).fit_transform(X)

    k_t, sw_t = _pick_k(Z_tsne, "tsne")
    k_p, sw_p = _pick_k(Z_pca, "pca")
    lab_t = DenMune(k_t).fit(Z_tsne).labels_
    lab_p = DenMune(k_p).fit(Z_pca).labels_
    print(f"t-SNE: K={k_t} clusters={_ncl(lab_t)} noise={100*np.mean(lab_t<0):.1f}% | "
          f"PCA: K={k_p} clusters={_ncl(lab_p)} noise={100*np.mean(lab_p<0):.1f}%")
    agree = adjusted_rand_score(lab_t, lab_p)
    print(f"ARI between t-SNE and PCA pipelines: {agree:.3f}")

    # 1) business interpretation: median spend per cluster
    d = df.copy()
    d["cluster"] = lab_t
    prof = d[d.cluster >= 0].groupby("cluster")[SPEND].median()
    prof["size"] = d[d.cluster >= 0].groupby("cluster").size()
    prof.to_csv(TABLES / "ext_wholesale_cluster_profiles.csv")
    print(prof)

    # 2) are the DenMune noise points really unusual customers?
    d["max_abs_z"] = np.abs(X).max(axis=1)
    nz, cl = d[d.cluster < 0], d[d.cluster >= 0]
    p = mannwhitneyu(nz.max_abs_z, cl.max_abs_z, alternative="greater").pvalue if len(nz) and len(cl) else np.nan
    print(f"max|z|: noise mean={nz.max_abs_z.mean():.2f} vs clustered mean={cl.max_abs_z.mean():.2f} "
          f"(Mann-Whitney one-sided p={p:.4g})")
    nz.sort_values("max_abs_z", ascending=False).to_csv(TABLES / "ext_wholesale_noise_customers.csv", index=False)

    # 3) external sanity check only (Channel is NOT ground truth)
    ct = pd.crosstab(d.cluster, d.Channel)
    ct.to_csv(TABLES / "ext_wholesale_vs_channel.csv")
    print(ct)

    save_raw("ext_wholesale", dict(Z_tsne=Z_tsne, Z_pca=Z_pca, lab_t=lab_t, lab_p=lab_p, k_t=k_t, k_p=k_p,
                                   profile=prof, agree=agree, pval=p, sweep_t=sw_t, sweep_p=sw_p,
                                   noise_z=float(nz.max_abs_z.mean()) if len(nz) else np.nan,
                                   clus_z=float(cl.max_abs_z.mean())))


if __name__ == "__main__":
    run()