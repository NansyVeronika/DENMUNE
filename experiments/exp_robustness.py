"""EXTENSION - Robustness of the t-SNE + DenMune pipeline.
Exp A: t-SNE random-seed sensitivity on real high-dimensional datasets.
Exp B: increasing amounts of uniform background noise on 2-D synthetic datasets."""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

from src.config import TABLES
from src.data_loader import load_dataset, find_file
from src.denmune import DenMune
from src.metrics import evaluate
from src.utils import save_raw, load_raw, raw_exists

SEEDS = range(10)
HIGH_D = ["breast_cancer", "ecoli"]
NOISE_DS = ["aggregation", "compound", "jain"]
NOISE_FRACS = [0, 0.05, 0.10, 0.20, 0.30, 0.50]
REPS = 5


def _k_for(name, default):
    """Reuse the best K found in your K-sweep so the extension uses the same settings."""
    if raw_exists("k_sweep"):
        r = load_raw("k_sweep")
        if name in r:
            return r[name]["best_k"]
    return default


def _ncl(lab):
    return int(lab.max() + 1) if (lab >= 0).any() else 0


# ----------------------------- Experiment A -----------------------------
def run_a():
    out = {}
    for name in HIGH_D:
        if find_file(name) is None:
            print(f"(skipping {name}: file not found)")
            continue
        d = load_dataset(name)
        X, y = d["X_raw"], d["y"]
        if y is None or X.shape[1] == 2:
            continue
        k = _k_for(name, 20)
        rows, labs, embs = [], [], []
        for s in SEEDS:
            # init="random" on purpose: with init="pca" (used by data_loader) t-SNE is
            # almost deterministic, so changing the seed would reveal nothing.
            Z = TSNE(n_components=2, init="random", random_state=s,
                     perplexity=min(30, (len(X) - 1) / 3)).fit_transform(X)
            lab = DenMune(k).fit(Z).labels_
            m = evaluate(y, lab)
            rows.append(dict(seed=s, K=k, n_clusters=_ncl(lab),
                             noise_pct=100 * float(np.mean(lab < 0)), **m))
            labs.append(lab)
            embs.append(Z)
            print(f"[Exp A] {d['title']} seed={s}: clusters={_ncl(lab)} NMI={m['NMI']:.3f} ARI={m['ARI']:.3f}")
        df = pd.DataFrame(rows)
        pair = [adjusted_rand_score(a, b) for a, b in combinations(labs, 2)]
        df.to_csv(TABLES / f"ext_A_tsne_seeds_{name}.csv", index=False)
        print(f"   -> {d['title']}: NMI {df.NMI.mean():.3f} +- {df.NMI.std():.3f} | "
              f"clusters {df.n_clusters.min()}-{df.n_clusters.max()} | "
              f"mean pairwise ARI between seeds {np.mean(pair):.3f}")
        out[name] = dict(title=d["title"], df=df, pair_ari=pair, k=k, embs=embs, labs=labs)
    save_raw("ext_tsne_seeds", out)
    return out


# ----------------------------- Experiment B -----------------------------
def run_b(seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name in NOISE_DS:
        if find_file(name) is None:
            print(f"(skipping {name}: file not found)")
            continue
        d = load_dataset(name)
        X, y, n = d["X"], d["y"], len(d["X"])
        if y is None:
            continue
        k = _k_for(name, 15)
        lo, hi = X.min(0), X.max(0)
        pad = 0.05 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        rows, examples = [], {}
        for f in NOISE_FRACS:
            m = int(round(f * n))
            for r in range(REPS if m else 1):
                noise = rng.uniform(lo, hi, size=(m, 2))
                Xn = np.vstack([X, noise])
                lab = DenMune(k).fit(Xn).labels_
                orig, inj = lab[:n], lab[n:]
                met = evaluate(y, orig)           # judged on the ORIGINAL points only
                rows.append(dict(noise_frac=f, rep=r, K=k, n_clusters=_ncl(lab),
                                 injected_detected_pct=100 * float(np.mean(inj < 0)) if m else np.nan,
                                 original_lost_pct=100 * float(np.mean(orig < 0)),
                                 F1=met["F1"], NMI=met["NMI"], ARI=met["ARI"]))
                if r == 0 and f in (0, 0.2, 0.5):
                    examples[f] = (Xn, lab, n)
            print(f"[Exp B] {d['title']} noise={int(f*100)}%: "
                  f"ARI={np.mean([q['ARI'] for q in rows if q['noise_frac']==f]):.3f}")
        df = pd.DataFrame(rows)
        agg = df.groupby("noise_frac").mean(numeric_only=True).reset_index()
        agg.to_csv(TABLES / f"ext_B_noise_{name}.csv", index=False)
        out[name] = dict(title=d["title"], raw=df, agg=agg, std=df.groupby("noise_frac").std(numeric_only=True).reset_index(),
                         examples=examples, k=k)
    save_raw("ext_noise", out)
    return out


def run():
    run_a()
    run_b()


if __name__ == "__main__":
    run()