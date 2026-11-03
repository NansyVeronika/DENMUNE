import pickle

import numpy as np

from .config import RAW_RES


def save_raw(name, obj):
    with open(RAW_RES / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def raw_exists(name):
    return (RAW_RES / f"{name}.pkl").exists()


def load_raw(name):
    p = RAW_RES / f"{name}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"results/raw/{name}.pkl missing - run the experiments first.")
    with open(p, "rb") as f:
        return pickle.load(f)


# ---- small helpers shared by the new experiments ----
def n_clusters(lab):
    lab = np.asarray(lab)
    return int(lab.max() + 1) if (lab >= 0).any() else 0


def noise_pct(lab):
    return 100.0 * float(np.mean(np.asarray(lab) < 0))


def best_k(name, default=20):
    """K with the highest NMI in the K sweep (the paper also reports each method at its best setting)."""
    if raw_exists("k_sweep"):
        r = load_raw("k_sweep")
        if name in r:
            return r[name]["best_k"]
    return default
