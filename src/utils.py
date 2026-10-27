import pickle
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