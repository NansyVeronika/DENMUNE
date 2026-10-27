"""Loads whatever format the datasets are in (.mat / .txt / .csv / .data).
Prints which arrays it picked so you can verify. High-dimensional data is
reduced to 2-D with t-SNE (as in the paper, Sec. 4.1) and cached in data/processed."""
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.manifold import TSNE

from .config import DATA_RAW, DATA_PROCESSED, DATASETS, SEED

EXTS = {".mat", ".txt", ".csv", ".data", ".dat", ".tsv"}
LABEL_KEYS = ["y", "label", "labels", "gt", "class", "classes", "target", "groundtruth",
              "ground_truth", "truth", "cluster", "clusters", "gnd", "c"]
FEAT_KEYS = ["x", "data", "fea", "features", "points", "d", "samples", "input"]


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_file(name):
    keys = DATASETS[name]["keys"]
    hits = [p for p in DATA_RAW.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTS and any(k in _norm(p.stem) for k in keys)]
    if not hits:
        return None
    hits.sort(key=lambda p: (p.suffix.lower() != ".mat", len(p.name)))
    return hits[0]


def _label_like(v):
    v = np.asarray(v, float).ravel()
    if not np.allclose(v, np.round(v)):
        return False
    return 2 <= len(np.unique(v)) <= max(2, min(60, len(v) // 5))


def _split_matrix(M):
    if M.shape[1] >= 3 and _label_like(M[:, -1]):
        return M[:, :-1], M[:, -1]
    return M, None


def _read_mat(path):
    try:
        return {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    except NotImplementedError:                      # MATLAB v7.3 files
        import h5py
        with h5py.File(path, "r") as f:
            return {k: np.array(v).T for k, v in f.items() if isinstance(v, h5py.Dataset)}


def _load_mat(path):
    arrs = {}
    for k, v in _read_mat(path).items():
        v = np.asarray(v)
        if v.dtype.kind in "iuf" and v.size > 1:
            arrs[k] = v.astype(float)
    mats = {k: v for k, v in arrs.items() if v.ndim == 2 and min(v.shape) >= 2}
    vecs = {k: v.ravel() for k, v in arrs.items() if v.ndim == 1 or (v.ndim == 2 and 1 in v.shape)}
    if not mats:
        raise ValueError(f"no feature matrix found in {path.name}: keys={list(arrs)}")
    xkey = next((k for k in mats if k.lower() in FEAT_KEYS), None) or max(mats, key=lambda k: mats[k].size)
    X = mats[xkey]
    ykey = next((k for k in vecs if k.lower() in LABEL_KEYS), None)
    if ykey is None:
        ykey = next((k for k, v in vecs.items() if len(v) in X.shape), None)
    if ykey is not None:
        y = vecs[ykey]
        if X.shape[0] != len(y) and X.shape[1] == len(y):
            X = X.T
    else:
        if X.shape[0] < X.shape[1] and X.shape[0] <= 3:
            X = X.T
        X, y = _split_matrix(X)
    print(f"  [{path.name}] features='{xkey}' labels='{ykey if ykey else 'last column / none'}'")
    return X, y


def _id_like(v):
    v = np.asarray(v, float)
    return bool(np.allclose(v, np.round(v)) and len(np.unique(v)) == len(v))


def _load_text(path):
    with open(path, errors="ignore") as f:
        first = f.readline()
    sep = "," if "," in first else (";" if ";" in first else r"\s+")
    df = pd.read_csv(path, sep=sep, header=None, engine="python")
    df = df.dropna(axis=1, how="all")                        # empty trailing columns

    # header row? first cell non-numeric while the rest of that column is numeric
    head = pd.to_numeric(df.iloc[0], errors="coerce")
    body = df.iloc[1:].apply(pd.to_numeric, errors="coerce")
    if (head.isna() & (body.notna().mean() > 0.95)).any():
        df = df.iloc[1:].reset_index(drop=True)

    num, txt = [], []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.95:                           # '?' missing values -> median
            num.append(s.fillna(s.median()).to_numpy(float))
        else:
            txt.append(c)
    M = np.column_stack(num)
    if M.shape[1] > 4 and _id_like(M[:, 0]):                  # sample-id column (breast cancer)
        M = M[:, 1:]
    print(f"  [{path.name}] numeric columns={M.shape[1]} text columns={len(txt)}")

    if txt:                                                   # class given as text (ecoli, diagnosis M/B)
        lab = pd.factorize(df[txt[-1]])[0]
        if lab.max() + 1 <= max(2, min(60, len(lab) // 5)):
            return M, lab                                     # else it was a name column: ignore it
    if M.shape[1] >= 3 and not _label_like(M[:, -1]) and _label_like(M[:, 0]):
        return M[:, 1:], M[:, 0]                              # label stored in the first column
    return _split_matrix(M)

def _tsne(name, X):
    cache = DATA_PROCESSED / f"{name}_tsne.npy"
    if cache.exists():
        Z = np.load(cache)
        if len(Z) == len(X):
            return Z
    print(f"  running t-SNE for {name} ({X.shape[1]}-D -> 2-D) ...")
    Z = TSNE(n_components=2, init="pca", random_state=SEED,
             perplexity=min(30, (len(X) - 1) / 3)).fit_transform(X)
    np.save(cache, Z)
    return Z


@lru_cache(maxsize=None)
def load_dataset(name):
    path = find_file(name)
    if path is None:
        raise FileNotFoundError(f"No file for '{name}' in {DATA_RAW}")
    X, y = _load_mat(path) if path.suffix.lower() == ".mat" else _load_text(path)
    X = np.asarray(X, float)
    if y is not None:
        y = np.unique(np.asarray(y).ravel(), return_inverse=True)[1]
    X2 = X if X.shape[1] == 2 else _tsne(name, X)
    info = DATASETS[name]
    print(f"Loaded {name}: n={len(X)}, d={X.shape[1]}, classes={None if y is None else len(np.unique(y))}")
    return dict(name=name, title=info["title"], kind=info["kind"], X_raw=X, X=X2, y=y, file=path.name)


def available_datasets(labeled_only=False):
    out = []
    for name in DATASETS:
        if find_file(name) is None:
            print(f"(skipping '{name}': file not found in data/raw)")
            continue
        try:
            d = load_dataset(name)
        except Exception as e:
            print(f"(skipping '{name}': could not load -> {e})")
            continue
        if labeled_only and d["y"] is None:
            print(f"(skipping '{name}' for metrics: no ground-truth labels)")
            continue
        out.append(d)
    return out


def ds7_or_largest():
    if find_file("ds7") is not None:
        return load_dataset("ds7")
    return max(available_datasets(), key=lambda d: len(d["X"]))