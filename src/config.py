from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RAW_RES = RESULTS / "raw"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
for _p in (DATA_PROCESSED, RAW_RES, TABLES, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 0
N_RUNS = 20      # runs per randomized baseline (paper: 100)
DS7_K = 39       # K used for Chameleon DS7 in the paper (Table 1)

# keys: substrings searched (case/underscore-insensitive) in file names inside data/raw
DATASETS = {
    "aggregation":   dict(keys=["aggregation"], kind="synthetic", title="Aggregation", fig=8),
    "spiral":        dict(keys=["spiral"],      kind="synthetic", title="Spiral",      fig=9),
    "compound":      dict(keys=["compound"],    kind="synthetic", title="Compound",    fig=10),
    "jain":          dict(keys=["jain"],        kind="synthetic", title="Jain",        fig=11),
    "pathbased":     dict(keys=["pathbased"],   kind="synthetic", title="Pathbased",   fig=12),
    "ds7":           dict(keys=["ds7", "chameleon"], kind="synthetic", title="Chameleon DS7", fig=None),
    "breast_cancer": dict(keys=["breastcancer", "breast"], kind="real", title="Breast cancer", fig=None),
    "ecoli":         dict(keys=["ecoli"],       kind="real", title="Ecoli", fig=None),
}