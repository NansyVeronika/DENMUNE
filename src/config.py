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
N_RUNS = 20
DS7_K = 39       # K used for Chameleon DS7 in the paper
FIXED_K = 20     # a label-free "default" K, used to measure the cost of NOT tuning K
K_GRID = [3, 5, 6, 8, 10, 12, 15, 20, 30, 39]   
DATASETS = {
    "spiral":    dict(keys=["spiral"],    kind="synthetic", title="Spiral",    fig=9),
    "compound":  dict(keys=["compound"],  kind="synthetic", title="Compound",  fig=10),
    "jain":      dict(keys=["jain"],      kind="synthetic", title="Jain",      fig=11),
    "pathbased": dict(keys=["pathbased"], kind="synthetic", title="Pathbased", fig=12),
    "mouse":     dict(keys=["mouse"],     kind="synthetic", title="Mouse",     fig=8),
    "aggregation": dict(keys=["aggregation"], kind="synthetic", title="Aggregation", fig=None),
    "flame":     dict(keys=["flame"],     kind="synthetic", title="Flame",     fig=None),
    "ds7":       dict(keys=["ds7", "chameleon"], kind="synthetic", title="Chameleon DS7", fig=None),
    "iris":      dict(keys=["iris"],      kind="real", title="Iris",      fig=None),
    "pen": dict(keys=["pen"], kind="real", title="Pen", fig=None),
    "ecoli":     dict(keys=["ecoli"],     kind="real", title="Ecoli",     fig=None),
}

SHOWCASE = ["spiral", "compound", "jain", "pathbased", "mouse", "aggregation", "flame"]
