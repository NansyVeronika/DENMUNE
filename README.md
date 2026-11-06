# DenMune: Density peak based clustering using mutual nearest neighbors

A Python re-implementation of **DenMune** (*Pattern Recognition*, 2021), a clustering
algorithm that uses mutual nearest neighbours and density to find clusters of any shape,
and to separate noise.

## What this project does

1. Implements DenMune from the paper[DENMUNE](https://www.sciencedirect.com/science/article/pii/S0031320320303927).
2. Reproduces the paper's tables and figures on the datasets we have.
3. Compares DenMune with 9 other algorithms.
4. Adds extra experiments on reliability and noise.

## Project structure

```

├── data/raw/          # datasets (CSV and .mat files)
├── src/               # DenMune algorithm, baselines, metrics, utilities
├── experiments/       # all experiments
├── visualization/     # all plots and figures
├── results/           # created automatically (tables, figures)
└── README.md
```

## Datasets

| Type | Datasets |
|---|---|
| Synthetic (2-D) | Spiral, Compound, Jain, Pathbased, Mouse, Aggregation, Flame |
| Real | Iris, Pen digits, Ecoli |

Real datasets with more than 2 features are reduced to 2-D with t-SNE, as in the paper.

## Compared algorithms

NPIR, CBKM, RS, FastDP, FINCH, HDBSCAN, RCC, Spectral clustering and K-Means++.

## Metrics

F1, NMI, ARI, Homogeneity and Completeness. Higher is better.

## Experiments

**Reproducing the paper**
- Strong, weak and noise points (Tables 1 and 2)
- Cluster propagation (Fig. 5) and speed (Fig. 6)
- DenMune vs the 9 baselines (Tables 6 to 11)
- Homogeneity and completeness (Tables 12 and 13)
- Our results vs the paper's

**Our extensions**
- Where DenMune works and where it struggles (failure analysis)
- Effect of K: tuned with labels vs a fixed K
- Effect of the input: raw data, PCA, FA, NMF and t-SNE
- t-SNE random seeds
- Noise robustness: random noise points added, up to 50%

## How to run

```bash
python -m visualization.run_all.py
```

Tables are saved in `results/tables` and figures in `results/figures`.

## Main findings

- Our results are close to the paper's, usually within 0.01 to 0.05.
- DenMune has the best overall rank among the 10 algorithms on F1, NMI and ARI,
  for both synthetic and real data.
- It keeps the real cluster shapes even with 50% added noise.
- It struggles on very small datasets and on overlapping classes.
- Its quality depends on K. The paper tunes K using the true labels, and without
  tuning some datasets (for example Spiral) get much worse.


## Reference

DenMune: Density peak based clustering using mutual nearest neighbors.
*Pattern Recognition*, 2021.
