# DenMune: Implementation, Experiments and Extensions

This project focuses on understanding and implementing **DenMune**, a density-based clustering algorithm that uses Mutual Nearest Neighbours (MNN) to identify clusters of different shapes, sizes, and densities.

The main aim of the project is to reproduce the important experiments presented in the original research paper and then explore the algorithm further through additional experiments and a real-world case study.

Along with implementing the algorithm, I have worked on generating results, visualising the clustering process, comparing DenMune with other clustering methods, and investigating how the algorithm behaves under different conditions.

## Table of Contents

- [About the Project](#about-the-project)
- [Research Paper](#research-paper)
- [How DenMune Works](#how-denmune-works)
- [Project Features](#project-features)
- [Experiments Based on the Paper](#experiments-based-on-the-paper)
- [Additional Experiments](#additional-experiments)
- [Real-World Case Study](#real-world-case-study)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running the Project](#running-the-project)
- [Results and Visualisations](#results-and-visualisations)
- [Important Observations](#important-observations)
- [Limitations](#limitations)
- [References](#references)

## About the Project

Traditional clustering algorithms may struggle when clusters have irregular shapes, different densities, or noise. Some algorithms also require the number of clusters to be specified in advance.

DenMune takes a different approach. It uses the relationships between a point and its mutual nearest neighbours to identify dense points, construct cluster structures, and assign the remaining points.

In this project, I have implemented the main DenMune workflow in Python and organised the experiments into separate modules. The project also includes additional experiments that investigate the stability of the t-SNE and DenMune pipeline, the effect of background noise, and the usefulness of DenMune for customer segmentation.

The project has two main parts:

1. **Research reproduction:** Implementation of DenMune and experiments inspired by the original paper.
2. **Extensions:** Additional experiments to study robustness and apply the approach to a real-world customer dataset.

## Research Paper

**Paper:** *DenMune: Density peak based clustering using mutual nearest neighbors*

**Authors:** M. Abbas, A. El-Zoghabi and A. Shoukry

**Journal:** Pattern Recognition, 2021

The paper proposes a clustering algorithm based on Mutual Nearest Neighbours (MNN). It aims to discover clusters without requiring the number of clusters to be supplied beforehand.

The implementation and experiments in this project are based on the ideas and evaluation procedures described in the paper. Some comparison algorithms are simplified reimplementations, so their results should not automatically be treated as identical to those of the original implementations.

## How DenMune Works

DenMune uses the number of nearest neighbours, represented by **K**, as its main parameter.

The algorithm can be understood through the following steps.

### 1. Find the K-nearest neighbours

For every point in the dataset, the algorithm identifies its K nearest neighbours.

### 2. Identify Mutual Nearest Neighbours

Two points are mutual nearest neighbours when each point appears in the other's nearest-neighbour list.

These relationships help identify local density and connections between points.

### 3. Classify the points

The implementation tracks four point categories:

- **Strong points:** Dense points that can participate in constructing the cluster structure.
- **Weak points:** Points that have mutual-neighbour relationships but do not satisfy the strong-point condition.
- **Type-1 noise:** Points without mutual nearest neighbours.
- **Type-2 noise:** Weak points that cannot be assigned to a cluster during the propagation stage.

### 4. Construct the cluster structure

In Phase I, strong points connected through mutual-neighbour relationships form the initial cluster structure.

### 5. Assign weak points

In Phase II, weak points are examined and assigned to a cluster when their mutual-neighbour relationships connect them to the existing cluster structure. Points that cannot be assigned remain noise.

The algorithm produces cluster labels, with `-1` representing noise in this implementation.

## Project Features

- Python implementation of the DenMune clustering workflow.
- Mutual-neighbour analysis and strong/weak/noise point classification.
- Cluster formation and weak-point assignment.
- Experiments on synthetic and real datasets.
- Evaluation using clustering metrics such as NMI, ARI and F1, where applicable.
- Investigation of how the choice of K affects clustering.
- Analysis of cluster propagation and execution time.
- Comparison with several other clustering methods.
- Additional robustness experiments.
- A customer-segmentation case study using the Wholesale Customers dataset.
- CSV outputs and visualisations for analysing experimental results.

## Experiments Based on the Paper

The `experiments/` and `visualization/` folders contain scripts for reproducing and analysing several experiments inspired by the research paper.

### 1. DS7 point classification

The DS7 experiment studies the classification of points into strong, weak, and two noise categories.

It records the number of points in each category for different K values and generates visualisations of the resulting clusters.

### 2. Cluster propagation

This experiment investigates how the cluster structure develops during the algorithm's first phase.

It helps illustrate how strong points form the initial cluster structure and how the number of clusters changes as the process progresses.

### 3. K sensitivity

K is the main parameter of DenMune. This experiment evaluates the algorithm at different K values and records clustering performance.

The objective is to understand how sensitive the results are to the choice of K.

### 4. Execution time

The speed experiment measures the execution time of the clustering process on selected datasets and compares the behaviour of different algorithms.

### 5. Comparison with other algorithms

The comparison module includes DenMune and several other clustering approaches, including K-means variants, Spectral Clustering, HDBSCAN, FINCH, FastDP, NPIR and RCC.

The available metrics and plots help compare clustering quality and behaviour across the selected datasets.

Some comparison methods are simplified reimplementations rather than complete reproductions of the original algorithms. This should be considered when interpreting the results.

## Additional Experiments

Beyond the experiments based on the paper, I added three investigations to explore the behaviour of the DenMune pipeline further.

### Extension A: Sensitivity to the t-SNE Random Seed

**Question:** How much do the clustering results change when t-SNE uses different random seeds?

The original pipeline reduces high-dimensional data to two dimensions using t-SNE before clustering. Since t-SNE can produce different embeddings depending on its settings and random initialisation, it is useful to investigate whether those changes affect the final clustering.

This experiment uses high-dimensional datasets, including Breast Cancer and Ecoli, and repeats the t-SNE plus DenMune pipeline with multiple random seeds.

The experiment records:

- NMI and ARI against the available ground-truth labels.
- Number of clusters found.
- Percentage of points classified as noise.
- Pairwise ARI between clustering results obtained using different seeds.

The pairwise ARI measures the agreement between different runs, independently of the original ground-truth labels.

This helps distinguish two questions: whether a run agrees with the known labels and whether the pipeline consistently produces similar cluster assignments.

### Extension B: Robustness to Increasing Background Noise

**Question:** What happens when additional noise points are introduced into a dataset?

This experiment adds uniformly distributed background points to selected two-dimensional datasets, including Compound and Jain.

Different noise levels are tested, and the experiment is repeated to observe how the results vary.

The measurements include:

- ARI, NMI and F1 on the original dataset points.
- Percentage of injected points identified as noise.
- Percentage of original points classified as noise.
- Number of clusters found.

The clustering metrics are evaluated on the original points so that the experiment can investigate how adding background noise affects the clustering of the data that was already present.

This experiment provides a practical way to examine the robustness of DenMune when the data contains additional irrelevant or noisy observations.

### Extension C: Customer Segmentation Using Wholesale Customers

**Question:** Can DenMune reveal useful customer groups from real-world spending data?

This extension applies DenMune to the UCI Wholesale Customers dataset, which contains annual spending information across product categories.

The six spending features used in the analysis are:

- Fresh
- Milk
- Grocery
- Frozen
- Detergents_Paper
- Delicassen

The pipeline includes:

1. Applying `log1p` to reduce the influence of highly skewed spending values.
2. Standardising the features.
3. Reducing the data to two dimensions using t-SNE and PCA separately.
4. Selecting K using silhouette score on the non-noise points, without using the ground-truth labels.
5. Applying DenMune to each two-dimensional representation.
6. Comparing the resulting cluster assignments.

The extension also analyses median spending profiles for each cluster and investigates whether points classified as noise have unusual spending values.

The dataset's `Channel` column is used only for an additional comparison. It is not treated as ground truth for the clusters.

Since the dataset does not provide ground-truth cluster labels, this experiment focuses on exploratory analysis and interpretation rather than claiming that the clusters are objectively correct customer segments.

## Project Structure

The project is organised into separate folders for the algorithm, experiments, visualisations, datasets and results.

```text
DENMUNE/
│
├── experiments/
│   ├── exp_ds7_points.py
│   ├── exp_fig5_propagation.py
│   ├── exp_fig6_speed.py
│   ├── exp_k_sweep.py
│   ├── exp_robustness.py
│   └── exp_wholesale.py
│
├── src/
│   ├── __init__.py
│   ├── baselines.py
│   ├── config.py
│   ├── data_loader.py
│   ├── denmune.py
│   ├── metrics.py
│   └── utils.py
│
├── visualization/
│   ├── plot_comparison.py
│   ├── plot_curves.py
│   ├── plot_ds7.py
│   ├── plot_extension.py
│   ├── plot_utils.py
│   └── run_all.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── raw/
│
├── README.md
└── requirements.txt
```

The exact files and result subfolders may vary slightly depending on which experiments have been run.

### Main folders

- `src/`: Core DenMune implementation, dataset loading, evaluation metrics, configuration and baseline algorithms.
- `experiments/`: Scripts for the paper-based experiments and the new extensions.
- `visualization/`: Code for generating plots and presenting experimental results.
- `data/raw/`: Input datasets.
- `data/processed/`: Processed data and cached t-SNE embeddings.
- `results/`: Generated figures, tables and saved experimental outputs.

## Technologies Used

- **Python** — main programming language.
- **NumPy** — numerical operations.
- **Pandas** — dataset handling and CSV output.
- **SciPy** — sparse matrices, graph connectivity and numerical operations.
- **Scikit-learn** — nearest neighbours, t-SNE, PCA, clustering baselines, preprocessing and evaluation metrics.
- **Matplotlib** — plotting and visualisation.
- **h5py** — support for certain MATLAB data files.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/NansyVeronika/DENMUNE.git
cd DENMUNE
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install the required libraries

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the project uses a different Python environment, activate that environment before installing the dependencies.

## Dataset Setup

Place the datasets in the `data/raw/` directory.

The loader supports several file formats, including:

- `.mat`
- `.txt`
- `.csv`
- `.data`
- `.dat`
- `.tsv`

The project includes dataset-discovery and loading functions that attempt to identify the relevant feature and label columns.

The paper-based experiments require the corresponding datasets to be available locally. The Wholesale Customers extension can read a local CSV file or attempt to retrieve the dataset from the UCI repository.

For reproducible experiments, it is recommended to retain the dataset versions used for the reported results and document any preprocessing performed.

## Running the Project

Run commands from the repository's root directory, where `src/`, `experiments/` and `visualization/` are located.

### Run the paper-based experiments and visualisations

```bash
python -m visualization.run_all
```

This runs the configured experiments and generates their associated plots.

To generate plots without rerunning the main experiments:

```bash
python -m visualization.run_all --skip-experiments
```

This option is useful when the required intermediate results have already been generated.

### Run the robustness extension

```bash
python -m experiments.exp_robustness
```

This runs the t-SNE seed-sensitivity experiment and the background-noise experiment, subject to the required datasets being available.

### Run the Wholesale Customers extension

```bash
python -m experiments.exp_wholesale
```

This runs the customer-segmentation analysis and saves the associated tabular outputs.

### Generate the extension visualisations

```bash
python -m visualization.plot_extension
```

Run this after generating the extension results required by the plotting script.

**Note:** The exact commands assume the module filenames shown in the project structure and that the scripts use module-compatible imports. If a script has a different entry point in your local version, use its actual filename and entry point.

## Results and Visualisations

The project generates plots and tables to support analysis of the clustering behaviour.

Examples of the visualisations include:

- Cluster assignments on the DS7 dataset.
- Strong, weak and noise point distributions.
- Cluster propagation during Phase I.
- Sensitivity to the choice of K.
- Execution-time comparisons.
- Comparisons between clustering algorithms.
- t-SNE random-seed sensitivity.
- Clustering behaviour under increasing background noise.
- Wholesale customer cluster profiles.
- Comparison between t-SNE-based and PCA-based clustering.

The figures and tables are intended to support the discussion of the results, rather than serve only as illustrations.

When interpreting the results, it is important to distinguish between experiments that use known ground-truth labels and exploratory clustering on data without known cluster labels.

## Important Observations

The experiments are designed to investigate several questions:

- How does K affect the cluster structure and point classification?
- How stable are the results when the t-SNE random seed changes?
- Does increasing background noise affect the detection of noise points?
- How do DenMune's results compare with those of other clustering methods?
- Do t-SNE and PCA produce similar cluster assignments?
- Can the resulting customer groups be interpreted through their spending patterns?

The answers should be based on the generated tables and figures. The experiments do not assume in advance that DenMune will perform best in every situation.

## Limitations

There are some important limitations to consider.

1. **Two-dimensional representation:** Reducing high-dimensional data to two dimensions may change neighbourhood relationships and therefore affect clustering.
2. **Parameter selection:** DenMune depends on K. Results may vary with the selected value.
3. **Baseline implementations:** Some comparison algorithms are simplified reimplementations, so comparisons may not exactly match the original published implementations.
4. **Noise experiments:** Injected noise is uniformly sampled within an expanded bounding box. Other types of noise may produce different results.
5. **Real-world interpretation:** The Wholesale Customers dataset does not provide ground-truth cluster labels. The resulting groups are exploratory and should not automatically be treated as objectively correct customer segments.
6. **Reproducibility:** Results depend on the datasets, preprocessing, parameter settings, random seeds and software environment.

These limitations are useful considerations when interpreting the experiments and identifying possible future improvements.

## References

Abbas, M., El-Zoghabi, A., & Shoukry, A. (2021). *DenMune: Density peak based clustering using mutual nearest neighbors*. Pattern Recognition, 109, 107589.

The original research paper provides the foundation for the algorithm and the experiments reproduced in this project.

**Repository:** https://github.com/NansyVeronika/DENMUNE

---

*This project was developed as an academic implementation and experimental study of DenMune, with additional investigations into robustness and real-world customer segmentation.*