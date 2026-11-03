"""Numbers transcribed from the DenMune paper (Abbas et al., Pattern Recognition 2021).
F1/NMI/ARI = DenMune's score (Tables 6-11). best_other_F1 = best competitor's F1 (Tables 6-7).
Used ONLY to compare our reproduction against the paper - never as input to any algorithm."""

PAPER = {
    "spiral":    dict(F1=1.00, NMI=1.00, ARI=1.00, best_other_F1=1.00, best_other="HDBSCAN/Spectral"),
    "compound":  dict(F1=0.97, NMI=0.94, ARI=0.97, best_other_F1=0.89, best_other="RCC"),
    "jain":      dict(F1=1.00, NMI=1.00, ARI=1.00, best_other_F1=1.00, best_other="NPIR/Spectral"),
    "pathbased": dict(F1=0.97, NMI=0.89, ARI=0.92, best_other_F1=0.87, best_other="RCC"),
    "mouse":     dict(F1=0.98, NMI=0.94, ARI=0.97, best_other_F1=0.96, best_other="RCC"),
    "aggregation": dict(F1=1.00, NMI=0.99, ARI=0.99, best_other_F1=0.81, best_other="RCC"),
    "flame":     dict(F1=1.00, NMI=1.00, ARI=1.00, best_other_F1=1.00, best_other="NPIR"),
    "iris":      dict(F1=0.90, NMI=0.81, ARI=0.76, best_other_F1=0.98, best_other="Spectral"),
    "pendigits": dict(F1=0.89, NMI=0.88, ARI=0.83, best_other_F1=0.67, best_other="KM++"),
    "breast_cancer": dict(F1=0.97, NMI=0.80, ARI=0.88, best_other_F1=0.97, best_other="NPIR/FastDP"),
    "ecoli":     dict(F1=0.77, NMI=0.71, ARI=0.75, best_other_F1=0.71, best_other="CBKM"),
    "glass":     dict(F1=0.57, NMI=0.39, ARI=0.26, best_other_F1=0.59, best_other="Spectral"),
    "seeds":     dict(F1=0.89, NMI=0.69, ARI=0.69, best_other_F1=0.91, best_other="NPIR"),
    "yeast":     dict(F1=0.40, NMI=0.27, ARI=0.13, best_other_F1=0.44, best_other="CBKM"),
}

# Paper Table 4/5: best NMI of DenMune on raw features vs 2-D reductions
PAPER_DR = {
    "pendigits":     dict(raw=0.81, pca=0.48, fa=0.48, nmf=0.36, tsne=0.88),
    "iris":          dict(raw=0.73, tsne=0.81),
    "breast_cancer": dict(raw=0.75, tsne=0.80),
    "ecoli":         dict(raw=0.01, tsne=0.71),
}

# Paper Table 1 / Table 2 (Chameleon DS7, K=39)
PAPER_DS7 = dict(strong=5858, weak=3471, noise1=0, noise2=671)
