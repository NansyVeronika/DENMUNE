import numpy as np
from sklearn.metrics import (normalized_mutual_info_score, adjusted_rand_score,
                             homogeneity_score, completeness_score)


def f1_score_clustering(y, labels):
    """Clustering F-measure: each class is matched to its best cluster, weighted by class size.
    Noise (-1) points are never part of a cluster, so they lower recall."""
    y = np.asarray(y); labels = np.asarray(labels); n = len(y)
    classes = np.unique(y)
    clusters = np.unique(labels[labels >= 0])
    if len(clusters) == 0:
        return 0.0
    ci = np.searchsorted(classes, y)
    m = labels >= 0
    li = np.searchsorted(clusters, labels[m])
    cont = np.zeros((len(classes), len(clusters)))
    np.add.at(cont, (ci[m], li), 1)
    class_sz = np.bincount(ci, minlength=len(classes)).astype(float)
    clu_sz = cont.sum(0)
    prec = cont / np.maximum(clu_sz, 1)[None, :]
    rec = cont / class_sz[:, None]
    f = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float((class_sz / n * f.max(axis=1)).sum())


def evaluate(y, labels):
    """NMI/ARI/homogeneity/completeness treat noise (-1) as one extra cluster."""
    return {
        "F1": f1_score_clustering(y, labels),
        "NMI": normalized_mutual_info_score(y, labels),
        "ARI": adjusted_rand_score(y, labels),
        "Homogeneity": homogeneity_score(y, labels),
        "Completeness": completeness_score(y, labels),
    }