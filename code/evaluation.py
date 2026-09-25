"""Evaluation metrics for the NMF robustness experiments.

The assignment's required metric is Relative Reconstruction Error (RRE).
Clustering Accuracy and NMI are optional and use scikit-learn for evaluation
only; the NMF implementations themselves remain NumPy-only.
"""

from __future__ import annotations

import numpy as np


def relative_reconstruction_error(V_clean, W, H):
    """Return ||V_clean - WH||_F / ||V_clean||_F.

    NMF must be fitted to the corresponding corrupted matrix.  ``V_clean``
    is the uncorrupted reference containing the same image columns in the same
    order.
    """

    V_clean = _finite_real_matrix("V_clean", V_clean)
    W = _finite_real_matrix("W", W)
    H = _finite_real_matrix("H", H)

    if W.shape[0] != V_clean.shape[0]:
        raise ValueError("W and V_clean must have the same number of rows")
    if H.shape[1] != V_clean.shape[1]:
        raise ValueError("H and V_clean must contain the same number of images")
    if W.shape[1] != H.shape[0]:
        raise ValueError("W and H have incompatible factor dimensions")

    denominator = np.linalg.norm(V_clean, ord="fro")
    if denominator == 0:
        raise ValueError("RRE is undefined when V_clean has zero Frobenius norm")

    reconstruction = W @ H
    numerator = np.linalg.norm(V_clean - reconstruction, ord="fro")
    return float(numerator / denominator)


def clustering_metrics(H, labels, n_clusters=None, random_state=0, n_init=10):
    """Return optional clustering Accuracy and NMI from image codes ``H.T``.

    Each predicted cluster is mapped to the most common ground-truth label in
    that cluster, matching the assignment tutorial.  Scikit-learn is imported
    lazily so RRE-only experiments do not require it.
    """

    H = _finite_real_matrix("H", H)
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size != H.shape[1]:
        raise ValueError("labels must be one-dimensional with one label per image")
    if labels.size == 0:
        raise ValueError("labels must not be empty")

    unique_labels = np.unique(labels)
    if n_clusters is None:
        n_clusters = len(unique_labels)
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)):
        raise ValueError("n_clusters must be an integer")
    if not 1 <= n_clusters <= labels.size:
        raise ValueError("n_clusters must be between 1 and the number of images")

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import normalized_mutual_info_score
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "scikit-learn is required only when optional clustering metrics are enabled"
        ) from exc

    predicted_clusters = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    ).fit_predict(H.T)

    predicted_labels = np.empty(labels.shape, dtype=labels.dtype)
    for cluster in np.unique(predicted_clusters):
        mask = predicted_clusters == cluster
        values, counts = np.unique(labels[mask], return_counts=True)
        predicted_labels[mask] = values[np.argmax(counts)]

    accuracy = float(np.mean(predicted_labels == labels))
    # Use majority-mapped class labels consistently for both metrics.
    nmi = float(normalized_mutual_info_score(labels, predicted_labels))
    return {
        "accuracy": accuracy,
        "nmi": nmi,
        "cluster_labels": predicted_clusters,
        "predicted_labels": predicted_labels,
    }


def _finite_real_matrix(name, value):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite, non-empty 2D matrix")
    return value
