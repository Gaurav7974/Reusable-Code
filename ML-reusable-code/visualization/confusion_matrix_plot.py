"""
Simple confusion matrix plotter using matplotlib.
This is intentionally lightweight — no sklearn dependency.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix"
):
    # Build confusion matrix manually; keeps it reusable
    classes = np.unique(y_true) if labels is None else labels
    num_classes = len(classes)

    mat = np.zeros((num_classes, num_classes), dtype=float)

    # fill matrix
    for t, p in zip(y_true, y_pred):
        ti = classes.index(t) if labels else np.where(classes == t)[0][0]
        pi = classes.index(p) if labels else np.where(classes == p)[0][0]
        mat[ti, pi] += 1

    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid div-by-zero
        mat = mat / row_sums

    # basic plot; simple on purpose
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()

    # annotate cells
    for i in range(num_classes):
        for j in range(num_classes):
            val = mat[i, j]
            plt.text(j, i, f"{val:.2f}" if normalize else int(val),
                     ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()
