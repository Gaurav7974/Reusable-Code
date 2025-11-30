"""
Small collection of common classification metrics.
Nothing fancy — just the stuff you use all the time and get tired of rewriting.
"""

from typing import Tuple
import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # basic accuracy; assumes labels already aligned
    return float((y_true == y_pred).mean())


def precision_recall_f1(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    positive_label: int = 1
) -> Tuple[float, float, float]:
    """
    Computes precision, recall, and F1.
    Note: this is a simple version for binary classification.
    """

    tp = ((y_pred == positive_label) & (y_true == positive_label)).sum()
    fp = ((y_pred == positive_label) & (y_true != positive_label)).sum()
    fn = ((y_pred != positive_label) & (y_true == positive_label)).sum()

    # avoid divide-by-zero headaches
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0

    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * (prec * rec) / (prec + rec)

    return float(prec), float(rec), float(f1)
