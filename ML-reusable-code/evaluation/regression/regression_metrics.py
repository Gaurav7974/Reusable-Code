"""
Simple regression metrics for quick evaluation.
"""

import numpy as np
from typing import Tuple


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # mean squared error
    diff = y_true - y_pred
    return float((diff ** 2).mean())


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # mean absolute error
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # root MSE; used a lot in real ML
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Basic R² implementation.
    Not the sklearn-perfect version, but works well for most tasks.
    """

    mean_y = y_true.mean()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - mean_y) ** 2).sum()

    # guard: if data is constant, r2 is undefined
    if ss_tot == 0:
        return 0.0

    return float(1 - ss_res / ss_tot)
