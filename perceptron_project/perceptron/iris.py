"""Загрузка и подготовка реального набора данных Iris."""

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris


def load_iris_data() -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
]:

    data, target = load_iris(return_X_y=True)

    X = np.asarray(data, dtype=np.float64)
    y = np.asarray(target, dtype=np.int64)

    mask = y < 2

    X = X[mask][:, [2, 3]]
    y = y[mask]

    return X, y
