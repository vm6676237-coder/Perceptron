from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from perceptron.model import Perceptron


def plot_decision_boundary(
    model: Perceptron,
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        label="Class 0",
        alpha=0.7,
    )

    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        label="Class 1",
        alpha=0.7,
    )

    x_min = float(np.min(X[:, 0])) - 1.0
    x_max = float(np.max(X[:, 0])) + 1.0
    y_min = float(np.min(X[:, 1])) - 1.0
    y_max = float(np.max(X[:, 1])) + 1.0

    w1 = float(model.weights[0])
    w2 = float(model.weights[1])
    bias = float(model.bias)

    if w2 != 0:
        x_values = np.linspace(x_min, x_max, 100)
        y_values = -(w1 * x_values + bias) / w2

        ax.plot(
            x_values,
            y_values,
            color="black",
            label="Decision boundary",
        )

    elif w1 != 0:
        x_boundary = -bias / w1

        ax.axvline(
            x=x_boundary,
            color="black",
            label="Decision boundary",
        )

    else:
        ax.text(
            0.5,
            0.5,
            "Decision boundary is undefined",
            transform=ax.transAxes,
            ha="center",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title("Perceptron decision boundary")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)

    if show:
        plt.show()

    plt.close(fig)


def plot_errors(
    errors_per_epoch: list[int],
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:

    if not errors_per_epoch:
        raise ValueError("История ошибок не должна быть пустой.")

    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = range(1, len(errors_per_epoch) + 1)

    ax.plot(
        epochs,
        errors_per_epoch,
        marker="o",
        label="Errors",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Number of errors")
    ax.set_title("Perceptron training errors")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)

    if show:
        plt.show()

    plt.close(fig)
