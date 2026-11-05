from pathlib import Path

import pytest

from perceptron.dataset import generate_data
from perceptron.model import Perceptron
from perceptron.visualization import plot_decision_boundary, plot_errors


def test_plot_decision_boundary(tmp_path: Path) -> None:
    X, y = generate_data(samples=100)

    model = Perceptron(n_features=2, epochs=20)
    model.train(X, y)

    file_path = tmp_path / "boundary.png"

    plot_decision_boundary(
        model,
        X,
        y,
        save_path=file_path,
        show=False,
    )

    assert file_path.exists()


def test_plot_errors(tmp_path: Path) -> None:
    errors = [20, 15, 10, 5, 2, 0]

    file_path = tmp_path / "errors.png"

    plot_errors(
        errors,
        save_path=file_path,
        show=False,
    )

    assert file_path.exists()


def test_plot_errors_empty() -> None:
    """Проверяет обработку пустой истории ошибок."""
    with pytest.raises(ValueError):
        plot_errors([])
