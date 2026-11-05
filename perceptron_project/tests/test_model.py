import numpy as np

from perceptron.model import Perceptron


def test_initial_weights() -> None:
    model = Perceptron(n_features=2)

    assert len(model.weights) == 2
    assert np.all(model.weights == 0)
    assert model.bias == 0.0


def test_activate() -> None:
    """Проверяет работу функции активации."""
    model = Perceptron(n_features=2)

    assert model.activate(0) == 1
    assert model.activate(1) == 1
    assert model.activate(-1) == 0


def test_predict() -> None:
    """Проверяет правильность предсказания."""
    model = Perceptron(n_features=2)

    model.weights = np.array([1.0, 1.0])
    model.bias = -2.0

    assert model.predict(np.array([3.0, 3.0])) == 1
    assert model.predict(np.array([0.0, 0.0])) == 0


def test_train() -> None:
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )

    y = np.array([0, 0, 1, 1], dtype=np.int64)

    model = Perceptron(n_features=2, epochs=20)

    errors = model.train(X, y)

    assert len(errors) == 20
    assert errors[-1] == 0
    assert model.score(X, y) == 1.0


def test_score() -> None:
    model = Perceptron(n_features=2)

    model.weights = np.array([1.0, 1.0])
    model.bias = -2.0

    X = np.array(
        [
            [3.0, 3.0],
            [0.0, 0.0],
            [1.0, 0.5],
            [4.0, 4.0],
        ],
        dtype=np.float64,
    )

    y = np.array([1, 0, 0, 1], dtype=np.int64)

    assert model.score(X, y) == 1.0
