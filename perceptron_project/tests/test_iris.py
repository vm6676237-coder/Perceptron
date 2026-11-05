import numpy as np

from perceptron.iris import load_iris_data
from perceptron.model import Perceptron


def test_iris_data_shape() -> None:
    X, y = load_iris_data()

    assert X.shape == (100, 2)
    assert y.shape == (100,)


def test_iris_classes() -> None:
    X, y = load_iris_data()

    assert np.count_nonzero(y == 0) == 50
    assert np.count_nonzero(y == 1) == 50

    assert np.all(np.isin(y, [0, 1]))
    assert np.all(X > 0)


def test_iris_perceptron_training() -> None:
    X, y = load_iris_data()

    model = Perceptron(
        n_features=2,
        learning_rate=0.1,
        epochs=50,
    )

    model.train(X, y)

    assert model.score(X, y) >= 0.95
