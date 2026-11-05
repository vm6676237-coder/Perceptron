import numpy as np
import pytest

from perceptron.dataset import generate_data


def test_generate_data_shape() -> None:
    X, y = generate_data(samples=200)

    assert X.shape == (200, 2)
    assert y.shape == (200,)


def test_generate_data_classes() -> None:
    X, y = generate_data(samples=200)

    assert np.count_nonzero(y == 0) == 100
    assert np.count_nonzero(y == 1) == 100

    assert np.all(X[y == 0] < 0)
    assert np.all(X[y == 1] > 0)


def test_generate_data_reproducibility() -> None:
    X1, y1 = generate_data(samples=100, random_state=42)
    X2, y2 = generate_data(samples=100, random_state=42)

    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_generate_data_invalid_samples() -> None:
    with pytest.raises(ValueError):
        generate_data(samples=1)
