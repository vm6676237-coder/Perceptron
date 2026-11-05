import numpy as np
from numpy.typing import NDArray


def generate_data(
    samples: int,
    random_state: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:

    if samples < 2:
        raise ValueError("Количество точек должно быть не меньше двух.")

    rng = np.random.default_rng(random_state)

    class_0_size = samples // 2
    class_1_size = samples - class_0_size

    class_0 = rng.uniform(
        low=-3.0,
        high=-1.0,
        size=(class_0_size, 2),
    )

    class_1 = rng.uniform(
        low=1.0,
        high=3.0,
        size=(class_1_size, 2),
    )

    X = np.vstack((class_0, class_1))

    y = np.concatenate(
        (
            np.zeros(class_0_size, dtype=np.int64),
            np.ones(class_1_size, dtype=np.int64),
        )
    )

    indices = rng.permutation(samples)

    X = X[indices]
    y = y[indices]

    return X, y
