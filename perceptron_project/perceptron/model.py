import numpy as np
from numpy.typing import NDArray


class Perceptron:
    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.1,
        epochs: int = 50,
    ) -> None:
        if n_features <= 0:
            raise ValueError("Количество признаков должно быть положительным.")

        if learning_rate <= 0:
            raise ValueError("Скорость обучения должна быть положительной.")

        if epochs <= 0:
            raise ValueError("Количество эпох должно быть положительным.")

        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = np.zeros(n_features)

        self.bias: float = 0.0

    def activate(self, z: float) -> int:
        return 1 if z >= 0 else 0

    def predict(self, x: NDArray[np.float64]) -> int:
        if x.shape != (self.n_features,):
            raise ValueError("Количество признаков не соответствует модели.")

        z = float(np.dot(self.weights, x) + self.bias)

        return self.activate(z)

    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> list[int]:

        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("Неверная форма массива признаков.")

        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("Количество меток не совпадает с количеством объектов.")

        if X.shape[0] == 0:
            raise ValueError("Обучающая выборка не должна быть пустой.")

        if not np.all((y == 0) | (y == 1)):
            raise ValueError("Метки классов должны быть равны 0 или 1.")

        errors_per_epoch: list[int] = []

        for _epoch in range(self.epochs):
            errors = 0

            for i in range(X.shape[0]):
                prediction = self.predict(X[i])
                target = int(y[i])

                error = target - prediction

                if error != 0:
                    self.weights += self.learning_rate * error * X[i]

                    self.bias += self.learning_rate * error
                    errors += 1

            errors_per_epoch.append(errors)

        return errors_per_epoch

    def score(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> float:

        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("Неверная форма массива признаков.")

        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("Количество меток не совпадает с количеством объектов.")

        if X.shape[0] == 0:
            raise ValueError("Проверочная выборка не должна быть пустой.")

        correct = 0

        for i in range(X.shape[0]):
            prediction = self.predict(X[i])

            if prediction == int(y[i]):
                correct += 1

        return float(correct / X.shape[0])
