import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from perceptron.dataset import generate_data
from perceptron.iris import load_iris_data
from perceptron.model import Perceptron
from perceptron.visualization import (
    plot_decision_boundary,
    plot_errors,
)


def split_data(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
]:

    if not 0 < test_size < 1:
        raise ValueError("test_size должен находиться между 0 и 1.")

    if len(X) != len(y) or len(y) < 2:
        raise ValueError("Недостаточно данных для разделения.")

    rng = np.random.default_rng(random_state)

    indices = rng.permutation(len(X))

    test_count = max(1, int(len(X) * test_size))

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices],
    )


def create_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Обучение однослойного перцептрона.")

    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Количество искусственных объектов.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Скорость обучения.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Количество эпох обучения.",
    )

    parser.add_argument(
        "--iris",
        action="store_true",
        help="Использовать реальный набор данных Iris.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Папка для сохранения графиков.",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Не открывать графики в отдельных окнах.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запускает программу и выполняет обучение."""

    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        if args.iris:
            X, y = load_iris_data()
            dataset_name = "Iris"
        else:
            X, y = generate_data(samples=args.samples)
            dataset_name = "Synthetic"

        X_train, X_test, y_train, y_test = split_data(X, y)

        model = Perceptron(
            n_features=X.shape[1],
            learning_rate=args.lr,
            epochs=args.epochs,
        )

        errors = model.train(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        args.output_dir.mkdir(parents=True, exist_ok=True)

        plot_decision_boundary(
            model,
            X,
            y,
            save_path=args.output_dir / "decision_boundary.png",
            show=not args.no_show,
        )

        plot_errors(
            errors,
            save_path=args.output_dir / "training_errors.png",
            show=not args.no_show,
        )

        print(f"Dataset: {dataset_name}")
        print(f"Количество объектов: {len(X)}")
        print(f"Обучающая выборка: {len(X_train)}")
        print(f"Тестовая выборка: {len(X_test)}")
        print(f"Количество эпох: {args.epochs}")
        print(f"Точность: {accuracy:.2%}")
        print(f"Ошибки последней эпохи: {errors[-1]}")
        print(f"Графики сохранены в: {args.output_dir}")

        return 0

    except ValueError as error:
        parser.error(str(error))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
