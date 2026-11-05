# Perceptron Project

Проект реализует однослойный перцептрон для бинарной классификации.

## Возможности

- Генерация синтетических данных.
- Обучение перцептрона по правилу коррекции весов.
- Предсказание классов 0 и 1.
- Расчёт точности классификации.
- Визуализация границы решения.
- График количества ошибок по эпохам.
- Проверка модели на реальном наборе данных Iris.
- Интерфейс командной строки.

## Установка

Установите зависимости:

```bash
uv sync
```

Запуск на синтетических данных:

```bash
uv run perceptron --samples 200 --lr 0.1 --epochs 50
```
Если не сработает поробуйте другой вариант:

```bash
uv run python -m perceptron.cli --samples 200 --lr 0.1 --epochs 50
```

Запуск на Iris:

```bash
uv run perceptron --iris --epochs 50
```

Если не работает попробуйте другой вариант:

```bash
uv run python -m perceptron.cli --iris --epochs 50
```

Запуск без открытия окон с графиками:
```bash
uv run perceptron --iris --no-show
```

Тестирование
```bash
uv run pytest
```

Проверка качества кода
```bash
uv run ruff check .
uv run ruff format .
uv run mypy --strict perceptron tests
```
Используемые библиотеки
NumPy
Matplotlib
Scikit-learn
Pytest