"""Минимальная подготовка признаков для baseline-моделей."""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd


def build_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    exclude_columns: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Выбирает числовые колонки, при необходимости исключает часть признаков
    и заполняет пропуски медианами train.
    """
    exclude_columns = set(exclude_columns or [])

    numeric_train = x_train.select_dtypes(include=["number"]).copy()
    numeric_test = x_test.select_dtypes(include=["number"]).copy()

    if exclude_columns:
        numeric_train = numeric_train.drop(columns=list(exclude_columns), errors="ignore")
        numeric_test = numeric_test.drop(columns=list(exclude_columns), errors="ignore")

    if numeric_train.empty:
        raise ValueError("После отбора числовых признаков не осталось колонок")

    medians = numeric_train.median()
    numeric_train = numeric_train.fillna(medians)
    numeric_test = numeric_test.fillna(medians)

    common_columns = numeric_train.columns.intersection(numeric_test.columns)
    numeric_train = numeric_train[common_columns]
    numeric_test = numeric_test[common_columns]

    if numeric_train.empty:
        raise ValueError("Нет общих числовых признаков между train и test")

    return numeric_train, numeric_test