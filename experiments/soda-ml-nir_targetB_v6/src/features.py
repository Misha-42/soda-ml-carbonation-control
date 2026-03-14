"""Подготовка признаков для baseline-моделей: числовые признаки, лаги и rolling mean."""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd


def _add_lag_features(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    """Добавляет лаговые признаки для указанных колонок."""
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        for lag in lags:
            result[f"{col}_lag{lag}"] = result[col].shift(lag)
    return result


def _add_rolling_mean_features(df: pd.DataFrame, columns: list[str], windows: list[int]) -> pd.DataFrame:
    """Добавляет признаки скользящего среднего для указанных колонок."""
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        for window in windows:
            result[f"{col}_roll{window}_mean"] = result[col].rolling(window=window).mean()
    return result


def build_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    exclude_columns: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Строит матрицы признаков:
    - берёт числовые колонки,
    - исключает указанные признаки,
    - добавляет лаги,
    - добавляет rolling mean,
    - заполняет пропуски медианами train.
    """
    exclude_columns = set(exclude_columns or [])

    numeric_train = x_train.select_dtypes(include=["number"]).copy()
    numeric_test = x_test.select_dtypes(include=["number"]).copy()

    if exclude_columns:
        numeric_train = numeric_train.drop(columns=list(exclude_columns), errors="ignore")
        numeric_test = numeric_test.drop(columns=list(exclude_columns), errors="ignore")

    if numeric_train.empty:
        raise ValueError("После отбора числовых признаков не осталось колонок")

    lag_columns = [
        "temp_offgas_k1",
        "gas_flow_in1_k1",
        "gas_flow_in2_k1",
        "gas_header_pressure_1",
        "gas_header_pressure_2",
        "fg_header_pressure",
    ]
    lags = [1, 2]

    rolling_columns = [
        "temp_offgas_k1",
        "gas_flow_in1_k1",
        "gas_flow_in2_k1",
        "gas_header_pressure_1",
        "gas_header_pressure_2",
        "fg_header_pressure",
    ]
    rolling_windows = [3]

    numeric_train = _add_lag_features(numeric_train, lag_columns, lags)
    numeric_test = _add_lag_features(numeric_test, lag_columns, lags)

    numeric_train = _add_rolling_mean_features(numeric_train, rolling_columns, rolling_windows)
    numeric_test = _add_rolling_mean_features(numeric_test, rolling_columns, rolling_windows)

    common_columns = numeric_train.columns.intersection(numeric_test.columns)
    numeric_train = numeric_train[common_columns]
    numeric_test = numeric_test[common_columns]

    medians = numeric_train.median()
    numeric_train = numeric_train.fillna(medians)
    numeric_test = numeric_test.fillna(medians)

    if numeric_train.empty:
        raise ValueError("Нет общих числовых признаков между train и test")

    return numeric_train, numeric_test