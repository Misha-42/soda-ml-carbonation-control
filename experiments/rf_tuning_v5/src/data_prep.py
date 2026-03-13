"""Минимальная подготовка данных для baseline-обучения."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


REQUIRED_COLUMNS = ["target"]


def load_csv(data_path: str | Path) -> pd.DataFrame:
    """Загружает CSV-файл в DataFrame."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV файл не найден: {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Делает базовую очистку: удаляет полные дубликаты и пустые строки."""
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.dropna(how="all")
    return cleaned


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Проверяет наличие обязательных колонок."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")


def time_based_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    time_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Делит данные на train/test по времени или по порядку строк."""
    if not 0 < test_size < 1:
        raise ValueError("test_size должен быть в диапазоне (0, 1)")

    work_df = df.copy()
    if time_column:
        if time_column not in work_df.columns:
            raise ValueError(f"Временная колонка не найдена: {time_column}")
        work_df[time_column] = pd.to_datetime(work_df[time_column], errors="coerce")
        work_df = work_df.dropna(subset=[time_column]).sort_values(by=time_column)

    split_index = int(len(work_df) * (1 - test_size))
    if split_index <= 0 or split_index >= len(work_df):
        raise ValueError("Некорректный split: проверьте размер данных и test_size")

    train_df = work_df.iloc[:split_index].copy()
    test_df = work_df.iloc[split_index:].copy()

    x_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    x_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return x_train, x_test, y_train, y_test
