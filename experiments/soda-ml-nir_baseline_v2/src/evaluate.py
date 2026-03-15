"""Оценка baseline-моделей, сохранение результатов и простых графиков."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Возвращает MAE, RMSE и R2."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}


def save_metrics(results: list[dict], output_csv: str | Path) -> pd.DataFrame:
    """Сохраняет метрики моделей в CSV."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(output_path, index=False)
    return metrics_df


def build_summary(metrics_df: pd.DataFrame) -> str:
    """Строит краткий текстовый summary по результатам."""
    if metrics_df.empty:
        return "Нет результатов для summary."

    best_by_mae = metrics_df.sort_values("mae", ascending=True).iloc[0]
    lines = [
        "# Baseline report",
        "",
        "Сравнение baseline-моделей (чем меньше MAE/RMSE и выше R2, тем лучше):",
        "",
    ]

    for _, row in metrics_df.iterrows():
        lines.append(
            f"- {row['model']} ({row['experiment_name']}): "
            f"MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, R2={row['r2']:.4f}"
        )

    lines.extend(
        [
            "",
            (
                "Лучшая конфигурация по MAE: "
                f"{best_by_mae['model']} ({best_by_mae['experiment_name']}) "
                f"(MAE={best_by_mae['mae']:.4f})"
            ),
        ]
    )
    return "\n".join(lines)


def save_summary(summary_text: str, output_md: str | Path) -> None:
    """Сохраняет markdown summary в файл."""
    output_path = Path(output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary_text, encoding="utf-8")


def save_experiments_summary(results: list[dict], output_csv: str | Path) -> pd.DataFrame:
    """Сохраняет сводку экспериментов: модель, параметры, метрики и дата запуска."""
    run_date = datetime.now().isoformat(timespec="seconds")
    rows = []

    for result in results:
        row = result.copy()
        row["run_date"] = run_date
        rows.append(row)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_path, index=False)
    return summary_df


def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    metric_name: str,
    output_path: str | Path,
) -> None:
    """Строит простой bar-график для выбранной метрики."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plot_df = metrics_df[["experiment_name", metric_name]].copy()
    plt.figure(figsize=(9, 4))
    plt.bar(plot_df["experiment_name"], plot_df[metric_name])
    plt.title(f"{metric_name.upper()} comparison")
    plt.xlabel("Experiment")
    plt.ylabel(metric_name.upper())
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_feature_importance(
    feature_names: Sequence[str],
    importances: np.ndarray,
    model_name: str,
    output_path: str | Path,
    top_n: int = 15,
) -> None:
    """Строит bar-график важности признаков для модели."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    importance_df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": importances,
        }
    )
    top_df = importance_df.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 5))
    plt.bar(top_df["feature"], top_df["importance"])
    plt.title(f"Feature importance: {model_name}")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
