"""Запуск baseline-обучения RandomForest и XGBoost на одном pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from data_prep import clean_data, load_csv, time_based_split, validate_required_columns
from evaluate import (
    build_summary,
    calculate_metrics,
    plot_feature_importance,
    plot_metric_comparison,
    save_experiments_summary,
    save_metrics,
    save_summary,
)
from features import build_features


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Train baseline RF and XGB models")
    parser.add_argument("--data-path", required=True, help="Путь к CSV в папке data/")
    parser.add_argument("--target", default="target", help="Название целевой колонки")
    parser.add_argument(
        "--time-column",
        default=None,
        help="Название временной колонки (если есть)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Доля test")
    parser.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        help="Название признака, который нужно исключить. Можно указать несколько раз.",
    )
    parser.add_argument(
        "--run-tag",
        default="baseline",
        help="Тег запуска для имён выходных файлов, например baseline, with_offgas, no_offgas.",
    )
    return parser.parse_args()


def train_and_evaluate_experiment(
    model_name: str,
    model_params: dict[str, Any],
    x_train_features,
    y_train,
    x_test_features,
    y_test,
):
    """Обучает одну конфигурацию модели и возвращает метрики."""
    if model_name == "RandomForestRegressor":
        model = RandomForestRegressor(**model_params)
    elif model_name == "XGBRegressor":
        model = XGBRegressor(**model_params)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    model.fit(x_train_features, y_train)
    predictions = model.predict(x_test_features)
    metrics = calculate_metrics(y_test, predictions)

    return model, metrics


def train_baseline_models() -> None:
    """Загружает данные, запускает несколько baseline-экспериментов и сохраняет отчёты."""
    args = parse_args()

    df = load_csv(args.data_path)
    df = clean_data(df)
    required_columns = [args.target] + ([args.time_column] if args.time_column else [])
    validate_required_columns(df, required_columns)

    x_train, x_test, y_train, y_test = time_based_split(
        df=df,
        target_column=args.target,
        test_size=args.test_size,
        time_column=args.time_column,
    )

    x_train_features, x_test_features = build_features(
        x_train,
        x_test,
        exclude_columns=args.exclude_feature,
    )

    models_dir = Path("models")
    reports_dir = Path("reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "model": "RandomForestRegressor",
            "name": "rf_small",
            "params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        },
        {
            "model": "RandomForestRegressor",
            "name": "rf_tuned_150",
            "params": {"n_estimators": 150, "random_state": 42, "n_jobs": -1},
        },
        {
            "model": "RandomForestRegressor",
            "name": "rf_tuned_300",
            "params": {"n_estimators": 300, "random_state": 42, "n_jobs": -1},
        },
        {
            "model": "RandomForestRegressor",
            "name": "rf_tuned_leaf3",
            "params": {
                "n_estimators": 200,
                "min_samples_leaf": 3,
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        {
            "model": "XGBRegressor",
            "name": "xgb_small",
            "params": {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": 42,
                "n_jobs": -1,
            },
        },
    ]

    experiment_results = []
    rf_reference_model = None
    xgb_reference_model = None

    for experiment in experiments:
        model, metrics = train_and_evaluate_experiment(
            model_name=experiment["model"],
            model_params=experiment["params"],
            x_train_features=x_train_features,
            y_train=y_train,
            x_test_features=x_test_features,
            y_test=y_test,
        )

        if experiment["model"] == "RandomForestRegressor" and rf_reference_model is None:
            rf_reference_model = model
        if experiment["model"] == "XGBRegressor" and xgb_reference_model is None:
            xgb_reference_model = model

        model_filename = f"{args.run_tag}_{experiment['name']}.joblib"
        joblib.dump(model, models_dir / model_filename)

        experiment_results.append(
            {
                "experiment_name": experiment["name"],
                "model": experiment["model"],
                "params": json.dumps(experiment["params"], ensure_ascii=False),
                **metrics,
            }
        )

    metrics_df = save_metrics(
        experiment_results,
        reports_dir / f"{args.run_tag}_metrics.csv",
    )

    save_experiments_summary(
        experiment_results,
        reports_dir / f"{args.run_tag}_experiments_summary.csv",
    )

    plot_metric_comparison(
        metrics_df,
        metric_name="mae",
        output_path=reports_dir / f"{args.run_tag}_mae_comparison.png",
    )
    plot_metric_comparison(
        metrics_df,
        metric_name="rmse",
        output_path=reports_dir / f"{args.run_tag}_rmse_comparison.png",
    )

    feature_names = list(x_train_features.columns)
    if rf_reference_model is not None:
        plot_feature_importance(
            feature_names=feature_names,
            importances=rf_reference_model.feature_importances_,
            model_name="RandomForestRegressor",
            output_path=reports_dir / f"{args.run_tag}_rf_feature_importance.png",
        )
    if xgb_reference_model is not None:
        plot_feature_importance(
            feature_names=feature_names,
            importances=xgb_reference_model.feature_importances_,
            model_name="XGBRegressor",
            output_path=reports_dir / f"{args.run_tag}_xgb_feature_importance.png",
        )

    summary = build_summary(metrics_df)
    save_summary(summary, reports_dir / f"{args.run_tag}_report.md")

    print("Baseline эксперименты завершены.")
    print(f"Метрики сохранены: {reports_dir / f'{args.run_tag}_metrics.csv'}")
    print(f"Сводка экспериментов сохранена: {reports_dir / f'{args.run_tag}_experiments_summary.csv'}")
    print(f"Графики сохранены в папке: {reports_dir}")
    print(f"Отчёт сохранён: {reports_dir / f'{args.run_tag}_report.md'}")


if __name__ == "__main__":
    train_baseline_models()