from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# MATPLOTLIB / WINDOWS FONT
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "launch_target1_k1" / "dataset_target1_baseline_v1.csv"
TARGET_COL = "target_value"
TIME_COL = "target_timestamp_for_scada"

EXPERIMENT_NAME = "target1_exp04_mean_last_only"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp_04_target1_mean_last_only"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_EXP02_RF = {
    "RMSE": 4.116668,
    "R2": 0.113823,
}


# =========================
# FEATURE RULES
# =========================
EXCLUDED_EXACT = {
    TARGET_COL,
    TIME_COL,
}

EXCLUDED_SUBSTRINGS = [
    "target",
]

ALLOWED_PREFIXES = [
    "w60__mean__",
    "w60__last__",
]


# =========================
# HELPERS
# =========================
def is_service_or_target_related(col: str) -> bool:
    if col in EXCLUDED_EXACT:
        return True
    col_l = col.lower()
    return any(x in col_l for x in EXCLUDED_SUBSTRINGS)


def is_allowed_feature(col: str) -> bool:
    if is_service_or_target_related(col):
        return False
    return any(col.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def calc_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def prepare_X(X: pd.DataFrame) -> pd.DataFrame:
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    medians = X.median()
    X = X.fillna(medians)
    X = X.fillna(0.0)

    return X


def get_model_display_name(model_name: str) -> str:
    mapping = {
        "ridge": "Ridge",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
    }
    return mapping.get(model_name, model_name)


def save_metrics_csv(results: dict, output_dir: Path) -> pd.DataFrame:
    rows = []

    for model_name, metrics in results["models"].items():
        rows.append(
            {
                "experiment_name": results["experiment_name"],
                "model": model_name,
                "model_display_name": get_model_display_name(model_name),
                "n_rows": results["n_rows"],
                "n_features": results["n_features"],
                "train_size": results["split"]["train_size"],
                "test_size": results["split"]["test_size"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(
        output_dir / "metrics_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return df_metrics


def save_experiments_summary_csv(results: dict, output_dir: Path) -> None:
    rf_metrics = results["models"].get("random_forest", {})
    gb_metrics = results["models"].get("gradient_boosting", {})
    ridge_metrics = results["models"].get("ridge", {})

    summary_row = {
        "experiment_name": results["experiment_name"],
        "n_rows": results["n_rows"],
        "n_features": results["n_features"],
        "rf_MAE": rf_metrics.get("MAE"),
        "rf_RMSE": rf_metrics.get("RMSE"),
        "rf_R2": rf_metrics.get("R2"),
        "gb_MAE": gb_metrics.get("MAE"),
        "gb_RMSE": gb_metrics.get("RMSE"),
        "gb_R2": gb_metrics.get("R2"),
        "ridge_MAE": ridge_metrics.get("MAE"),
        "ridge_RMSE": ridge_metrics.get("RMSE"),
        "ridge_R2": ridge_metrics.get("R2"),
    }

    pd.DataFrame([summary_row]).to_csv(
        output_dir / "experiments_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_metric_comparison(df_metrics: pd.DataFrame, metric_name: str, output_path: Path) -> None:
    metric_name_lower = metric_name.lower()

    if metric_name_lower == "mae":
        metric_col = "MAE"
        title = "Сравнение моделей по MAE"
        ylabel = "MAE"
    elif metric_name_lower == "rmse":
        metric_col = "RMSE"
        title = "Сравнение моделей по RMSE"
        ylabel = "RMSE"
    elif metric_name_lower == "r2":
        metric_col = "R2"
        title = "Сравнение моделей по R²"
        ylabel = "R²"
    else:
        raise ValueError(f"Неизвестная метрика для графика: {metric_name}")

    labels = df_metrics["model_display_name"]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, df_metrics[metric_col])
    plt.title(title)
    plt.xlabel("Модель")
    plt.ylabel(ylabel)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rf_feature_importance(model, feature_names: list[str], output_path: Path, top_n: int = 20) -> None:
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    fi_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    fi_df.to_csv(
        output_path.parent / "rf_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top_df = fi_df.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.title(f"Важность признаков Random Forest (top {top_n})")
    plt.xlabel("Важность")
    plt.ylabel("Признак")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_report_md(results: dict) -> str:
    lines = []
    lines.append(f"# Отчёт: {results['experiment_name']}")
    lines.append("")
    lines.append("## Датасет")
    lines.append(f"- Путь: `{results['dataset_path']}`")
    lines.append(f"- Число строк: {results['n_rows']}")
    lines.append(f"- Число признаков: {results['n_features']}")
    lines.append(f"- Целевая колонка: `{results['target_col']}`")
    lines.append(f"- Временная колонка: `{results['time_col']}`")
    lines.append("")
    lines.append("## Разбиение")
    lines.append(f"- Размер train: {results['split']['train_size']}")
    lines.append(f"- Размер test: {results['split']['test_size']}")
    lines.append(f"- Метод: {results['split']['method']}")
    lines.append("")
    lines.append("## Результаты моделей")
    lines.append("Чем меньше MAE/RMSE и выше R2, тем лучше.")
    lines.append("")

    best_model_name = None
    best_model_mae = None

    for model_name, metrics in results["models"].items():
        model_display_name = get_model_display_name(model_name)
        lines.append(f"### {model_display_name}")
        lines.append(f"- MAE: {metrics['MAE']:.6f}")
        lines.append(f"- RMSE: {metrics['RMSE']:.6f}")
        lines.append(f"- R2: {metrics['R2']:.6f}")
        lines.append("")

        if best_model_mae is None or metrics["MAE"] < best_model_mae:
            best_model_mae = metrics["MAE"]
            best_model_name = model_display_name

    if "comparison_vs_exp02_rf" in results:
        cmp_ = results["comparison_vs_exp02_rf"]
        lines.append("## Сравнение с exp_02 (Random Forest)")
        lines.append(f"- RF RMSE в exp_02: {results['baseline_exp02_rf']['RMSE']:.6f}")
        lines.append(f"- RF RMSE в текущем эксперименте: {results['models']['random_forest']['RMSE']:.6f}")
        lines.append(f"- delta RMSE: {cmp_['delta_RMSE']:.6f}")
        lines.append(f"- RF R2 в exp_02: {results['baseline_exp02_rf']['R2']:.6f}")
        lines.append(f"- RF R2 в текущем эксперименте: {results['models']['random_forest']['R2']:.6f}")
        lines.append(f"- delta R2: {cmp_['delta_R2']:.6f}")
        lines.append("")

    if best_model_name is not None:
        lines.append("## Лучшая конфигурация по MAE")
        lines.append(f"- {best_model_name}: MAE={best_model_mae:.6f}")
        lines.append("")

    rf_metrics = results["models"].get("random_forest")
    if rf_metrics is not None:
        lines.append("## Вывод")
        lines.append("- Random Forest остаётся основной baseline-моделью для сравнения.")
        if "comparison_vs_exp02_rf" in results:
            delta_rmse = results["comparison_vs_exp02_rf"]["delta_RMSE"]
            delta_r2 = results["comparison_vs_exp02_rf"]["delta_R2"]

            if delta_rmse <= 0 and delta_r2 >= 0:
                lines.append("- Текущий эксперимент не хуже exp_02 по ключевым метрикам.")
            else:
                lines.append("- Текущий эксперимент уступает exp_02 по ключевым метрикам, но полезен как упрощённый вариант.")
    else:
        lines.append("## Вывод")
        lines.append("- Результаты Random Forest отсутствуют.")

    return "\n".join(lines)


def save_report_md(results: dict, output_dir: Path) -> None:
    report_text = build_report_md(results)
    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report_text)


# =========================
# MAIN
# =========================
def main():
    print(f"[INFO] Загрузка датасета: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    if TIME_COL not in df.columns:
        raise ValueError(f"Не найдена временная колонка: {TIME_COL}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Не найдена целевая колонка: {TARGET_COL}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, TARGET_COL]).copy()
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    feature_cols = [c for c in df.columns if is_allowed_feature(c)]

    print(f"[INFO] Выбрано исходных признаков: {len(feature_cols)}")
    if len(feature_cols) == 0:
        raise ValueError("Не выбраны признаки. Проверь правила фильтрации.")

    X = df[feature_cols].copy()
    X = prepare_X(X)
    y = df[TARGET_COL].copy()

    final_feature_cols = list(X.columns)

    print(f"[INFO] Итоговых числовых признаков: {len(final_feature_cols)}")
    if len(final_feature_cols) == 0:
        raise ValueError("После предобработки не осталось числовых признаков.")

    split_idx = int(len(df) * 0.8)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError(f"Некорректный индекс split: {split_idx} для размера датасета {len(df)}")

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[INFO] Размер train: {len(X_train)} | Размер test: {len(X_test)}")

    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=42,
        ),
    }

    results = {
        "experiment_name": EXPERIMENT_NAME,
        "dataset_path": str(DATA_PATH),
        "target_col": TARGET_COL,
        "time_col": TIME_COL,
        "n_rows": int(len(df)),
        "n_features": int(len(final_feature_cols)),
        "feature_columns": final_feature_cols,
        "split": {
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "method": "time_based_80_20_no_shuffle",
        },
        "models": {},
        "baseline_exp02_rf": BASELINE_EXP02_RF,
    }

    trained_models = {}

    for model_name, model in models.items():
        print(f"[INFO] Обучение модели: {get_model_display_name(model_name)}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = calc_metrics(y_test, preds)

        trained_models[model_name] = model
        results["models"][model_name] = metrics

        print(f"[RESULT] {model_name}: {metrics}")

        predictions_df = pd.DataFrame(
            {
                TIME_COL: df.iloc[split_idx:][TIME_COL].values,
                "y_true": y_test.values,
                f"y_pred_{model_name}": preds,
            }
        )
        predictions_df.to_csv(
            OUTPUT_DIR / f"{model_name}_test_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if model_name == "random_forest":
            results["comparison_vs_exp02_rf"] = {
                "delta_RMSE": float(metrics["RMSE"] - BASELINE_EXP02_RF["RMSE"]),
                "delta_R2": float(metrics["R2"] - BASELINE_EXP02_RF["R2"]),
            }

    pd.DataFrame({"feature": final_feature_cols}).to_csv(
        OUTPUT_DIR / "selected_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_json(results, OUTPUT_DIR / "metrics.json")

    for model_name, model in trained_models.items():
        joblib.dump(model, OUTPUT_DIR / f"{model_name}.joblib")

    df_metrics = save_metrics_csv(results, OUTPUT_DIR)
    save_experiments_summary_csv(results, OUTPUT_DIR)

    plot_metric_comparison(
        df_metrics=df_metrics,
        metric_name="mae",
        output_path=OUTPUT_DIR / "mae_comparison.png",
    )
    plot_metric_comparison(
        df_metrics=df_metrics,
        metric_name="rmse",
        output_path=OUTPUT_DIR / "rmse_comparison.png",
    )
    plot_metric_comparison(
        df_metrics=df_metrics,
        metric_name="r2",
        output_path=OUTPUT_DIR / "r2_comparison.png",
    )

    if "random_forest" in trained_models:
        plot_rf_feature_importance(
            model=trained_models["random_forest"],
            feature_names=final_feature_cols,
            output_path=OUTPUT_DIR / "rf_feature_importance.png",
            top_n=20,
        )

    save_report_md(results, OUTPUT_DIR)

    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Эксперимент: {EXPERIMENT_NAME}\n")
        f.write(f"Датасет: {DATA_PATH}\n")
        f.write(f"Число строк: {len(df)}\n")
        f.write(f"Число признаков: {len(final_feature_cols)}\n")
        f.write(f"Размер train: {len(X_train)}\n")
        f.write(f"Размер test: {len(X_test)}\n\n")

        f.write("Результаты моделей:\n")
        for model_name, metrics in results["models"].items():
            f.write(f"{get_model_display_name(model_name)}\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.6f}\n")
            f.write("\n")

        if "comparison_vs_exp02_rf" in results:
            f.write("Сравнение с exp_02 (Random Forest)\n")
            f.write(f"  RF RMSE в exp_02: {BASELINE_EXP02_RF['RMSE']:.6f}\n")
            f.write(f"  RF RMSE в exp_04: {results['models']['random_forest']['RMSE']:.6f}\n")
            f.write(f"  delta RMSE: {results['comparison_vs_exp02_rf']['delta_RMSE']:.6f}\n")
            f.write(f"  RF R2 в exp_02: {BASELINE_EXP02_RF['R2']:.6f}\n")
            f.write(f"  RF R2 в exp_04: {results['models']['random_forest']['R2']:.6f}\n")
            f.write(f"  delta R2: {results['comparison_vs_exp02_rf']['delta_R2']:.6f}\n")

    print(f"[INFO] Готово. Результаты сохранены в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()