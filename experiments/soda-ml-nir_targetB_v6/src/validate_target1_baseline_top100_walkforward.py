from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


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
FEATURES_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "exp_05_target1_topn_importance"
    / "baseline_feature_set_top100.csv"
)

TARGET_COL = "target_value"
TIME_COL = "target_timestamp_for_scada"

EXPERIMENT_NAME = "target1_baseline_top100_walkforward"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline_top100_walkforward"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 4

RF_PARAMS = {
    "n_estimators": 300,
    "random_state": 42,
    "n_jobs": -1,
}


# =========================
# HELPERS
# =========================
def calc_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_feature_list(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл с baseline feature set:\n{path}"
        )

    df = pd.read_csv(path)

    if "feature" not in df.columns:
        raise ValueError(
            f"В файле должен быть столбец 'feature':\n{path}"
        )

    features = df["feature"].dropna().astype(str).tolist()

    if len(features) == 0:
        raise ValueError("Список baseline-признаков пуст")

    # удалим возможные дубли, сохранив порядок
    unique_features = list(dict.fromkeys(features))

    if len(unique_features) != len(features):
        print(
            f"[WARN] В feature set были дубликаты: "
            f"{len(features)} -> {len(unique_features)}"
        )

    return unique_features


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Не найден датасет:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Не найдена целевая колонка: {TARGET_COL}")
    if TIME_COL not in df.columns:
        raise ValueError(f"Не найдена временная колонка: {TIME_COL}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, TARGET_COL]).copy()
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    return df


def prepare_X_y(df, selected_features):
    missing_features = [f for f in selected_features if f not in df.columns]
    if missing_features:
        preview = "\n".join(missing_features[:20])
        raise ValueError(
            "В датасете отсутствуют признаки из baseline feature set:\n"
            f"{preview}"
        )

    X = df[selected_features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    non_numeric_features = [
        col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
    ]
    if non_numeric_features:
        preview = "\n".join(non_numeric_features[:20])
        raise ValueError(
            "Часть baseline-признаков не является числовой:\n"
            f"{preview}"
        )

    y = df[TARGET_COL].copy()
    return X, y


def plot_fold_metric(df_metrics, metric_col, output_path, title, ylabel):
    plt.figure(figsize=(8, 5))
    plt.bar(df_metrics["fold"], df_metrics[metric_col])
    plt.title(title)
    plt.xlabel("Фолд")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_report_md(
    n_rows,
    n_features,
    fold_results_df,
    summary_stats,
):
    lines = []
    lines.append(f"# Отчёт: {EXPERIMENT_NAME}")
    lines.append("")
    lines.append("## Конфигурация")
    lines.append(f"- Датасет: `{DATA_PATH}`")
    lines.append(f"- Feature set: `{FEATURES_PATH}`")
    lines.append(f"- Число строк: {n_rows}")
    lines.append(f"- Число признаков: {n_features}")
    lines.append("- Модель: Random Forest")
    lines.append(f"- Параметры RF: `{RF_PARAMS}`")
    lines.append(f"- Временная валидация: TimeSeriesSplit(n_splits={N_SPLITS})")
    lines.append("")
    lines.append("## Метрики по фолдам")
    lines.append("Чем меньше MAE/RMSE и выше R², тем лучше.")
    lines.append("")

    for _, row in fold_results_df.iterrows():
        lines.append(f"### {row['fold']}")
        lines.append(f"- train_size: {int(row['train_size'])}")
        lines.append(f"- test_size: {int(row['test_size'])}")
        lines.append(f"- MAE: {row['MAE']:.6f}")
        lines.append(f"- RMSE: {row['RMSE']:.6f}")
        lines.append(f"- R2: {row['R2']:.6f}")
        lines.append("")

    lines.append("## Сводка")
    lines.append(f"- mean MAE: {summary_stats['MAE_mean']:.6f}")
    lines.append(f"- std MAE: {summary_stats['MAE_std']:.6f}")
    lines.append(f"- mean RMSE: {summary_stats['RMSE_mean']:.6f}")
    lines.append(f"- std RMSE: {summary_stats['RMSE_std']:.6f}")
    lines.append(f"- mean R2: {summary_stats['R2_mean']:.6f}")
    lines.append(f"- std R2: {summary_stats['R2_std']:.6f}")
    lines.append("")
    lines.append("## Интерпретация")
    lines.append(
        "- Если RMSE не разваливается по фолдам, а R² не уходит системно в отрицательную зону, baseline можно считать рабоче устойчивым."
    )
    return "\n".join(lines)


def build_summary_text(n_rows, n_features, fold_results_df, summary_stats):
    lines = []
    lines.append(f"Эксперимент: {EXPERIMENT_NAME}")
    lines.append(f"Датасет: {DATA_PATH}")
    lines.append(f"Feature set: {FEATURES_PATH}")
    lines.append(f"Число строк: {n_rows}")
    lines.append(f"Число признаков: {n_features}")
    lines.append(f"TimeSeriesSplit n_splits: {N_SPLITS}")
    lines.append("")

    for _, row in fold_results_df.iterrows():
        lines.append(f"{row['fold']}")
        lines.append(f"  train_size: {int(row['train_size'])}")
        lines.append(f"  test_size: {int(row['test_size'])}")
        lines.append(f"  MAE: {row['MAE']:.6f}")
        lines.append(f"  RMSE: {row['RMSE']:.6f}")
        lines.append(f"  R2: {row['R2']:.6f}")
        lines.append("")

    lines.append("Сводка")
    lines.append(f"  mean MAE: {summary_stats['MAE_mean']:.6f}")
    lines.append(f"  std MAE: {summary_stats['MAE_std']:.6f}")
    lines.append(f"  mean RMSE: {summary_stats['RMSE_mean']:.6f}")
    lines.append(f"  std RMSE: {summary_stats['RMSE_std']:.6f}")
    lines.append(f"  mean R2: {summary_stats['R2_mean']:.6f}")
    lines.append(f"  std R2: {summary_stats['R2_std']:.6f}")
    lines.append("")
    lines.append("Правило интерпретации:")
    lines.append("- baseline считается более устойчивым, если нет сильного развала метрик между фолдами")
    lines.append("- mean RMSE должен быть близок к текущему рабочему уровню")
    lines.append("- R2 не должен системно уходить в отрицательную зону")

    return "\n".join(lines)


def build_conclusion_text(summary_stats, fold_results_df):
    negative_r2_count = int((fold_results_df["R2"] < 0).sum())
    total_folds = int(len(fold_results_df))

    lines = []
    lines.append("КРАТКИЙ ВЫВОД")
    lines.append("")
    lines.append(f"mean RMSE = {summary_stats['RMSE_mean']:.6f}")
    lines.append(f"std RMSE = {summary_stats['RMSE_std']:.6f}")
    lines.append(f"mean R2 = {summary_stats['R2_mean']:.6f}")
    lines.append(f"std R2 = {summary_stats['R2_std']:.6f}")
    lines.append(f"Фолдов с отрицательным R2: {negative_r2_count} из {total_folds}")
    lines.append("")

    if negative_r2_count == 0:
        lines.append("Предварительный вывод: baseline выглядит устойчиво.")
    elif negative_r2_count <= 1:
        lines.append("Предварительный вывод: baseline условно устойчив, но нужен аккуратный вывод в тексте НИР.")
    else:
        lines.append("Предварительный вывод: устойчивость baseline слабая, переносить дальше нужно осторожно.")

    return "\n".join(lines)


# =========================
# MAIN
# =========================
def main():
    print(f"[INFO] Загрузка датасета: {DATA_PATH}")
    print(f"[INFO] Загрузка baseline feature set: {FEATURES_PATH}")

    selected_features = load_feature_list(FEATURES_PATH)
    df = load_dataset()
    X, y = prepare_X_y(df, selected_features)

    print(f"[INFO] Число строк после очистки: {len(df)}")
    print(f"[INFO] Число baseline-признаков: {len(selected_features)}")

    if len(df) < 20:
        raise ValueError("Слишком мало строк для walk-forward проверки")

    if N_SPLITS >= len(df):
        raise ValueError(
            f"N_SPLITS={N_SPLITS} слишком велик для числа строк {len(df)}"
        )

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    fold_results = []
    all_predictions = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        fold_name = f"fold_{fold_idx}"
        print(f"[INFO] Обработка {fold_name}")

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        # train-only imputation
        train_medians = X_train.median()
        X_train = X_train.fillna(train_medians).fillna(0.0)
        X_test = X_test.fillna(train_medians).fillna(0.0)

        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = calc_metrics(y_test, preds)

        fold_results.append(
            {
                "fold": fold_name,
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )

        fold_pred_df = pd.DataFrame(
            {
                "fold": fold_name,
                TIME_COL: df.iloc[test_idx][TIME_COL].values,
                "y_true": y_test.values,
                "y_pred_rf": preds,
            }
        )
        all_predictions.append(fold_pred_df)

        print(
            f"[RESULT] {fold_name} | "
            f"train={len(X_train)} | test={len(X_test)} | "
            f"MAE={metrics['MAE']:.6f} | "
            f"RMSE={metrics['RMSE']:.6f} | "
            f"R2={metrics['R2']:.6f}"
        )

    fold_results_df = pd.DataFrame(fold_results)
    predictions_df = pd.concat(all_predictions, axis=0, ignore_index=True)

    summary_stats = {
        "MAE_mean": float(fold_results_df["MAE"].mean()),
        "MAE_std": float(fold_results_df["MAE"].std(ddof=0)),
        "RMSE_mean": float(fold_results_df["RMSE"].mean()),
        "RMSE_std": float(fold_results_df["RMSE"].std(ddof=0)),
        "R2_mean": float(fold_results_df["R2"].mean()),
        "R2_std": float(fold_results_df["R2"].std(ddof=0)),
    }

    fold_results_df.to_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "walkforward_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        {
            "experiment_name": EXPERIMENT_NAME,
            "dataset_path": str(DATA_PATH),
            "features_path": str(FEATURES_PATH),
            "target_col": TARGET_COL,
            "time_col": TIME_COL,
            "n_rows": int(len(df)),
            "n_features": int(len(selected_features)),
            "n_splits": N_SPLITS,
            "rf_params": RF_PARAMS,
            "folds": fold_results,
            "summary": summary_stats,
        },
        OUTPUT_DIR / "walkforward_summary.json",
    )

    plot_fold_metric(
        df_metrics=fold_results_df,
        metric_col="MAE",
        output_path=OUTPUT_DIR / "mae_by_fold.png",
        title="MAE по временным фолдам",
        ylabel="MAE",
    )
    plot_fold_metric(
        df_metrics=fold_results_df,
        metric_col="RMSE",
        output_path=OUTPUT_DIR / "rmse_by_fold.png",
        title="RMSE по временным фолдам",
        ylabel="RMSE",
    )
    plot_fold_metric(
        df_metrics=fold_results_df,
        metric_col="R2",
        output_path=OUTPUT_DIR / "r2_by_fold.png",
        title="R² по временным фолдам",
        ylabel="R²",
    )

    report_text = build_report_md(
        n_rows=len(df),
        n_features=len(selected_features),
        fold_results_df=fold_results_df,
        summary_stats=summary_stats,
    )
    with open(OUTPUT_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    summary_text = build_summary_text(
        n_rows=len(df),
        n_features=len(selected_features),
        fold_results_df=fold_results_df,
        summary_stats=summary_stats,
    )
    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    conclusion_text = build_conclusion_text(
        summary_stats=summary_stats,
        fold_results_df=fold_results_df,
    )
    with open(OUTPUT_DIR / "conclusion.txt", "w", encoding="utf-8") as f:
        f.write(conclusion_text)

    print("")
    print("[INFO] Готово.")
    print(f"[INFO] Результаты сохранены в: {OUTPUT_DIR}")
    print("[INFO] Основные файлы:")
    print(f"  - {OUTPUT_DIR / 'fold_metrics.csv'}")
    print(f"  - {OUTPUT_DIR / 'walkforward_summary.json'}")
    print(f"  - {OUTPUT_DIR / 'summary.txt'}")
    print(f"  - {OUTPUT_DIR / 'conclusion.txt'}")


if __name__ == "__main__":
    main()