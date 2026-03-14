from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
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

# === ВАЖНО ===
# При необходимости поправьте только эти 3 пути / названия колонок.
DATA_PATH = PROJECT_ROOT / "launch_targetB_k1" / "dataset_targetB_baseline_v1.csv"
FEATURES_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "exp_05_target1_topn_importance"
    / "baseline_feature_set_top100.csv"
)

TARGET_COL = "target_value"
TIME_COL = "target_timestamp_for_scada"

EXPERIMENT_NAME = "targetB_baseline_top100_transfer"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "targetB_baseline_top100"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.20

RF_PARAMS = {
    "n_estimators": 300,
    "random_state": 42,
    "n_jobs": -1,
}


# =========================
# HELPERS
# =========================
def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def calc_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_true, y_pred)),
    }


def load_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл baseline feature set:\n{path}"
        )

    df = pd.read_csv(path)

    if "feature" not in df.columns:
        raise ValueError(
            f"В файле должен быть столбец 'feature':\n{path}"
        )

    features = df["feature"].dropna().astype(str).tolist()
    if len(features) == 0:
        raise ValueError("Список baseline-признаков пуст")

    # убираем дубли, сохраняем порядок
    features = list(dict.fromkeys(features))
    return features


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден датасет:\n{path}")

    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Не найдена целевая колонка: {TARGET_COL}")
    if TIME_COL not in df.columns:
        raise ValueError(f"Не найдена временная колонка: {TIME_COL}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, TARGET_COL]).copy()
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(
            f"Слишком мало строк после очистки: {len(df)}"
        )

    return df


def split_features(feature_list: list[str], df_columns: list[str]):
    df_columns_set = set(df_columns)

    found_features = [f for f in feature_list if f in df_columns_set]
    missing_features = [f for f in feature_list if f not in df_columns_set]

    return found_features, missing_features


def validate_numeric_features(df: pd.DataFrame, features: list[str]):
    non_numeric = [
        col for col in features
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    return non_numeric


def make_time_split(df: pd.DataFrame, test_size: float):
    n_rows = len(df)
    test_count = max(1, int(round(n_rows * test_size)))
    train_count = n_rows - test_count

    if train_count <= 0:
        raise ValueError(
            f"Некорректный split: train_count={train_count}, test_count={test_count}"
        )

    train_df = df.iloc[:train_count].copy()
    test_df = df.iloc[train_count:].copy()

    return train_df, test_df


def build_report_md(
    n_rows: int,
    n_requested_features: int,
    n_found_features: int,
    n_missing_features: int,
    train_size: int,
    test_size: int,
    metrics: dict,
    used_features_path: Path,
    missing_features_path: Path,
) -> str:
    lines = []
    lines.append(f"# Отчёт: {EXPERIMENT_NAME}")
    lines.append("")
    lines.append("## Конфигурация")
    lines.append(f"- Датасет: `{DATA_PATH}`")
    lines.append(f"- Baseline feature set: `{FEATURES_PATH}`")
    lines.append(f"- Число строк: {n_rows}")
    lines.append(f"- Запрошено baseline-признаков: {n_requested_features}")
    lines.append(f"- Найдено признаков в B1: {n_found_features}")
    lines.append(f"- Отсутствует признаков: {n_missing_features}")
    lines.append(f"- Train/Test split: {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} по времени")
    lines.append(f"- Train size: {train_size}")
    lines.append(f"- Test size: {test_size}")
    lines.append("- Модель: Random Forest")
    lines.append(f"- Параметры RF: `{RF_PARAMS}`")
    lines.append("")
    lines.append("## Метрики")
    lines.append(f"- MAE: {metrics['MAE']:.6f}")
    lines.append(f"- RMSE: {metrics['RMSE']:.6f}")
    lines.append(f"- R2: {metrics['R2']:.6f}")
    lines.append("")
    lines.append("## Артефакты")
    lines.append(f"- Использованные признаки: `{used_features_path}`")
    lines.append(f"- Отсутствующие признаки: `{missing_features_path}`")
    lines.append("")
    lines.append("## Вывод")
    lines.append("- Это первый перенос рабочего baseline target1 -> B1.")
    lines.append("- Проверяется fixed top100 baseline logic без нового подбора признаков.")
    lines.append("- Если найдено большинство признаков и метрики вменяемые, перенос выполнен корректно.")
    return "\n".join(lines)


def build_summary_text(
    n_rows: int,
    n_requested_features: int,
    n_found_features: int,
    n_missing_features: int,
    train_size: int,
    test_size: int,
    metrics: dict,
) -> str:
    lines = []
    lines.append(f"Эксперимент: {EXPERIMENT_NAME}")
    lines.append(f"Датасет: {DATA_PATH}")
    lines.append(f"Baseline feature set: {FEATURES_PATH}")
    lines.append(f"Число строк: {n_rows}")
    lines.append(f"Запрошено baseline-признаков: {n_requested_features}")
    lines.append(f"Найдено признаков в B1: {n_found_features}")
    lines.append(f"Отсутствует признаков: {n_missing_features}")
    lines.append(f"Train size: {train_size}")
    lines.append(f"Test size: {test_size}")
    lines.append("")
    lines.append("Метрики:")
    lines.append(f"  MAE: {metrics['MAE']:.6f}")
    lines.append(f"  RMSE: {metrics['RMSE']:.6f}")
    lines.append(f"  R2: {metrics['R2']:.6f}")
    lines.append("")
    lines.append("Интерпретация:")
    lines.append("- если найдено 80+ признаков из baseline top100, перенос логики можно считать близким к исходному")
    lines.append("- если R2 не провален и RMSE выглядит вменяемо, baseline можно использовать как первый B1-ориентир")
    return "\n".join(lines)


def plot_predictions(y_true, y_pred, output_path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(y_true)), y_true, label="Факт")
    plt.plot(range(len(y_pred)), y_pred, label="Прогноз RF")
    plt.title("B1 baseline top100: факт vs прогноз на test")
    plt.xlabel("Индекс точки в test")
    plt.ylabel(TARGET_COL)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# MAIN
# =========================
def main():
    print(f"[INFO] Загрузка B1 датасета: {DATA_PATH}")
    print(f"[INFO] Загрузка baseline feature set: {FEATURES_PATH}")

    baseline_features = load_feature_list(FEATURES_PATH)
    df = load_dataset(DATA_PATH)

    print(f"[INFO] Число строк после очистки: {len(df)}")
    print(f"[INFO] Число baseline-признаков в списке: {len(baseline_features)}")

    found_features, missing_features = split_features(
        baseline_features,
        list(df.columns),
    )

    print(f"[INFO] Найдено признаков в B1: {len(found_features)}")
    print(f"[INFO] Отсутствует признаков: {len(missing_features)}")

    if len(found_features) == 0:
        raise ValueError(
            "В датасете B1 не найдено ни одного признака из baseline top100"
        )

    non_numeric_features = validate_numeric_features(df, found_features)
    if non_numeric_features:
        preview = "\n".join(non_numeric_features[:20])
        raise ValueError(
            "Среди найденных baseline-признаков есть нечисловые:\n"
            f"{preview}"
        )

    train_df, test_df = make_time_split(df, TEST_SIZE)

    X_train = train_df[found_features].copy()
    X_test = test_df[found_features].copy()
    y_train = train_df[TARGET_COL].copy()
    y_test = test_df[TARGET_COL].copy()

    # train-only imputation
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians).fillna(0.0)
    X_test = X_test.fillna(train_medians).fillna(0.0)

    print(f"[INFO] Train size: {len(X_train)}")
    print(f"[INFO] Test size: {len(X_test)}")
    print(f"[INFO] Обучение Random Forest...")

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = calc_metrics(y_test, preds)

    print("[RESULT] Метрики на B1:")
    print(f"  MAE  = {metrics['MAE']:.6f}")
    print(f"  RMSE = {metrics['RMSE']:.6f}")
    print(f"  R2   = {metrics['R2']:.6f}")

    # =========================
    # SAVE ARTIFACTS
    # =========================
    used_features_path = OUTPUT_DIR / "used_features_from_top100.csv"
    missing_features_path = OUTPUT_DIR / "missing_features_from_top100.csv"
    predictions_path = OUTPUT_DIR / "rf_test_predictions.csv"
    metrics_path = OUTPUT_DIR / "metrics.json"
    summary_json_path = OUTPUT_DIR / "run_summary.json"
    report_path = OUTPUT_DIR / "report.md"
    summary_txt_path = OUTPUT_DIR / "summary.txt"
    plot_path = OUTPUT_DIR / "test_predictions_plot.png"
    model_path = OUTPUT_DIR / "random_forest.joblib"

    pd.DataFrame({"feature": found_features}).to_csv(
        used_features_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"feature": missing_features}).to_csv(
        missing_features_path,
        index=False,
        encoding="utf-8-sig",
    )

    pred_df = pd.DataFrame(
        {
            TIME_COL: test_df[TIME_COL].values,
            "y_true": y_test.values,
            "y_pred_rf": preds,
        }
    )
    pred_df.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    save_json(metrics, metrics_path)

    run_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "dataset_path": str(DATA_PATH),
        "baseline_features_path": str(FEATURES_PATH),
        "target_col": TARGET_COL,
        "time_col": TIME_COL,
        "n_rows": int(len(df)),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "n_requested_features": int(len(baseline_features)),
        "n_found_features": int(len(found_features)),
        "n_missing_features": int(len(missing_features)),
        "rf_params": RF_PARAMS,
        "metrics": metrics,
    }
    save_json(run_summary, summary_json_path)

    report_text = build_report_md(
        n_rows=len(df),
        n_requested_features=len(baseline_features),
        n_found_features=len(found_features),
        n_missing_features=len(missing_features),
        train_size=len(train_df),
        test_size=len(test_df),
        metrics=metrics,
        used_features_path=used_features_path,
        missing_features_path=missing_features_path,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    summary_text = build_summary_text(
        n_rows=len(df),
        n_requested_features=len(baseline_features),
        n_found_features=len(found_features),
        n_missing_features=len(missing_features),
        train_size=len(train_df),
        test_size=len(test_df),
        metrics=metrics,
    )
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    plot_predictions(
        y_true=y_test.values,
        y_pred=preds,
        output_path=plot_path,
    )

    try:
        import joblib
        joblib.dump(model, model_path)
    except Exception as e:
        print(f"[WARN] Не удалось сохранить model.joblib: {e}")

    print("")
    print("[INFO] Готово. Результаты сохранены в:")
    print(f"  {OUTPUT_DIR}")
    print("[INFO] Основные файлы:")
    print(f"  - {used_features_path.name}")
    print(f"  - {missing_features_path.name}")
    print(f"  - {predictions_path.name}")
    print(f"  - {metrics_path.name}")
    print(f"  - {summary_txt_path.name}")
    print(f"  - {report_path.name}")


if __name__ == "__main__":
    main()