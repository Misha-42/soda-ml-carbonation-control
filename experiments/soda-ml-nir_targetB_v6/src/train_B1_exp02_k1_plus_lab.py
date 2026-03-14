from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "merged_dataset_B1_v1.xlsx"
SHEET_NAME = "MERGED_B1"

TARGET_COL = "target_B1_sv_NH3_susp"
DATE_COL = "lab_date"
SHIFT_COL = "shift"
TIME_COL = "time_idx"

EXPERIMENT_NAME = "B1_exp02_k1_plus_lab_rf_v1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "B1_exp02_k1_plus_lab_rf_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RF_PARAMS = {
    "n_estimators": 300,
    "random_state": 42,
    "n_jobs": -1,
}

LAB_FEATURES_CANDIDATES = [
    "cl_susp",
    "co2_after_KLPK",
    "bf_moisture",
    "bf_chlorides",
]


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


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def shift_to_hours(value):
    if pd.isna(value):
        return 12

    s = str(value).strip().lower()

    if s in {"1", "1.0", "i", "1 смена", "смена 1"}:
        return 8
    if s in {"2", "2.0", "ii", "2 смена", "смена 2"}:
        return 20

    if "1" in s:
        return 8
    if "2" in s:
        return 20

    return 12


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Не найден файл:\n{DATA_PATH}")

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    required_cols = [TARGET_COL, DATE_COL, SHIFT_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"В файле отсутствуют обязательные колонки: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET_COL]).copy()

    df[TIME_COL] = df[DATE_COL] + pd.to_timedelta(
        df[SHIFT_COL].apply(shift_to_hours),
        unit="h",
    )

    df = df.dropna(subset=[TIME_COL]).copy()
    df = df.sort_values([TIME_COL]).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(f"Слишком мало строк после очистки: {len(df)}")

    return df


def select_feature_columns(df: pd.DataFrame):
    k1_features = [c for c in df.columns if c.startswith("k1_")]
    lab_features = [c for c in LAB_FEATURES_CANDIDATES if c in df.columns]

    feature_cols = k1_features + lab_features
    feature_cols = list(dict.fromkeys(feature_cols))

    if not feature_cols:
        raise ValueError("Не найдены признаки для B1 exp_02")

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    dropped_non_numeric = [c for c in feature_cols if c not in numeric_cols]

    if not numeric_cols:
        raise ValueError("После отбора не осталось числовых признаков")

    used_k1 = [c for c in numeric_cols if c.startswith("k1_")]
    used_lab = [c for c in numeric_cols if c in LAB_FEATURES_CANDIDATES]

    return numeric_cols, dropped_non_numeric, used_k1, used_lab


def build_report_md(n_rows, n_features, n_k1, n_lab, train_size, test_size, metrics):
    lines = []
    lines.append(f"# Отчёт: {EXPERIMENT_NAME}")
    lines.append("")
    lines.append("## Конфигурация")
    lines.append(f"- Датасет: `{DATA_PATH}`")
    lines.append(f"- Лист: `{SHEET_NAME}`")
    lines.append(f"- Target: `{TARGET_COL}`")
    lines.append(f"- Proxy time: `{TIME_COL}` = `lab_date + shift`")
    lines.append(f"- Число строк: {n_rows}")
    lines.append(f"- Всего признаков: {n_features}")
    lines.append(f"- k1_* признаков: {n_k1}")
    lines.append(f"- lab признаков: {n_lab}")
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
    lines.append("## Интерпретация")
    lines.append("- Это диагностический запуск: k1_* + lab features.")
    lines.append("- Если метрики заметно улучшатся, значит baseline на одних k1_* был слишком узким.")
    return "\n".join(lines)


# =========================
# MAIN
# =========================
def main():
    print(f"[INFO] Загрузка файла: {DATA_PATH}")
    print(f"[INFO] Лист: {SHEET_NAME}")

    df = load_dataset()

    feature_cols, dropped_non_numeric, used_k1, used_lab = select_feature_columns(df)

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    split_idx = int(len(df) * 0.8)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError(
            f"Некорректный split_idx={split_idx} для размера датасета {len(df)}"
        )

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians).fillna(0.0)
    X_test = X_test.fillna(train_medians).fillna(0.0)

    print(f"[INFO] Число строк: {len(df)}")
    print(f"[INFO] Всего признаков: {len(feature_cols)}")
    print(f"[INFO] k1_* признаков: {len(used_k1)}")
    print(f"[INFO] lab признаков: {len(used_lab)}")
    print(f"[INFO] Train size: {len(X_train)} | Test size: {len(X_test)}")

    model = RandomForestRegressor(**RF_PARAMS)
    print("[INFO] Обучение Random Forest...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = calc_metrics(y_test, preds)

    print(
        f"[RESULT] MAE={metrics['MAE']:.6f} | "
        f"RMSE={metrics['RMSE']:.6f} | "
        f"R2={metrics['R2']:.6f}"
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    pd.DataFrame({"feature": feature_cols}).to_csv(
        OUTPUT_DIR / "selected_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"feature": used_k1}).to_csv(
        OUTPUT_DIR / "used_k1_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"feature": used_lab}).to_csv(
        OUTPUT_DIR / "used_lab_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if dropped_non_numeric:
        pd.DataFrame({"feature": dropped_non_numeric}).to_csv(
            OUTPUT_DIR / "dropped_non_numeric_features.csv",
            index=False,
            encoding="utf-8-sig",
        )

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            TIME_COL: df.iloc[split_idx:][TIME_COL].values,
            "y_true": y_test.values,
            "y_pred_rf": preds,
        }
    ).to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    joblib.dump(model, OUTPUT_DIR / "random_forest_B1_exp02.joblib")

    results = {
        "experiment_name": EXPERIMENT_NAME,
        "data_path": str(DATA_PATH),
        "sheet_name": SHEET_NAME,
        "target_col": TARGET_COL,
        "date_col": DATE_COL,
        "shift_col": SHIFT_COL,
        "time_col": TIME_COL,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "n_k1_features": int(len(used_k1)),
        "n_lab_features": int(len(used_lab)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "rf_params": RF_PARAMS,
        "metrics": metrics,
    }
    save_json(results, OUTPUT_DIR / "metrics.json")

    report_text = build_report_md(
        n_rows=len(df),
        n_features=len(feature_cols),
        n_k1=len(used_k1),
        n_lab=len(used_lab),
        train_size=len(X_train),
        test_size=len(X_test),
        metrics=metrics,
    )
    with open(OUTPUT_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Эксперимент: {EXPERIMENT_NAME}\n")
        f.write(f"Число строк: {len(df)}\n")
        f.write(f"Всего признаков: {len(feature_cols)}\n")
        f.write(f"k1_* признаков: {len(used_k1)}\n")
        f.write(f"lab признаков: {len(used_lab)}\n")
        f.write(f"Train size: {len(X_train)}\n")
        f.write(f"Test size: {len(X_test)}\n\n")
        f.write(f"MAE: {metrics['MAE']:.6f}\n")
        f.write(f"RMSE: {metrics['RMSE']:.6f}\n")
        f.write(f"R2: {metrics['R2']:.6f}\n")

    print("")
    print("[INFO] Готово.")
    print(f"[INFO] Результаты сохранены в: {OUTPUT_DIR}")
    print("[INFO] Основные файлы:")
    print(f"  - {OUTPUT_DIR / 'metrics.json'}")
    print(f"  - {OUTPUT_DIR / 'used_k1_features.csv'}")
    print(f"  - {OUTPUT_DIR / 'used_lab_features.csv'}")
    print(f"  - {OUTPUT_DIR / 'feature_importance.csv'}")
    print(f"  - {OUTPUT_DIR / 'summary.txt'}")


if __name__ == "__main__":
    main()