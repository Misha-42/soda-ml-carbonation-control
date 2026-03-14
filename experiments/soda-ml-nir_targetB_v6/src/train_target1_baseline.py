from pathlib import Path
import json
import math

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# === PATHS ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "launch_target1_k1" / "dataset_target1_baseline_v1.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = REPORTS_DIR / "logs"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# === SETTINGS ===
TARGET_COL = "target_value"
TIME_COL = "target_timestamp_for_scada"


def rmse_score(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def save_feature_importance(pipeline, model_name, feature_names, out_path, top_n=20):
    """
    Сохраняет график важности признаков, если модель поддерживает feature_importances_.
    """
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return

    fi = pd.Series(model.feature_importances_, index=feature_names)
    fi = fi.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(9, 6))
    fi.sort_values().plot(kind="barh")
    plt.title(f"{model_name} feature importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_metric_plot(results_df, metric_col, out_path, title):
    plt.figure(figsize=(8, 4))
    plt.bar(results_df["model"], results_df[metric_col])
    plt.xticks(rotation=20)
    plt.ylabel(metric_col.upper())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# === LOAD ===
df = pd.read_csv(DATA_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"Не найдена колонка target: {TARGET_COL}")

if TIME_COL not in df.columns:
    raise ValueError(f"Не найдена колонка времени: {TIME_COL}")

# Используем только готовые агрегированные признаки
feature_cols = [
    c for c in df.columns
    if c.startswith("w60__")
]

if not feature_cols:
    raise ValueError("Не найдены feature-колонки вида w60__... или w120_30__...")

# Удаляем пустой target
df = df.dropna(subset=[TARGET_COL]).copy()

# Время
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
df = df.dropna(subset=[TIME_COL]).copy()

# Сортировка по времени
df = df.sort_values(TIME_COL).reset_index(drop=True)

X = df[feature_cols].copy()
y = df[TARGET_COL].copy()

# train/test без shuffle
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

print(f"Rows total: {len(df)}")
print(f"Train rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print(f"Features: {len(feature_cols)}")

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Train/Test split дал пустую выборку. Проверьте размер датасета.")


# === MODELS ===
models = {
    "ridge": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", Ridge(alpha=1.0))
    ]),
    "random_forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "gradient_boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3
        ))
    ]),
}

results = []
trained_models = {}


# === TRAIN / EVAL ===
for model_name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = rmse_score(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append({
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "n_features": len(feature_cols),
    })

    trained_models[model_name] = model

    pred_df = pd.DataFrame({
        "timestamp": df.iloc[split_idx:][TIME_COL].values,
        "y_true": y_test.values,
        "y_pred": preds,
    })
    pred_df.to_csv(REPORTS_DIR / f"target1_predictions_{model_name}.csv", index=False)


# === RESULTS TABLE ===
results_df = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)

results_df.to_csv(REPORTS_DIR / "target1_baseline_metrics.csv", index=False)
results_df.to_csv(REPORTS_DIR / "target1_experiments_summary.csv", index=False)
results_df.to_csv(REPORTS_DIR / "baseline_results_target1_v1.csv", index=False)


# === SAVE MODELS ===
if "ridge" in trained_models:
    joblib.dump(trained_models["ridge"], MODELS_DIR / "target1_ridge.joblib")

if "random_forest" in trained_models:
    joblib.dump(trained_models["random_forest"], MODELS_DIR / "target1_random_forest.joblib")

if "gradient_boosting" in trained_models:
    joblib.dump(trained_models["gradient_boosting"], MODELS_DIR / "target1_gradient_boosting.joblib")


# === REPORT MARKDOWN ===
best_row = results_df.iloc[0]

report_text = f"""# Target1 baseline report

## Dataset
- Data path: `{DATA_PATH}`
- Rows total: {len(df)}
- Train rows: {len(X_train)}
- Test rows: {len(X_test)}
- Features: {len(feature_cols)}
- Target: `{TARGET_COL}`
- Time column: `{TIME_COL}`

## Best model
- Model: {best_row['model']}
- MAE: {best_row['mae']:.6f}
- RMSE: {best_row['rmse']:.6f}
- R2: {best_row['r2']:.6f}

## Models compared
- ridge
- random_forest
- gradient_boosting

## Notes
- Использованы только признаки с префиксами `w60__` и `w120_30__`
- Использован time-based split без shuffle
- Ridge здесь выступает как линейный baseline
- RandomForest и GradientBoosting — основные нелинейные baseline-модели

## Full results

{results_df.to_markdown(index=False)}
"""

with open(REPORTS_DIR / "target1_baseline_report.md", "w", encoding="utf-8") as f:
    f.write(report_text)


# === RUN INFO JSON ===
run_info = {
    "data_path": str(DATA_PATH),
    "target_col": TARGET_COL,
    "time_col": TIME_COL,
    "n_rows_total": int(len(df)),
    "n_rows_train": int(len(X_train)),
    "n_rows_test": int(len(X_test)),
    "n_features": int(len(feature_cols)),
    "feature_cols_sample": feature_cols[:20],
    "best_model": str(best_row["model"]),
    "best_mae": float(best_row["mae"]),
    "best_rmse": float(best_row["rmse"]),
    "best_r2": float(best_row["r2"]),
}

with open(LOGS_DIR / "target1_run_info.json", "w", encoding="utf-8") as f:
    json.dump(run_info, f, ensure_ascii=False, indent=2)


# === PLOTS ===
save_metric_plot(
    results_df,
    metric_col="mae",
    out_path=REPORTS_DIR / "target1_mae_comparison.png",
    title="Target1 baseline: MAE comparison"
)

save_metric_plot(
    results_df,
    metric_col="rmse",
    out_path=REPORTS_DIR / "target1_rmse_comparison.png",
    title="Target1 baseline: RMSE comparison"
)

if "random_forest" in trained_models:
    save_feature_importance(
        trained_models["random_forest"],
        model_name="Random Forest",
        feature_names=feature_cols,
        out_path=REPORTS_DIR / "target1_rf_feature_importance.png"
    )

if "gradient_boosting" in trained_models:
    save_feature_importance(
        trained_models["gradient_boosting"],
        model_name="Gradient Boosting",
        feature_names=feature_cols,
        out_path=REPORTS_DIR / "target1_gb_feature_importance.png"
    )


# === CONSOLE OUTPUT ===
print("\n=== BASELINE RESULTS ===")
print(results_df)

print("\nSaved files:")
print(f"- {REPORTS_DIR / 'target1_baseline_metrics.csv'}")
print(f"- {REPORTS_DIR / 'target1_experiments_summary.csv'}")
print(f"- {REPORTS_DIR / 'baseline_results_target1_v1.csv'}")
print(f"- {REPORTS_DIR / 'target1_baseline_report.md'}")
print(f"- {REPORTS_DIR / 'target1_mae_comparison.png'}")
print(f"- {REPORTS_DIR / 'target1_rmse_comparison.png'}")
print(f"- {MODELS_DIR / 'target1_ridge.joblib'}")
print(f"- {MODELS_DIR / 'target1_random_forest.joblib'}")
print(f"- {MODELS_DIR / 'target1_gradient_boosting.joblib'}")
print(f"- {LOGS_DIR / 'target1_run_info.json'}")