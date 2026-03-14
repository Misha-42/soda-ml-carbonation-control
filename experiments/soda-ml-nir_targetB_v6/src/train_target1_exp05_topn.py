from pathlib import Path
import json

import joblib
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

DATA_PATH = PROJECT_ROOT / "launch_target1_k1" / "dataset_target1_baseline_v1.csv"
TARGET_COL = "target_value"
TIME_COL = "target_timestamp_for_scada"

EXPERIMENT_NAME = "target1_exp05_topn_importance"
BASE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp_05_target1_topn_importance"
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_LIST = [30, 50, 100]

BASELINE_EXP02_RF = {
    "RMSE": 4.116668,
    "R2": 0.113823,
}

# Должны совпадать с exp_02 для честного сравнения
RF_PARAMS = {
    "n_estimators": 300,
    "random_state": 42,
    "n_jobs": -1,
}


# =========================
# HELPERS
# =========================
def calc_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def is_feature_allowed(col: str) -> bool:
    if col in {TARGET_COL, TIME_COL}:
        return False

    col_l = col.lower()

    banned_substrings = [
        "target",
    ]
    if any(x in col_l for x in banned_substrings):
        return False

    return col.startswith("w60__")


def load_and_prepare_split():
    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Не найдена целевая колонка: {TARGET_COL}")
    if TIME_COL not in df.columns:
        raise ValueError(f"Не найдена временная колонка: {TIME_COL}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, TARGET_COL]).copy()
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    raw_feature_cols = [c for c in df.columns if is_feature_allowed(c)]
    if not raw_feature_cols:
        raise ValueError("Не найдены признаки w60__")

    X = df[raw_feature_cols].copy()
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    y = df[TARGET_COL].copy()

    split_idx = int(len(df) * 0.8)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError(f"Некорректный индекс split: {split_idx} для размера датасета {len(df)}")

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    # ВАЖНО: медианы считаем только на train
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians).fillna(0.0)
    X_test = X_test.fillna(train_medians).fillna(0.0)

    final_feature_cols = list(X_train.columns)

    return df, final_feature_cols, X_train, X_test, y_train, y_test, split_idx


def fit_base_rf_and_get_importance(X_train, y_train):
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    importance_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return model, importance_df


def save_base_artifacts(
    df,
    split_idx,
    feature_cols,
    base_model,
    importance_df,
    base_metrics,
    base_preds,
):
    pd.DataFrame({"feature": feature_cols}).to_csv(
        BASE_OUTPUT_DIR / "all_w60_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    importance_df.to_csv(
        BASE_OUTPUT_DIR / "feature_importance_full_w60.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            TIME_COL: df.iloc[split_idx:][TIME_COL].values,
            "y_true": df.iloc[split_idx:][TARGET_COL].values,
            "y_pred_rf_full_w60": base_preds,
        }
    ).to_csv(
        BASE_OUTPUT_DIR / "base_full_w60_rf_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    joblib.dump(base_model, BASE_OUTPUT_DIR / "base_full_w60_random_forest.joblib")

    save_json(
        {
            "experiment_name": EXPERIMENT_NAME,
            "variant": "base_full_w60",
            "dataset_path": str(DATA_PATH),
            "target_col": TARGET_COL,
            "time_col": TIME_COL,
            "n_features_selected": len(feature_cols),
            "rf_params": RF_PARAMS,
            "metrics": base_metrics,
            "comparison_vs_exp02_rf": {
                "delta_RMSE": float(base_metrics["RMSE"] - BASELINE_EXP02_RF["RMSE"]),
                "delta_R2": float(base_metrics["R2"] - BASELINE_EXP02_RF["R2"]),
            },
        },
        BASE_OUTPUT_DIR / "base_full_w60_metrics.json",
    )


def run_topn_experiment(top_n, df, split_idx, X_train, X_test, y_train, y_test, importance_df):
    output_dir = BASE_OUTPUT_DIR / f"top_{top_n}"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_features = importance_df["feature"].head(top_n).tolist()

    X_train_top = X_train[selected_features].copy()
    X_test_top = X_test[selected_features].copy()

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train_top, y_train)
    preds = model.predict(X_test_top)

    metrics = calc_metrics(y_test, preds)

    results = {
        "experiment_name": EXPERIMENT_NAME,
        "variant": f"top_{top_n}",
        "dataset_path": str(DATA_PATH),
        "target_col": TARGET_COL,
        "time_col": TIME_COL,
        "n_features_selected": len(selected_features),
        "rf_params": RF_PARAMS,
        "metrics": metrics,
        "selected_features": selected_features,
        "comparison_vs_exp02_rf": {
            "delta_RMSE": float(metrics["RMSE"] - BASELINE_EXP02_RF["RMSE"]),
            "delta_R2": float(metrics["R2"] - BASELINE_EXP02_RF["R2"]),
        },
    }

    pd.DataFrame({"feature": selected_features}).to_csv(
        output_dir / "selected_features.csv",
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
        output_dir / "rf_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    joblib.dump(model, output_dir / "random_forest.joblib")
    save_json(results, output_dir / "metrics.json")

    return results


def build_comparison_table(base_metrics, variant_results):
    rows = []

    rows.append(
        {
            "variant": "base_full_w60",
            "n_features": None,
            "MAE": base_metrics["MAE"],
            "RMSE": base_metrics["RMSE"],
            "R2": base_metrics["R2"],
            "delta_RMSE_vs_exp02": base_metrics["RMSE"] - BASELINE_EXP02_RF["RMSE"],
            "delta_R2_vs_exp02": base_metrics["R2"] - BASELINE_EXP02_RF["R2"],
        }
    )

    for result in variant_results:
        rows.append(
            {
                "variant": result["variant"],
                "n_features": result["n_features_selected"],
                "MAE": result["metrics"]["MAE"],
                "RMSE": result["metrics"]["RMSE"],
                "R2": result["metrics"]["R2"],
                "delta_RMSE_vs_exp02": result["comparison_vs_exp02_rf"]["delta_RMSE"],
                "delta_R2_vs_exp02": result["comparison_vs_exp02_rf"]["delta_R2"],
            }
        )

    return pd.DataFrame(rows)


def plot_metric_comparison(df_comparison: pd.DataFrame, metric_name: str, output_path: Path) -> None:
    metric_name_lower = metric_name.lower()

    if metric_name_lower == "mae":
        metric_col = "MAE"
        title = "Сравнение вариантов по MAE"
        ylabel = "MAE"
    elif metric_name_lower == "rmse":
        metric_col = "RMSE"
        title = "Сравнение вариантов по RMSE"
        ylabel = "RMSE"
    elif metric_name_lower == "r2":
        metric_col = "R2"
        title = "Сравнение вариантов по R²"
        ylabel = "R²"
    else:
        raise ValueError(f"Неизвестная метрика: {metric_name}")

    labels = df_comparison["variant"]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, df_comparison[metric_col])
    plt.title(title)
    plt.xlabel("Вариант")
    plt.ylabel(ylabel)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_topn_feature_importance(importance_df: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    top_df = importance_df.head(top_n).copy().sort_values("importance", ascending=True)

    plt.figure(figsize=(11, 8))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.title(f"Важность признаков Random Forest для полного w60 (top {top_n})")
    plt.xlabel("Важность")
    plt.ylabel("Признак")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_report_md(df, feature_cols, split_idx, base_metrics, variant_results) -> str:
    lines = []
    lines.append(f"# Отчёт: {EXPERIMENT_NAME}")
    lines.append("")
    lines.append("## Датасет")
    lines.append(f"- Путь: `{DATA_PATH}`")
    lines.append(f"- Число строк: {len(df)}")
    lines.append(f"- Полное число признаков `w60__`: {len(feature_cols)}")
    lines.append(f"- Целевая колонка: `{TARGET_COL}`")
    lines.append(f"- Временная колонка: `{TIME_COL}`")
    lines.append("")
    lines.append("## Разбиение")
    lines.append(f"- Размер train: {split_idx}")
    lines.append(f"- Размер test: {len(df) - split_idx}")
    lines.append("- Метод: time_based_80_20_no_shuffle")
    lines.append("")
    lines.append("## Базовый вариант: полный w60")
    lines.append(f"- MAE: {base_metrics['MAE']:.6f}")
    lines.append(f"- RMSE: {base_metrics['RMSE']:.6f}")
    lines.append(f"- R2: {base_metrics['R2']:.6f}")
    lines.append(f"- delta RMSE vs exp_02: {base_metrics['RMSE'] - BASELINE_EXP02_RF['RMSE']:.6f}")
    lines.append(f"- delta R2 vs exp_02: {base_metrics['R2'] - BASELINE_EXP02_RF['R2']:.6f}")
    lines.append("")

    lines.append("## Варианты top-N")
    lines.append("Чем меньше MAE/RMSE и выше R2, тем лучше.")
    lines.append("")

    best_variant_name = None
    best_variant_rmse = None

    for result in variant_results:
        lines.append(f"### {result['variant']}")
        lines.append(f"- Число признаков: {result['n_features_selected']}")
        lines.append(f"- MAE: {result['metrics']['MAE']:.6f}")
        lines.append(f"- RMSE: {result['metrics']['RMSE']:.6f}")
        lines.append(f"- R2: {result['metrics']['R2']:.6f}")
        lines.append(f"- delta RMSE vs exp_02: {result['comparison_vs_exp02_rf']['delta_RMSE']:.6f}")
        lines.append(f"- delta R2 vs exp_02: {result['comparison_vs_exp02_rf']['delta_R2']:.6f}")
        lines.append("")

        if best_variant_rmse is None or result["metrics"]["RMSE"] < best_variant_rmse:
            best_variant_rmse = result["metrics"]["RMSE"]
            best_variant_name = result["variant"]

    lines.append("## Вывод")
    if best_variant_name is not None:
        lines.append(f"- Лучший вариант внутри exp_05 по RMSE: {best_variant_name} (RMSE={best_variant_rmse:.6f})")
    else:
        lines.append("- Варианты top-N не были обработаны.")

    lines.append("- Сравнение с exp_02 нужно проводить прежде всего по RMSE и R2.")
    lines.append("- Если один из вариантов top-N близок к exp_02 или лучше него, его можно считать кандидатом на новый компактный baseline.")

    return "\n".join(lines)


def save_report_md(text: str, output_dir: Path):
    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(text)


# =========================
# MAIN
# =========================
def main():
    print(f"[INFO] Загрузка датасета: {DATA_PATH}")

    df, feature_cols, X_train, X_test, y_train, y_test, split_idx = load_and_prepare_split()

    print(f"[INFO] Число строк: {len(df)}")
    print(f"[INFO] Число признаков w60__: {len(feature_cols)}")
    print(f"[INFO] Размер train: {len(X_train)} | Размер test: {len(X_test)}")

    print("[INFO] Обучение базового Random Forest на полном наборе w60__ для feature importance...")
    base_model, importance_df = fit_base_rf_and_get_importance(X_train, y_train)

    base_preds = base_model.predict(X_test)
    base_metrics = calc_metrics(y_test, base_preds)

    print(f"[BASE FULL W60] {base_metrics}")

    save_base_artifacts(
        df=df,
        split_idx=split_idx,
        feature_cols=feature_cols,
        base_model=base_model,
        importance_df=importance_df,
        base_metrics=base_metrics,
        base_preds=base_preds,
    )

    variant_results = []

    for top_n in TOP_N_LIST:
        print(f"[INFO] Запуск варианта top_{top_n} ...")
        result = run_topn_experiment(
            top_n=top_n,
            df=df,
            split_idx=split_idx,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            importance_df=importance_df,
        )
        variant_results.append(result)
        print(f"[RESULT top_{top_n}] {result['metrics']}")

    df_comparison = build_comparison_table(base_metrics, variant_results)
    df_comparison.to_csv(
        BASE_OUTPUT_DIR / "comparison_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        {
            "experiment_name": EXPERIMENT_NAME,
            "rows": int(len(df)),
            "full_w60_feature_count": int(len(feature_cols)),
            "split": {
                "method": "time_based_80_20_no_shuffle",
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
            },
            "rf_params": RF_PARAMS,
            "baseline_exp02_rf": BASELINE_EXP02_RF,
            "base_full_w60_rf_metrics": base_metrics,
            "variants": {
                result["variant"]: result["metrics"] for result in variant_results
            },
        },
        BASE_OUTPUT_DIR / "summary_all_results.json",
    )

    plot_metric_comparison(
        df_comparison=df_comparison,
        metric_name="mae",
        output_path=BASE_OUTPUT_DIR / "mae_comparison.png",
    )
    plot_metric_comparison(
        df_comparison=df_comparison,
        metric_name="rmse",
        output_path=BASE_OUTPUT_DIR / "rmse_comparison.png",
    )
    plot_metric_comparison(
        df_comparison=df_comparison,
        metric_name="r2",
        output_path=BASE_OUTPUT_DIR / "r2_comparison.png",
    )
    plot_topn_feature_importance(
        importance_df=importance_df,
        output_path=BASE_OUTPUT_DIR / "rf_feature_importance_full_w60.png",
        top_n=20,
    )

    report_text = build_report_md(
        df=df,
        feature_cols=feature_cols,
        split_idx=split_idx,
        base_metrics=base_metrics,
        variant_results=variant_results,
    )
    save_report_md(report_text, BASE_OUTPUT_DIR)

    with open(BASE_OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Эксперимент: {EXPERIMENT_NAME}\n")
        f.write(f"Датасет: {DATA_PATH}\n")
        f.write(f"Число строк: {len(df)}\n")
        f.write(f"Полное число признаков w60__: {len(feature_cols)}\n")
        f.write(f"Размер train: {len(X_train)}\n")
        f.write(f"Размер test: {len(X_test)}\n\n")

        f.write("Базовый вариант: полный w60\n")
        for k, v in base_metrics.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write(f"  delta RMSE vs exp_02: {base_metrics['RMSE'] - BASELINE_EXP02_RF['RMSE']:.6f}\n")
        f.write(f"  delta R2 vs exp_02: {base_metrics['R2'] - BASELINE_EXP02_RF['R2']:.6f}\n")
        f.write("\n")

        for result in variant_results:
            f.write(f"{result['variant']}\n")
            f.write(f"  n_features: {result['n_features_selected']}\n")
            for k, v in result["metrics"].items():
                f.write(f"  {k}: {v:.6f}\n")
            f.write(f"  delta RMSE vs exp_02: {result['comparison_vs_exp02_rf']['delta_RMSE']:.6f}\n")
            f.write(f"  delta R2 vs exp_02: {result['comparison_vs_exp02_rf']['delta_R2']:.6f}\n")
            f.write("\n")

    print(f"[INFO] Готово. Результаты сохранены в: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()