# soda-ml-nir

Минимальный baseline ML-проект для НИР по анализу технологического процесса производства соды.

На этапе 1 цель — быстро и прозрачно сравнить несколько простых вариантов:
- `RandomForestRegressor`
- `XGBRegressor`

## Структура проекта

- `data/` — входные CSV-данные для обучения (ожидаются локально).
- `src/data_prep.py` — загрузка CSV, базовая очистка, проверка обязательных столбцов, time-based split.
- `src/features.py` — минимальная подготовка признаков (числовые колонки + обработка пропусков).
- `src/train_baseline.py` — запуск baseline-экспериментов для RandomForest и XGBoost.
- `src/evaluate.py` — расчёт метрик и сохранение результатов/отчёта.
- `models/` — сохранённые модели по экспериментам.
- `reports/` — CSV с метриками, сводка экспериментов и краткий Markdown-отчёт.

## Быстрый запуск baseline

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. Положите CSV в папку `data/`.

CSV должен содержать:
- целевую колонку (например, `target`)
- временную колонку (например, `timestamp`) — если есть, будет использован time-based split

3. Запустите эксперименты:

```bash
python src/train_baseline.py --data-path data/your_data.csv --target target --time-column timestamp
```

Если временной колонки нет, можно не передавать `--time-column`; тогда будет использован split по порядку строк.

После запуска появятся:
- `models/rf_small.joblib`
- `models/rf_medium.joblib`
- `models/xgb_small.joblib`
- `models/xgb_medium.joblib`
- `reports/baseline_metrics.csv`
- `reports/experiments_summary.csv`
- `reports/baseline_report.md`
- `reports/mae_comparison.png`
- `reports/rmse_comparison.png`
- `reports/rf_feature_importance.png`
- `reports/xgb_feature_importance.png`

## Что лежит в experiments_summary.csv

Для каждого эксперимента сохраняются:
- название модели,
- параметры,
- метрики (MAE, RMSE, R2),
- дата запуска.


## Простые графики

Pipeline автоматически строит графики и сохраняет их в `reports/`:
- сравнение MAE (`mae_comparison.png`),
- сравнение RMSE (`rmse_comparison.png`),
- важности признаков для RandomForest (`rf_feature_importance.png`),
- важности признаков для XGBoost (`xgb_feature_importance.png`).
