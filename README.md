# ML-система адаптивного управления карбонизацией БСК

Проект посвящён разработке baseline-системы машинного обучения для прогнозирования сменных показателей процесса карбонизации бикарбонатной суспензии кальцинированной соды (БСК) на основе производственных и лабораторных данных.

## Задача проекта

Цель работы — построить воспроизводимый baseline для soft sensor / decision support контура, который позволяет прогнозировать ключевые сменные показатели карбонизации по данным технологического процесса.

На текущем этапе решается задача:
- подготовки сменного ML-датасета;
- отбора технологически осмысленных признаков;
- построения и сравнения baseline-моделей;
- фиксации контрольной точки для дальнейшего развития НИР.

## Данные

Источник данных:
- 17 Excel-файлов лабораторного контроля;
- период наблюдений: **20.02.2026 – 08.03.2026**;
- исходные листы:
  - `Лист контроля`
  - `Качеств.показатели`

Принятая единица наблюдения:
- **1 строка = 1 смена**

Размер текущего сменного датасета:
- **34 смены**
- `17 дней × 2 смены`

Сформированные датасеты:
- `carbonation_full_dataset.csv`
- `carbonation_full_dataset_ml_ready.csv`
- `carbonation_full_dataset_for_ml.csv`
- `shift_dataset_export.csv`

## Таргеты

В проекте рассматриваются три сменных таргета:

- `target_t` — температура суспензии гидрокарбоната натрия
- `target_cl` — содержание хлорид-ионов
- `target_nh3` — содержание свободного аммиака

## Лучшие модели на текущем этапе

По итогам baseline-этапа зафиксированы следующие лучшие модели:

- `target_t` → **rf_shift_whitelist**
- `target_cl` → **ridge**
- `target_nh3` → **rf_shift_whitelist**

Это означает, что оптимальная baseline-модель зависит от таргета:
- для `target_t` лучше работает нелинейный `Random Forest`;
- для `target_cl` лучшей оказалась регуляризованная линейная модель `Ridge`;
- для `target_nh3` текущий baseline пока остаётся слабым и требует отдельного улучшения.

## Ключевые метрики

### `target_t`
- модель: `rf_shift_whitelist`
- MAE = **0.387203**
- RMSE = **0.475321**
- R² = **0.036030**

### `target_cl`
- модель: `ridge`
- MAE = **0.645223**
- RMSE = **0.773226**
- R² = **0.087208**

### `target_nh3`
- модель: `rf_shift_whitelist`
- MAE = **1.334134**
- RMSE = **1.535068**
- R² = **-0.009601**

## Основные результаты

На текущем этапе установлено:

- baseline-пайплайн построен и зафиксирован;
- ручной white-list признаков оказался полезным;
- лучший таргет по качеству baseline-прогноза — `target_t`;
- `target_cl` показал улучшение после перехода от `Random Forest` к `Ridge`;
- `target_nh3` остаётся наиболее сложным таргетом;
- результаты зафиксированы как текстово, так и в machine-readable формате (`csv`, `json`, `md`).

## Структура репозитория

```text
.
├── nir/
│   ├── 05_experiments.md
│   ├── 06_results.md
│   ├── 07_conclusion.md
│   └── 08_limitations_and_next_steps.md
│
├── reports/
│   ├── baseline_report.md
│   ├── baseline_summary_all_targets.csv
│   ├── baseline_metrics.csv
│   ├── model_compare_t_cl.csv
│   ├── model_compare_t_cl_predictions.csv
│   ├── best_model_t_cl.csv
│   ├── best_model_t_cl.json
│   ├── target_t_baseline_summary.json
│   ├── target_cl_baseline_summary.json
│   └── target_nh3_baseline_summary.json
│
├── plots/
│   ├── *.png
│   ├── README_plots.md
│   └── FIGURE_CAPTIONS.md
│
├── plots_ru/
│   ├── *_ru.png
│   ├── README_PLOTS_RU.md
│   ├── FIGURE_CAPTIONS_RU.md
│   ├── FIGURES_FOR_NIR_ORDER_RU.md
│   └── READY_FOR_NIR_CHECKLIST.md
│
├── PROJECT_PROGRESS.md
├── DECISION.txt
├── README.md
└── .gitignore
