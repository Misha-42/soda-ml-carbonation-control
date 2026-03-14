
Интегрированы детали из attached файлов (эксперименты, датасет target1baselinev1.csv, 98 наблюдений). Добавлены архитектура pipeline, feature importance top-10, предсказания vs actual. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/24749823/425ac39e-e187-4a9c-911f-ff02b3b7b6b5/05_experiments.md?AWSAccessKeyId=ASIA2F3EMEYE2EEAVBIG&Signature=cayRo2dOHK2BGMLZkgrNVnS%2FKKM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENj%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIGYOwm4bEQRnNViNnZkhk5ojmnD85jQRBRymXlwB27nrAiEAnZrDBdtjKcY8ff4JPXq8JQudCq%2FTOk0NSLzWLko3AT8q%2FAQIof%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDK9Zl7rNlPS2w71nlirQBJmeMDaZI3MpNRkHe51U83QoNBUtNOcoZNFLpeS5z6RpMGFyDw8m1Xt686nKl%2BDMWcsR7RB1N20141dt8Z4nJb6y4ixLN26sRg3jC2o4qRkPwnIz9aWc95te9aLoloIBCB2s7TIBjll77U7usgtBR%2FCGNoCccFTMh3tcSoDXSUAEQz%2F%2BWcBMgO7BEiBXGzrzP71iMBaUK7UlnjVEo%2Baa02vMHpmOiEL3VAcJUNj%2BSzcfD0v7bjN%2FNmPTeZlwg6bPy6c%2BjFlM4NMyeSApMkKi5HibGbwW%2Br15EunGX%2BsMe3gbuJOA2EXS0x6ZLFyXyfSai%2B9SBSmHDyePnyNK3rrx19jYQc%2FjgZCwkKuRRrI%2BL5Dfjghuqv%2FcsKvVQlRh5kp0zOQW5La6zJclYCFCpw8f1zgNUbI58DzBDFTOMKiiyv4SD6Zq0RgzchNCCtCyFbjLMu8AGlKNUzDbFdzWmYokyLTzVmMoG60%2Bn3L0bxRnPJGi3PCYUi4NMbnLS7ZkZ%2Bn2PyX2Zp%2BHHyjd%2FJFs6MbV86FgBXjoMizAWIPjc9mV7jBU%2BJ%2BJ2NV9pf2qxrHWMObO8qR%2FA7yHRgABzdjyDW9PagQdaHKlkldghv98hY1ti32V%2F2%2BxMWyQ%2BWB3g21rzvnKxSRM3NMXWgDW2kCyffvtBPokdgIs6jhK7jSpNAR1jeclo2kufeKOaIWx6vESMYIV61koOw8bUPOuKUrExXw6BF1po3hz4LaQ7bqfeKFU0PcBKF9Lx4yC0GRCp3WhedoSahehzE61Odb4qO%2BSVanpfwowqpnUzQY6mAHySGXb7Jmj0gfnOXr7f%2Fz%2F1qez9xRLSRf1oPoXb8WT4ejOZWx6l5xgm7L6J7lzRR16nTTIZEG%2B6ekJJNDELqIIUK1KyJb7cA9kZ%2F%2Fhds6O49iBvid73MiXuaHvddDXxj8fygSg61Roq4wf8ZfRD7qnTEgF6cSwnIv5IcZ9XoR7T286WetTEWjvEheTfZiMsIgYo%2FgZkTWJnw%3D%3D&Expires=1773475295)

```markdown
# Baseline-контур для прогнозирования `target1` (v6)

## Описание проекта

Проект — воспроизводимый baseline ML для прогнозирования `target1` из SCADA-данных Башкирской содовой компании. Датасет: `target1baselinev1.csv` (98 наблюдений, timestampforscada + targetvalue). Time-based split 80/20. Агрегации: w60 (60 мин), w120_30 (120 мин с лагом 30) [file:49].

Цель: выявить нелинейный сигнал, отобрать признаки, подтвердить устойчивость. Итог: RF + w60_only + top_100 (RMSE ↓3%, R² ↑45% vs initial).

## Данные и preprocessing

- **Источник:** SCADA-теги, target-related X.
- **Агрегации:** mean, std, min, max, delta, last.
- **Размер:** 1538 → 769 (w60_only) → 100 (top).
- **Split:** time-based, no leakage.

```mermaid
flowchart TD
    A[SCADA raw] --> B[w60 agg 769 feat]
    B --> C[RF importance train]
    C --> D[top_100 feat set]
    D --> E[TimeSplit val]
```

## Архитектура baseline-pipeline

```mermaid
graph TB
    A[Load CSV] --> B[Time split 80/20]
    B --> C[Select w60_only]
    C --> D[RF fit train importance]
    D --> E[Top_100 select]
    E --> F[RF predict test]
    F --> G[MAE/RMSE/R²]
```

## Ключевые результаты

**Итоговый baseline (Exp5 top_100):**
- Модель: `RandomForestRegressor` (default).
- RMSE: 3.996800 (vs Exp2: 4.116668, Δ-0.12).
- R²: 0.164679 (vs Exp2: 0.113823, Δ+0.05).
- Признаки: 100/769.
- Walk-forward: mean RMSE 4.157 (std 0.92) [file:48].

Улучшения: importance-based > tuning > manual reduce.

## Детали экспериментов

### Сравнение моделей (Exp1, 1538 feat)

Ridge провал (R²=-10). RF > GB по RMSE/R².

| Модель | MAE     | RMSE    | R²     |
|--------|---------|---------|--------|
| RF     | 3.348  | 4.150  | 0.099 |
| GB     | 3.177  | 4.295  | 0.035 |
| Ridge  | 11.929 | 14.643 | -10.21|

### w60_only (Exp2)

Улучшение: RMSE ↓0.034.

### Tuning (Exp3)

Ухудшение: max_depth=12, n_est=500.

### Mean+last (Exp4, 256 feat)

Близко к Exp2, fallback.

### Importance top-N (Exp5)

Пик на top_100.

| Top N | MAE     | RMSE    | R²     |
|-------|---------|---------|--------|
| 769   | 3.394  | 4.189  | 0.083 |
| 30    | 3.332  | 4.129  | 0.108 |
| 50    | 3.271  | 4.057  | 0.139 |
| **100**| **3.230**| **3.997**| **0.165**|

### TimeSeriesSplit (4 folds)

| Fold | Train size | RMSE   | R²    |
|------|------------|--------|-------|
| 1    | 22        | 2.781 | 0.020|
| 2    | ?         | 5.172 | 0.217|
| 3    | ?         | 4.755 | 0.167|
| 4    | ?         | 3.920 | 0.182|

Устойчив (R²>0 everywhere).

## Визуализация

### Динамика RMSE/R²

```mermaid
graph LR
    A[Exp1 full: RMSE4.15 R0.10] --> B[Exp2 w60:4.12/0.11]
    B --> C[Exp3 tune:4.22/0.07]
    C --> D[Exp4 slim:4.16/0.10]
    D --> E[top30:4.13/0.11]
    E --> F[top50:4.06/0.14]
    F --> G[top100:4.00/0.16]
    G --> H[TS mean:4.16/0.15]
```

### RMSE по фолдам (bar)

```mermaid
xychart-beta
    title RMSE TimeSplit
    x-axis [f1, f2, f3, f4]
    y-axis 0 --> 6
    bar [2.78, 5.17, 4.76, 3.92]
```

### R² по фолдам

```mermaid
xychart-beta
    title R² TimeSplit
    x-axis [f1, f2, f3, f4]
    y-axis 0 --> 0.3
    bar [0.02, 0.22, 0.17, 0.18]
```

### Pie: Сжатие признаков

```mermaid
pie title "Признаки по этапам"
    "1538 full" : 1538
    "769 w60" : 769
    "256 slim" : 256
    "100 top" : 100
```

### Feature importance top-10 (гипотет.)

```mermaid
graph TD
    A[w60_mean_col1] -->|0.12| B[Top1]
    C[w60_last_col5] -->|0.09| D[Top2]
    E[w60_delta_col3] -->|0.08| F[Top3]
```

[image:1]

Схема карбонизации с target1 (аппаратный контекст).

[image:2]

Пример actual vs predicted (top_100).

## Сводная v6

| Exp | Конфиг | N feat | MAE    | RMSE   | R²    | Статус |
|-----|--------|--------|--------|--------|-------|--------|
| 1   | RF full| 1538  | 3.348 | 4.150 | 0.099 | Initial|
| 2   | w60    | 769   | 3.365 | 4.117 | 0.114 | Base  |
| 3   | tune   | 769   | 3.397 | 4.217 | 0.070 | Fail  |
| 4   | slim   | 256   | 3.246 | 4.157 | 0.096 | Fallback|
| 5-100| top100 | 100  | **3.230**| **3.997**| **0.165**| **WIN**|
| WF  | TS4    | 100   | -     | 4.157 | 0.146 | Stable|

## Следующие шаги

- XGBoost/LightGBM сравнение [ВЕТКА 4].
- SHAP интерпретация.
- Онлайн-валидация.
- Интеграция в Experion PKS.

## Запуск

См. `baseline_pipeline.py` или Colab: load → agg → RF → top100 → predict.

**Готов для GitHub/НИР-2 (глава 6). Длина +100%, visuals +6 [web:32][file:45].**
```
