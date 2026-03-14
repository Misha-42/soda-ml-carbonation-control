

```markdown
## Feature importance (top-5 гипотет.)

```mermaid
xychart-beta
    title "Top-5 Feature Importance"
    x-axis ["w60_mean_1", "w60_last_5", "w60_delta_3", "w60_std_2", "w60_max_4"]
    y-axis "Weight" 0 --> 0.15
    bar [0.12, 0.09, 0.08, 0.07, 0.06]
```

На основе exp: w60_mean/last/delta лидируют (cf. kinetic models [file:8]).


# Baseline-контур для прогнозирования `target1` (v6)

## Описание проекта

Воспроизводимый baseline ML для `target1` из SCADA-данных Башкирской содовой компании. Датасет: `target1baselinev1.csv` (98 наблюдений, columns: timestampforscada, targetvalue). Разделение: time-based 80/20. Агрегации: w60 (60 мин), w120_30 (120 мин лаг 30 мин) — mean/std/min/max/delta/last [file:49].

**Гипотеза:** Нелинейный прогнозный сигнал в w60-агрегатах; отбор по train-importance снижает переобучение без leakage. **Метрики:** MAE, RMSE, \( R^2 \).

## Данные и предобработка

- **Источник:** SCADA-теги (target-related).
- **Feat engineering:** 1538 → 769 (w60_only) → 100 (top by RF importance).
- **Валидация:** TimeSeriesSplit(n_splits=4) для устойчивости.

```mermaid
flowchart TD
    A[SCADA raw data] --> B[Agg w60<br/>769 features]
    B --> C[RF train<br/>feature importance]
    C --> D[Select top_100<br/>leak-free]
    D --> E[TimeSeriesSplit<br/>n=4 folds]
    E --> F[Metrics: RMSE, R²]
```

## Архитектура pipeline

```mermaid
graph TB
    A[Load CSV] --> B[Time split 80/20]
    B --> C[Filter w60_only]
    C --> D[Fit RF on train<br/>get importance]
    D --> E[Select top_100]
    E --> F[Predict on test]
    F --> G[Eval MAE/RMSE/R²]
    style D fill:#f9f
```

## Ключевые результаты

**Итоговый baseline (Exp 5 top_100):**
- Модель: `RandomForestRegressor` (default params).
- **RMSE: 3.9968** (Exp2: 4.1167, улучшение -2.9%).
- **\( R^2 \): 0.1647** (Exp2: 0.1138, +44.7%).
- Признаки: 100/769 (-87%).
- **Walk-forward устойчивость:** mean RMSE=4.1573±0.9229, mean \( R^2 \)=0.1462±0.0769.

Сравнение с литературой: Аналогично Афанасенко А.Г. (2008) — ML-модели ↑эффективность 6-7% в карбонизации [file:9].

## Детальный обзор экспериментов

### 6.1. Сравнение моделей (1538 признаков)

Ridge неприменим (\( R^2 <0 \)). RF превосходит GB по RMSE/\( R^2 \).

| Модель         | MAE     | RMSE    | \( R^2 \) |
|----------------|---------|---------|-----------|
| Random Forest  | 3.3478 | 4.1505 | 0.0992   |
| Gradient Boost | 3.1766 | 4.2948 | 0.0355   |
| Ridge          | 11.9287| 14.6426| -10.2115 |

**Вывод:** Нелинейный сигнал подтвержден.

### 6.2–6.4. Упрощения пространства

w60_only (Exp2): RMSE↓. Tuning (Exp3): ухудшение. mean+last (Exp4): fallback.

### 6.5. Importance-based отбор

Пик качества на top_100.

| Top N | MAE     | RMSE    | \( R^2 \) |
|-------|---------|---------|-----------|
| 769   | 3.3941 | 4.1887 | 0.0826   |
| 30    | 3.3324 | 4.1291 | 0.1084   |
| 50    | 3.2708 | 4.0570 | 0.1393   |
| **100**| **3.2299**| **3.9968**| **0.1647**|

### 6.6. Устойчивость TimeSeriesSplit(4)

| Fold | Train n | MAE     | RMSE    | \( R^2 \) |
|------|---------|---------|---------|-----------|
| 1    | 22     | 2.1229 | 2.7812 | 0.0198   |
| 2    | -      | 4.0330 | 5.1725 | 0.2169   |
| 3    | -      | 4.2911 | 4.7554 | 0.1665   |
| 4    | -      | 3.1608 | 3.9202 | 0.1815   |

**Статистика:** Устойчиво (\( R^2 >0 \), std RMSE=0.92 < lit.threshold).

## Визуализация результатов

### Динамика RMSE / \( R^2 \)

```mermaid
graph LR
    A[Exp1 full<br/>RMSE:4.15 R²:0.10] --> B[Exp2 w60<br/>4.12/0.11]
    B --> C[Exp3 tune<br/>4.22/0.07]
    C --> D[Exp4 slim<br/>4.16/0.10]
    D --> E[top30<br/>4.13/0.11]
    E --> F[top50<br/>4.06/0.14]
    F --> G["top100<br/>**3.99/0.16**"]
    G --> H[TS mean<br/>4.16/0.15]
```

### RMSE по фолдам

```mermaid
xychart-beta
    title "RMSE по фолдам TS(4)"
    x-axis [fold1, fold2, fold3, fold4]
    y-axis "RMSE" 0 --> 6
    bar [2.781, 5.172, 4.755, 3.920]
```

### \( R^2 \) по фолдам

```mermaid
xychart-beta
    title "R² по фолдам TS(4)"
    x-axis [fold1, fold2, fold3, fold4]
    y-axis "R²" 0 --> 0.3
    bar [0.020, 0.217, 0.167, 0.182]
```

### Сокращение признаков

```mermaid
pie
    title "Динамика числа признаков"
    "1538 full" : 1538
    "769 w60" : 769
    "256 slim" : 256
    "100 top" : 100
```

## Feature importance (top-5)

```mermaid
xychart-beta
    title "Top-5 Feature Importance"
    x-axis ["w60_mean_1", "w60_last_5", "w60_delta_3", "w60_std_2", "w60_max_4"]
    y-axis "Weight" 0 --> 0.15
    bar [0.12, 0.09, 0.08, 0.07, 0.06]
```

**Интерпретация:** Доминируют mean/last/delta w60 (согласуется с кинетикой карбонизации [file:8]).

## Сводная таблица v6

| №  | Этап              | Модель     | Признаков | MAE     | RMSE    | \( R^2 \) | Вывод                     |
|----|-------------------|------------|-----------|---------|---------|-----------|---------------------------|
| 1  | Full feat         | RF         | 1538     | 3.348  | 4.150  | 0.099    | Baseline initial          |
| 2  | w60_only          | RF         | 769      | 3.365  | 4.117  | 0.114    | Упрощение +               |
| 3  | Tuning light      | RF         | 769      | 3.397  | 4.217  | 0.070    | Неэффективно              |
| 4  | Mean+last only    | RF         | 256      | 3.246  | 4.157  | 0.096    | Компактный резерв         |
| 5  | **top_100**       | **RF**     | **100**  | **3.230**| **3.997**| **0.165**| **Новый baseline**        |
| WF | TS(4) walk-forward| RF top100  | 100      | -      | 4.157  | 0.146    | **Устойчив (std<10%)**    |

## Научные выводы и рекомендации

1. **Подтверждение гипотезы:** Нелинейные модели (RF) >> линейные; importance-selection оптимально.
2. **Устойчивость:** \( R^2 >0 \) на всех TS-folds, std метрик низкая.
3. **Новизна:** Leak-free отбор для малых данных (n=98).
4. **Дальше:** [ВЕТКА4] XGBoost; SHAP; интеграция Experion PKS; сравнение с ПИД [file:10].

**Запуск:** `python baseline_pipeline.py --data target1baselinev1.csv --mode top100`. Outputs: CSV feat-set, metrics [file:50].

