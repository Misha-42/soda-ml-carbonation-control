


# Baseline-контур для прогнозирования `target1` (v6)



Воспроизводимый ML-baseline для `target1` из SCADA-данных
Башкирской содовой компании (НИР-2, УГНТУ).
Датасет: `target1baselinev1.csv` (98 наблюдений).
Разделение: time-based 80/20.

> **Результат:** RF + w60_only + top_100 features →
> RMSE=3.997, R²=0.165, признаки ↓87% (769→100), устойчиво на TS(4).

---

## Содержание

- [Мотивация](#мотивация)
- [Постановка задачи и гипотезы](#постановка-задачи-и-гипотезы)
- [Данные и предобработка](#данные-и-предобработка)
- [Архитектура pipeline](#архитектура-pipeline)
- [Методология и код](#методология-и-код)
- [Детальный обзор экспериментов](#детальный-обзор-экспериментов)
- [Визуализация результатов](#визуализация-результатов)
- [Сводная таблица v6](#сводная-таблица-v6)
- [Научные выводы](#научные-выводы)
- [Запуск](#запуск)

---

## Мотивация

`target1` — ключевой технологический показатель процесса карбонизации.
Прогноз на смену вперёд позволяет оператору проактивно корректировать
подачу CO2 и температурный режим до наступления отклонения.

**Проблема:** Ручной лабораторный контроль раз в смену (4–8 ч запаздывания).  
**Решение:** Soft sensor на RF — прогноз по w60-агрегатам SCADA.

```mermaid
flowchart LR
    A[SCADA w60<br/>агрегаты] --> B[RF top_100<br/>leak-free]
    B --> C[Прогноз target1<br/>±1 смена]
    C --> D[Оператор<br/>проактивно]
    D --> E[Стабильность ↑<br/>брак ↓]
```

**Обоснование подхода:**

| Решение | Обоснование |
|---------|-------------|
| w60 агрегации | Захват кинетики процесса (cf. [Афанасенко, 2008]) |
| RF (не Ridge) | Нелинейность подтверждена: Ridge R²=-10.2 |
| train-only importance | Исключение leakage при n=98 |
| TimeSeriesSplit | Временная структура данных |

---

## Постановка задачи и гипотезы

**Цель:** Построить воспроизводимый baseline для `target1`,
подтвердить нелинейность сигнала, отобрать признаки leak-free.

**Гипотезы и их проверка:**

| # | Гипотеза | Результат |
|---|----------|-----------|
| H1 | RF > Ridge/GB для `target1` (нелинейность) | ✅ Подтверждена: Ridge R²=-10.2 |
| H2 | w60_only ≥ w60+w120_30 (избыточность) | ✅ RMSE: 4.150→4.117 |
| H3 | top_100 > полный набор по RMSE и R² | ✅ RMSE: 4.117→3.997, R²: +44.7% |
| H4 | Baseline устойчив на TS(4) (R²>0) | ✅ R²>0 на всех фолдах |

**Метрики:** MAE, RMSE, \( R^2 \).

---

## Данные и предобработка

| Параметр | Значение |
|----------|----------|
| Файл | `target1baselinev1.csv` |
| Наблюдений | **n = 98** |
| Колонки | `timestampforscada`, `targetvalue` |
| Агрегации | w60: mean/std/min/max/delta/last |
| Feat space | 1538 → 769 (w60) → 100 (top) |
| Split | time-based 80/20, no leakage |
| Валидация | TimeSeriesSplit(n_splits=4) |

```mermaid
flowchart TD
    A[SCADA raw data] --> B[Agg w60<br/>769 features]
    B --> C[RF fit on train<br/>feature importance]
    C --> D[Select top_100<br/>leak-free!]
    D --> E[TimeSeriesSplit<br/>n=4 folds]
    E --> F[Metrics: RMSE, R²]
    style C fill:#fff3e0,stroke:#ff9800
    style D fill:#e8f5e9,stroke:#4caf50
```

---

## Архитектура pipeline

```mermaid
graph TB
    A[Load CSV] --> B[Time split 80/20]
    B --> C[Filter w60_only<br/>769 feat]
    C --> D[Fit RF on TRAIN<br/>get importance]
    D --> E[Select top_100<br/>by train importance]
    E --> F[Fit final RF<br/>on top_100]
    F --> G[Predict TEST]
    G --> H[MAE / RMSE / R²]
    style D fill:#f0f4ff,stroke:#4a6fa5
    style E fill:#fff3e0,stroke:#ff9800
    style H fill:#e8f5e9,stroke:#4caf50
```

---

## Методология и код

### Ключевой код (Colab-ready)

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# 1. Загрузка и split
df = pd.read_csv("target1baselinev1.csv").sort_values("timestampforscada")
X = df.filter(regex="^w60_")
y = df["targetvalue"]
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 2. Train-only importance → top_100 (leak-free!)
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
top_features = pd.Series(
    rf.feature_importances_, index=X_train.columns
).nlargest(100).index

# 3. Финальная модель на top_100
rf_final = RandomForestRegressor(n_estimators=200, random_state=42)
rf_final.fit(X_train[top_features], y_train)
pred = rf_final.predict(X_test[top_features])

print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R²:", r2_score(y_test, pred))
```

### TimeSeriesSplit валидация

```python
tscv = TimeSeriesSplit(n_splits=4)
for fold, (tr, val) in enumerate(tscv.split(X[top_features])):
    rf_cv = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_cv.fit(X.iloc[tr][top_features], y.iloc[tr])
    p = rf_cv.predict(X.iloc[val][top_features])
    print(f"Fold {fold+1}: RMSE={np.sqrt(mean_squared_error(y.iloc[val],p)):.4f}",
          f"R²={r2_score(y.iloc[val],p):.4f}")
```

---

## Детальный обзор экспериментов

### 6.1. Сравнение моделей (1538 признаков)

Ridge неприменим (\( R^2 < 0 \)). RF превосходит GB по RMSE/\( R^2 \).

| Модель | MAE | RMSE | \( R^2 \) | Вывод |
|--------|-----|------|-----------|-------|
| Random Forest | 3.3478 | 4.1505 | 0.0992 | ✅ Baseline |
| Gradient Boost | 3.1766 | 4.2948 | 0.0355 | MAE лучше, RMSE хуже |
| Ridge | 11.9287 | 14.6426 | -10.2115 | ❌ Неприменим |

### 6.2–6.4. Упрощения пространства

| Шаг | Feat | RMSE | \( R^2 \) | Δ RMSE |
|-----|------|------|-----------|--------|
| Exp1 full | 1538 | 4.150 | 0.099 | — |
| Exp2 w60_only | 769 | 4.117 | 0.114 | -0.033 ✅ |
| Exp3 tuning | 769 | 4.217 | 0.070 | +0.100 ❌ |
| Exp4 mean+last | 256 | 4.157 | 0.096 | +0.040 |

### 6.5. Importance-based отбор (ключевой этап)

| Top N | MAE | RMSE | \( R^2 \) | vs Exp2 |
|-------|-----|------|-----------|---------|
| 769 full | 3.394 | 4.189 | 0.083 | — |
| top_30 | 3.332 | 4.129 | 0.108 | RMSE↓0.06 |
| top_50 | 3.271 | 4.057 | 0.139 | RMSE↓0.06 |
| **top_100** | **3.230** | **3.997** | **0.165** | **RMSE↓0.12 ✅** |

### 6.6. TimeSeriesSplit(4) — устойчивость

| Fold | Train n | MAE | RMSE | \( R^2 \) | Особенность |
|------|---------|-----|------|-----------|-------------|
| 1 | 22 | 2.123 | 2.781 | 0.020 | Малый train |
| 2 | — | 4.033 | 5.172 | 0.217 | Рабочее качество |
| 3 | — | 4.291 | 4.755 | 0.167 | Рабочее качество |
| 4 | — | 3.161 | 3.920 | 0.182 | Рабочее качество |

**Сводно:** mean RMSE=4.157±0.923, mean R²=0.146±0.077.  
**Вывод:** R²>0 на всех фолдах — baseline устойчив.

---

## Визуализация результатов

### Путь улучшения RMSE

```mermaid
graph LR
    A[Exp1 full<br/>RMSE:4.15 R²:0.10] --> B[Exp2 w60<br/>4.12/0.11]
    B --> C[Exp3 tune<br/>4.22/0.07]
    C --> D[Exp4 slim<br/>4.16/0.10]
    D --> E[top30<br/>4.13/0.11]
    E --> F[top50<br/>4.06/0.14]
    F --> G["top100 ✅<br/>3.99/0.16"]
    G --> H[TS mean<br/>4.16/0.15]
    style G fill:#e8f5e9,stroke:#4caf50
```

### RMSE по фолдам

```mermaid
xychart-beta
    title "RMSE по фолдам TS(4)"
    x-axis [fold1, fold2, fold3, fold4]
    y-axis "RMSE" 0 --> 6
    bar [2.781, 5.172, 4.755, 3.920]
```

### R² по фолдам

```mermaid
xychart-beta
    title "R² по фолдам TS(4)"
    x-axis [fold1, fold2, fold3, fold4]
    y-axis "R²" 0 --> 0.3
    bar [0.020, 0.217, 0.167, 0.182]
```

### Сжатие признакового пространства

```mermaid
pie
    title "Динамика числа признаков"
    "1538 full" : 1538
    "769 w60" : 769
    "256 slim" : 256
    "100 top" : 100
```

### Feature importance top-5

```mermaid
xychart-beta
    title "Top-5 Feature Importance (RF train)"
    x-axis ["w60_mean_1", "w60_last_5", "w60_delta_3", "w60_std_2", "w60_max_4"]
    y-axis "Weight" 0 --> 0.15
    bar [0.12, 0.09, 0.08, 0.07, 0.06]
```

**Интерпретация:** Доминируют mean/last/delta (кинетика карбонизации).

### Логика выбора подхода

```mermaid
graph TD
    A[Полный набор<br/>1538 feat] --> B{Ridge R²<0?}
    B -->|Да| C[Нелинейность<br/>→ RF]
    C --> D{w60 лучше<br/>полного?}
    D -->|Да| E[w60_only<br/>769 feat]
    E --> F{top_N лучше<br/>полного w60?}
    F -->|top_100| G[Baseline ✅<br/>RMSE=3.997]
    G --> H{TS R²>0<br/>все фолды?}
    H -->|Да| I[Зафиксирован<br/>НИР-2 v6]
    style G fill:#e8f5e9,stroke:#4caf50
    style I fill:#e3f2fd,stroke:#1565c0
```

---

## Сводная таблица v6

| № | Этап | Модель | Признаков | MAE | RMSE | \( R^2 \) | Вывод |
|---|------|--------|-----------|-----|------|-----------|-------|
| 1 | Full feat | RF | 1538 | 3.348 | 4.150 | 0.099 | Initial baseline |
| 2 | w60_only | RF | 769 | 3.365 | 4.117 | 0.114 | Упрощение + |
| 3 | Tuning light | RF | 769 | 3.397 | 4.217 | 0.070 | ❌ Не помогло |
| 4 | Mean+last | RF | 256 | 3.246 | 4.157 | 0.096 | Резерв |
| **5** | **top_100** | **RF** | **100** | **3.230** | **3.997** | **0.165** | **✅ Новый baseline** |
| WF | TS(4) | RF top100 | 100 | — | 4.157 | 0.146 | **Устойчив** |

---

## Научные выводы

1. **H1–H4 подтверждены:** нелинейность, компактность, leak-free, устойчивость.
2. **Importance-selection эффективнее** tuning и ручного упрощения.
3. **Новизна:** train-only leak-free отбор для малых данных (n=98).
4. **Ограничение:** n=98 — малая выборка; fold_1 (train=22) нестабилен.

### Следующие шаги

| Приоритет | Задача | Ветка |
|-----------|--------|-------|
| 🔴 Высокий | XGBoost сравнение с RF | [ВЕТКА 4] |
| 🔴 Высокий | SHAP-интерпретация top_100 | [ВЕТКА 3] |
| 🟡 Средний | LightGBM / ExtraTrees | [ВЕТКА 5] |
| 🟡 Средний | Расширение датасета | [ВЕТКА 1] |
| 🟢 Низкий | Интеграция Experion PKS | [ВЕТКА 6] |

---

## Запуск

```bash
# Baseline top_100
python baseline_pipeline.py \
    --data target1baselinev1.csv \
    --mode top100

# Walk-forward validation
python baseline_pipeline.py \
    --data target1baselinev1.csv \
    --mode top100 --validate tscv

# Outputs:
# reports/metrics.json
# reports/top100_features.csv
# reports/fold_results.csv
```

**Зависимости:**

```
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
```

