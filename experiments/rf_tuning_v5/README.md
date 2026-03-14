

# Baseline ML-контур для карбонизации соды: RF + XGBoost

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![sklearn](https://img.shields.io/badge/sklearn-1.3-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red)](https://xgboost.readthedocs.io)
[![НИР-2](https://img.shields.io/badge/НИР--2-baseline%20v5-green)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

Воспроизводимый ML-baseline для анализа и прогнозирования технологического
показателя `k1` процесса карбонизации кальцинированной соды
(АО «Башкирская содовая компания», НИР-2 УГНТУ).

> **Задача:** Построить soft sensor — прогноз `k1` по данным промышленных датчиков,
> сравнить Random Forest и XGBoost, зафиксировать baseline для дальнейшего развития.

---

## Содержание

- [Мотивация](#мотивация)
- [Постановка задачи](#постановка-задачи)
- [Данные](#данные)
- [Архитектура pipeline](#архитектура-pipeline)
- [Модели и методы](#модели-и-методы)
- [Результаты](#результаты)
- [Визуализация](#визуализация)
- [Структура репозитория](#структура-репозитория)
- [Запуск](#запуск)
- [Следующие шаги](#следующие-шаги)

---

## Мотивация

Карбонизация аммонизированного рассола — центральная стадия Solvay-процесса.
Переменное качество CO2, известкового молока и входного рассола вызывают
отклонения `k1` (целевого показателя), что ведёт к браку и перерасходу реагентов.

**Проблема:** Лабораторный контроль — раз в смену (4–8 ч задержки).  
**Решение:** ML soft sensor — онлайн-прогноз по SCADA-параметрам.

```mermaid
flowchart LR
    A[SCADA<br/>датчики] --> B[ML soft sensor<br/>RF / XGBoost]
    B --> C[Прогноз k1<br/>±30 мин]
    C --> D[Оператор<br/>проактивно]
    D --> E[Брак ↓<br/>реагенты ↓]
```

**Обоснование выбора моделей:**

| Модель | Преимущество | Риск |
|--------|-------------|------|
| `RandomForestRegressor` | Устойчив к выбросам, интерпретируем | Медленнее при больших данных |
| `XGBRegressor` | Выше точность, встроенная регуляризация | Склонен к переобучению на малых n |

---

## Постановка задачи

**Цель:** Построить воспроизводимый baseline RF vs XGBoost для `k1`,
зафиксировать метрики контрольной точки НИР-2.

**Гипотезы:**

- **H1:** XGBoost превзойдёт RF по RMSE при достаточном объёме данных.
- **H2:** 6-минутные агрегации (w6) информативнее raw-признаков.
- **H3:** Отбор top-N признаков по importance снижает RMSE без ухудшения R².

**Метрики:** MAE, RMSE, \( R^2 \).  
**Сравнение с baseline:** mean-pred, последнее значение.

---

## Данные

| Параметр | Значение |
|----------|----------|
| Файл | `data/baseline_k1_6min_real.csv` |
| Таргет | `k1` (целевой показатель карбонизации) |
| Временная колонка | `timestamp` (опц.) |
| Агрегация | 6-мин. окна (mean/std/min/max/last) |
| Split | time-based, 80/20, no shuffle |

**Схема данных:**

| Колонка | Тип | Описание |
|---------|-----|----------|
| `timestamp` | datetime | Метка времени (опц.) |
| `target` | float | Целевая переменная k1 |
| `feat_*` | float | SCADA-параметры |

**Preprocessing:**

```mermaid
flowchart TD
    A[CSV raw] --> B[Проверка колонок<br/>target, timestamp]
    B --> C[Очистка<br/>NaN, outliers]
    C --> D[Агрегация 6 мин<br/>mean/std/min/max/last]
    D --> E[Time-based split<br/>80/20 no leakage]
    E --> F[Train / Test]
```

---

## Архитектура pipeline

```mermaid
graph TB
    A[data/baseline_k1_6min_real.csv] --> B[src/data_prep.py<br/>load + clean + split]
    B --> C[src/features.py<br/>numeric select + impute]
    C --> D[src/train_baseline.py]
    D --> E[RandomForestRegressor<br/>small / medium]
    D --> F[XGBRegressor<br/>small / medium]
    E --> G[src/evaluate.py]
    F --> G
    G --> H[reports/<br/>metrics.csv + plots]
    G --> I[models/<br/>*.joblib]
    style E fill:#e8f4fd,stroke:#2196f3
    style F fill:#fff3e0,stroke:#ff9800
    style G fill:#e8f5e9,stroke:#4caf50
```

---

## Модели и методы

### Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

rf_small = RandomForestRegressor(
    n_estimators=100, max_depth=None,
    random_state=42, n_jobs=-1
)
rf_medium = RandomForestRegressor(
    n_estimators=300, max_depth=12,
    min_samples_leaf=2, max_features="sqrt",
    random_state=42, n_jobs=-1
)
```

### XGBoost

```python
from xgboost import XGBRegressor

xgb_small = XGBRegressor(
    n_estimators=100, learning_rate=0.1,
    max_depth=6, random_state=42
)
xgb_medium = XGBRegressor(
    n_estimators=300, learning_rate=0.05,
    max_depth=8, subsample=0.8,
    colsample_bytree=0.8, random_state=42
)
```

### Полный pipeline запуска

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {
        "MAE":  mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R²":   r2_score(y_test, pred)
    }
```

---

## Результаты

### Итоговые метрики

| Модель | Конфиг | MAE | RMSE | \( R^2 \) | Статус |
|--------|--------|-----|------|-----------|--------|
| Random Forest | small | — | — | — | ✅ |
| Random Forest | medium | — | — | — | ✅ |
| XGBoost | small | — | — | — | ✅ |
| XGBoost | medium | — | — | — | ✅ |

> Метрики заполняются автоматически после запуска pipeline.
> Актуальные значения: `reports/baseline_metrics.csv`

### Сохранённые модели

```text
models/
├── rf_small.joblib       # RF 100 деревьев
├── rf_medium.joblib      # RF 300 деревьев (tuned)
├── xgb_small.joblib      # XGB базовый
└── xgb_medium.joblib     # XGB расширенный
```

---

## Визуализация

### Динамика метрик (template)

```mermaid
xychart-beta
    title "RMSE: RF vs XGBoost"
    x-axis ["RF small", "RF medium", "XGB small", "XGB medium"]
    y-axis "RMSE" 0 --> 5
    bar [3.5, 3.2, 3.3, 3.0]
```

```mermaid
xychart-beta
    title "R² по конфигурациям"
    x-axis ["RF small", "RF medium", "XGB small", "XGB medium"]
    y-axis "R²" 0 --> 0.3
    bar [0.10, 0.14, 0.12, 0.18]
```

### Выбор лучшей модели

```mermaid
graph TD
    A[Запуск baseline] --> B{XGB RMSE < RF RMSE?}
    B -->|Да| C[XGBoost winner<br/>→ tune hyperparams]
    B -->|Нет| D[RandomForest winner<br/>→ importance selection]
    C --> E{R² > 0.15?}
    D --> E
    E -->|Да| F[Baseline зафиксирован ✅]
    E -->|Нет| G[Feat eng + expand data]
```

### Графики (PNG — папка reports/)

**[Рис. 1]** Сравнение MAE:
![MAE models](reports/сравнение_моделей_MAE.png)

**[Рис. 2]** Сравнение RMSE:
![RMSE models](reports/сравнение_моделей_RMSE.png)

**[Рис. 3]** Feature importance — Random Forest:
![RF importance](reports/важность_признаков_RandomForest.png)

**[Рис. 4]** Feature importance — XGBoost:
![XGB importance](reports/важность_признаков_XGBoost.png)

---

## Структура репозитория

```text
rf_tuning_v5/
│
├── data/
│   └── baseline_k1_6min_real.csv     # Входные данные
│
├── models/
│   ├── rf_small.joblib
│   ├── rf_medium.joblib
│   ├── xgb_small.joblib
│   └── xgb_medium.joblib
│
├── nir/
│   ├── 03_data.md                    # Описание данных
│   ├── 04_methods.md                 # Методы
│   ├── 05_experiments.md             # Эксперименты
│   ├── 06_results.md                 # Результаты
│   └── 07_conclusion.md              # Выводы
│
├── reports/
│   ├── baseline_metrics.csv
│   ├── experiments_summary.csv
│   ├── baseline_report.md
│   ├── rf_tuning_v5_metrics.csv
│   ├── rf_tuning_v5_report.md
│   ├── сравнение_моделей_MAE.png
│   ├── сравнение_моделей_RMSE.png
│   ├── важность_признаков_RandomForest.png
│   └── важность_признаков_XGBoost.png
│
├── src/
│   ├── data_prep.py                  # Загрузка, очистка, split
│   ├── features.py                   # Генерация признаков
│   ├── train_baseline.py             # Обучение RF + XGB
│   └── evaluate.py                   # Метрики + графики
│
├── requirements.txt
└── README.md
```

---

## Запуск

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Baseline с временной колонкой
python src/train_baseline.py \
    --data-path data/baseline_k1_6min_real.csv \
    --target target \
    --time-column timestamp

# 3. Baseline без временной колонки
python src/train_baseline.py \
    --data-path data/file.csv \
    --target target

# 4. Вывод: models/ + reports/
```

**Зависимости (`requirements.txt`):**

```
scikit-learn>=1.3
xgboost>=1.7
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
joblib>=1.3
```

