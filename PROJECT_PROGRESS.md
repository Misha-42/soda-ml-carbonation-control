# PROJECT PROGRESS

## Baseline for carbonation shift targets

Статус: зафиксирован

Построен и зафиксирован baseline для 3 сменных таргетов процесса карбонизации БСК.

Контекст:
- источник данных: 17 Excel-файлов
- период: 20.02.2026–08.03.2026
- единица наблюдения: 1 строка = 1 смена
- размер посменного датасета: 34 строки
- split:
  - train: даты < 2026-03-01
  - test: даты >= 2026-03-01
- общая baseline-модель: RandomForestRegressor
- пространство признаков: ручной white-list технологически осмысленных признаков
- baseline-конфигурация: rf_shift_whitelist

Итоговые результаты:

### 1. target_t
- best_model = rf_shift_whitelist
- MAE = 0.387203
- RMSE = 0.475321
- R2 = 0.036030

### 2. target_cl
- best_model = rf_shift_whitelist
- MAE = 0.733118
- RMSE = 0.806518
- R2 = 0.006915

### 3. target_nh3
- best_model = rf_shift_whitelist
- MAE = 1.334134
- RMSE = 1.535068
- R2 = -0.009601

Вывод:
- pipeline рабочий
- ручной white-list признаков оправдан
- лучший baseline получен для target_t
- target_cl тоже прогнозируется лучше нуля по R2
- target_nh3 пока является самым сложным таргетом
- результаты следует рассматривать как baseline, а не как финальную модель

Ранжирование таргетов:
1. target_t
2. target_cl
3. target_nh3

Текущее решение:
- baseline для всех 3 таргетов зафиксировать
- приоритет дальнейшего улучшения: target_t, затем target_cl
- target_nh3 вести как отдельную более сложную ветку

Следующий этап:
- сравнить RF vs Ridge vs ElasticNet для target_t и target_cl
- улучшать признаки
- расширять объём сменных данных

## Update: best models after RF vs Ridge vs ElasticNet comparison

Дополнительное сравнение моделей показало, что единая baseline-модель не является оптимальной для всех таргетов.

Обновлённые лучшие модели:
- target_t → rf_shift_whitelist
  - MAE = 0.387203
  - RMSE = 0.475321
  - R2 = 0.036030

- target_cl → ridge
  - MAE = 0.645223
  - RMSE = 0.773226
  - R2 = 0.087208

- target_nh3 → rf_shift_whitelist
  - MAE = 1.334134
  - RMSE = 1.535068
  - R2 = -0.009601

Вывод:
- target_t лучше описывается нелинейной моделью Random Forest
- target_cl лучше описывается регуляризованной линейной моделью Ridge
- target_nh3 остаётся самым сложным таргетом
