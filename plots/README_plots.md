# README for plots

## Назначение папки

В папке `plots/` собраны графики для визуального анализа baseline-моделей по сменным таргетам карбонизации БСК:
- `target_t`
- `target_cl`
- `target_nh3`

Эти графики можно использовать в НИР, отчёте и презентации.

## Что уже зафиксировано по моделям

- `target_t` -> `rf_shift_whitelist`
- `target_cl` -> `ridge`
- `target_nh3` -> `rf_shift_whitelist`

## Рекомендация по вставке в НИР

### 1. Основные графики для раздела "Результаты"
Использовать в первую очередь:

- `target_t_actual_vs_pred.png`
- `target_cl_actual_vs_pred.png`
- `target_nh3_actual_vs_pred.png`

Назначение:
- показать визуальное соответствие факта и прогноза по каждому таргету;
- быстро сравнить качество baseline-моделей.

### 2. Графики для раздела "Анализ качества модели"
Использовать:

- `target_t_scatter_actual_vs_pred.png`
- `target_cl_scatter_actual_vs_pred.png`
- `target_nh3_scatter_actual_vs_pred.png`

Назначение:
- показать близость предсказаний к идеальной диагонали `y=x`;
- визуально оценить качество прогноза.

### 3. Графики для раздела "Анализ ошибок"
Использовать:

- `target_t_residuals.png`
- `target_cl_residuals.png`
- `target_nh3_residuals.png`

Назначение:
- показать наличие или отсутствие систематического смещения;
- проверить структуру ошибок.

### 4. Гистограммы остатков
Использовать:

- `target_t_residuals_hist.png`
- `target_cl_residuals_hist.png`
- `target_nh3_residuals_hist.png`

Назначение:
- показать ширину распределения ошибок;
- визуально сравнить стабильность baseline по таргетам.

### 5. Сравнение моделей
Использовать:

- `model_compare_t_cl_mae.png`
- `model_compare_t_cl_r2.png`

Назначение:
- показать, что для `target_cl` линейная модель лучше RF;
- показать, что для `target_t` RF остался лучшим.

## Краткая интерпретация для НИР

- `target_t` визуально должен выглядеть как лучший таргет:
  - более плотное совпадение факта и прогноза
  - меньший разброс остатков
  - более компактная гистограмма ошибок

- `target_cl` должен выглядеть как умеренно хороший таргет:
  - качество ниже, чем у `target_t`
  - но структура ошибки всё ещё приемлемая
  - лучшая модель: `ridge`

- `target_nh3` должен выглядеть как самый сложный таргет:
  - наибольший разброс ошибок
  - слабее визуальное совпадение факта и прогноза
  - baseline пока остаётся контрольным, а не рабочим целевым решением

## Список файлов в plots/

- `model_compare_t_cl_mae.png`
- `model_compare_t_cl_r2.png`
- `target_cl_actual_vs_pred.png`
- `target_cl_residuals.png`
- `target_cl_residuals_hist.png`
- `target_cl_scatter_actual_vs_pred.png`
- `target_nh3_actual_vs_pred.png`
- `target_nh3_residuals.png`
- `target_nh3_residuals_hist.png`
- `target_nh3_scatter_actual_vs_pred.png`
- `target_t_actual_vs_pred.png`
- `target_t_residuals.png`
- `target_t_residuals_hist.png`
- `target_t_scatter_actual_vs_pred.png`

## Рекомендуемый минимальный набор рисунков для НИР

Если нужно вставить только 4–6 рисунков, оптимальный набор такой:

1. `target_t_actual_vs_pred.png`
2. `target_cl_actual_vs_pred.png`
3. `target_nh3_actual_vs_pred.png`
4. `model_compare_t_cl_mae.png`
5. `target_t_residuals_hist.png`
6. `target_nh3_residuals_hist.png`

## Вывод

Папка `plots/` содержит достаточный baseline-визуальный пакет для оформления результатов этапа НИР и для обсуждения качества моделей по каждому таргету.
