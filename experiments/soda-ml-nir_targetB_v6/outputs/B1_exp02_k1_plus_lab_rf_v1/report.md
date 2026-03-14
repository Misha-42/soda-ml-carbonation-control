# Отчёт: B1_exp02_k1_plus_lab_rf_v1

## Конфигурация
- Датасет: `C:\Users\user\Desktop\soda-ml-nir-main\soda-ml-nir-main\soda-ml-nir_targetB_v6\merged_dataset_B1_v1.xlsx`
- Лист: `MERGED_B1`
- Target: `target_B1_sv_NH3_susp`
- Proxy time: `time_idx` = `lab_date + shift`
- Число строк: 34
- Всего признаков: 14
- k1_* признаков: 10
- lab признаков: 4
- Train size: 27
- Test size: 7
- Модель: Random Forest
- Параметры RF: `{'n_estimators': 300, 'random_state': 42, 'n_jobs': -1}`

## Метрики
- MAE: 1.675048
- RMSE: 1.851729
- R2: -2.862441

## Интерпретация
- Это диагностический запуск: k1_* + lab features.
- Если метрики заметно улучшатся, значит baseline на одних k1_* был слишком узким.