# Отчёт: B1_baseline_k1_rf_v1

## Конфигурация
- Датасет: `C:\Users\user\Desktop\soda-ml-nir-main\soda-ml-nir-main\soda-ml-nir_targetB_v6\merged_dataset_B1_v1.xlsx`
- Лист: `MERGED_B1`
- Target: `target_B1_sv_NH3_susp`
- Proxy time: `time_idx` = `lab_date + shift`
- Число строк: 34
- Число признаков k1_*: 10
- Train size: 27
- Test size: 7
- Модель: Random Forest
- Параметры RF: `{'n_estimators': 300, 'random_state': 42, 'n_jobs': -1}`

## Метрики
- MAE: 1.515238
- RMSE: 1.641406
- R2: -2.034863

## Технические замечания
- Это первый baseline для B1 на merged lab+SCADA данных.
- Все отобранные k1_* признаки числовые.