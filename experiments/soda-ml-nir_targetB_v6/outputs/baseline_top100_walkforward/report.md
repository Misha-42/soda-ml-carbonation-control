# Отчёт: target1_baseline_top100_walkforward

## Конфигурация
- Датасет: `C:\Users\user\Desktop\soda-ml-nir-main\soda-ml-nir-main\soda-ml-nir_targetB_v6\launch_target1_k1\dataset_target1_baseline_v1.csv`
- Feature set: `C:\Users\user\Desktop\soda-ml-nir-main\soda-ml-nir-main\soda-ml-nir_targetB_v6\outputs\exp_05_target1_topn_importance\baseline_feature_set_top100.csv`
- Число строк: 98
- Число признаков: 100
- Модель: Random Forest
- Параметры RF: `{'n_estimators': 300, 'random_state': 42, 'n_jobs': -1}`
- Временная валидация: TimeSeriesSplit(n_splits=4)

## Метрики по фолдам
Чем меньше MAE/RMSE и выше R², тем лучше.

### fold_1
- train_size: 22
- test_size: 19
- MAE: 2.122904
- RMSE: 2.781229
- R2: 0.019812

### fold_2
- train_size: 41
- test_size: 19
- MAE: 4.033000
- RMSE: 5.172466
- R2: 0.216860

### fold_3
- train_size: 60
- test_size: 19
- MAE: 4.291105
- RMSE: 4.755381
- R2: 0.166539

### fold_4
- train_size: 79
- test_size: 19
- MAE: 3.160833
- RMSE: 3.920155
- R2: 0.181504

## Сводка
- mean MAE: 3.401961
- std MAE: 0.848958
- mean RMSE: 4.157308
- std RMSE: 0.913520
- mean R2: 0.146179
- std R2: 0.075211

## Интерпретация
- Если RMSE не разваливается по фолдам, а R² не уходит системно в отрицательную зону, baseline можно считать рабоче устойчивым.