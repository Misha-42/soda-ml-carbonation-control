# Baseline report

Сравнение baseline-моделей (чем меньше MAE/RMSE и выше R2, тем лучше):

- RandomForestRegressor (rf_small): MAE=0.6369, RMSE=0.9426, R2=0.8877
- RandomForestRegressor (rf_medium): MAE=0.6245, RMSE=0.9311, R2=0.8904
- XGBRegressor (xgb_small): MAE=0.7239, RMSE=1.0579, R2=0.8585
- XGBRegressor (xgb_medium): MAE=0.7475, RMSE=1.1113, R2=0.8439

Лучшая конфигурация по MAE: RandomForestRegressor (rf_medium) (MAE=0.6245)