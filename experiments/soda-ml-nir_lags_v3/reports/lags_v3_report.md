# Baseline report

Сравнение baseline-моделей (чем меньше MAE/RMSE и выше R2, тем лучше):

- RandomForestRegressor (rf_small): MAE=0.5816, RMSE=0.8743, R2=0.9033
- RandomForestRegressor (rf_medium): MAE=0.5896, RMSE=0.8879, R2=0.9003
- XGBRegressor (xgb_small): MAE=0.6961, RMSE=1.0646, R2=0.8567
- XGBRegressor (xgb_medium): MAE=0.7157, RMSE=1.0816, R2=0.8521

Лучшая конфигурация по MAE: RandomForestRegressor (rf_small) (MAE=0.5816)