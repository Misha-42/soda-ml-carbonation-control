# Baseline report

Сравнение baseline-моделей (чем меньше MAE/RMSE и выше R2, тем лучше):

- RandomForestRegressor (rf_small): MAE=1.7022, RMSE=2.3115, R2=0.3244
- RandomForestRegressor (rf_medium): MAE=1.7412, RMSE=2.3473, R2=0.3033
- XGBRegressor (xgb_small): MAE=1.5904, RMSE=2.2084, R2=0.3833
- XGBRegressor (xgb_medium): MAE=1.6535, RMSE=2.3939, R2=0.2754

Лучшая конфигурация по MAE: XGBRegressor (xgb_small) (MAE=1.5904)