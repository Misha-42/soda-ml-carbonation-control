# Baseline report

Сравнение baseline-моделей (чем меньше MAE/RMSE и выше R2, тем лучше):

- RandomForestRegressor (rf_small): MAE=1.7779, RMSE=2.3986, R2=0.2725
- RandomForestRegressor (rf_medium): MAE=1.8094, RMSE=2.4380, R2=0.2484
- XGBRegressor (xgb_small): MAE=1.6036, RMSE=2.2529, R2=0.3582
- XGBRegressor (xgb_medium): MAE=1.7633, RMSE=2.4944, R2=0.2132

Лучшая конфигурация по MAE: XGBRegressor (xgb_small) (MAE=1.6036)