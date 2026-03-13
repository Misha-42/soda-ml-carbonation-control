# Baseline report

Сравнение baseline-моделей (чем меньше MAE/RMSE и выше R2, тем лучше):

- RandomForestRegressor (rf_small): MAE=0.5816, RMSE=0.8743, R2=0.9033
- RandomForestRegressor (rf_tuned_150): MAE=0.5933, RMSE=0.8985, R2=0.8979
- RandomForestRegressor (rf_tuned_300): MAE=0.5850, RMSE=0.8844, R2=0.9011
- RandomForestRegressor (rf_tuned_leaf3): MAE=0.5536, RMSE=0.7995, R2=0.9192
- XGBRegressor (xgb_small): MAE=0.6961, RMSE=1.0646, R2=0.8567

Лучшая конфигурация по MAE: RandomForestRegressor (rf_tuned_leaf3) (MAE=0.5536)