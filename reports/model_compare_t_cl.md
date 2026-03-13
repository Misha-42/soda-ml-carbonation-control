# Model comparison for target_t and target_cl

## Setup
- dataset: shift_dataset
- features: white_list_nh3_shift
- split:
  - train: date < 2026-03-01
  - test: date >= 2026-03-01
- models:
  - rf_shift_whitelist
  - ridge
  - elasticnet

## Results for target_t
- `rf_shift_whitelist`: MAE=0.387203, RMSE=0.475321, R2=0.036030
- `elasticnet`: MAE=0.573836, RMSE=0.682618, R2=-0.988125
- `ridge`: MAE=0.629179, RMSE=0.783805, R2=-1.621230

Итог:
- лучший по MAE: `rf_shift_whitelist`
  - MAE = 0.387203
  - RMSE = 0.475321
  - R2 = 0.036030


## Results for target_cl
- `ridge`: MAE=0.645223, RMSE=0.773226, R2=0.087208
- `elasticnet`: MAE=0.660437, RMSE=0.753044, R2=0.134236
- `rf_shift_whitelist`: MAE=0.733118, RMSE=0.806518, R2=0.006915

Итог:
- лучший по MAE: `ridge`
  - MAE = 0.645223
  - RMSE = 0.773226
  - R2 = 0.087208


## Files
- CSV summary: `/content/soda-ml-nir-main/reports/model_compare_t_cl.csv`
- CSV predictions: `/content/soda-ml-nir-main/reports/model_compare_t_cl_predictions.csv`