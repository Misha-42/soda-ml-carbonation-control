# Baseline report for carbonation shift targets

## Summary

A baseline pipeline was built for three shift-level targets of the carbonation process:
- target_t
- target_cl
- target_nh3

Configuration:
- dataset: shift_dataset
- model: rf_shift_whitelist
- split:
  - train < 2026-03-01
  - test >= 2026-03-01

## Metrics

target_t:
- MAE = 0.387203
- RMSE = 0.475321
- R2 = 0.036030

target_cl:
- MAE = 0.733118
- RMSE = 0.806518
- R2 = 0.006915

target_nh3:
- MAE = 1.334134
- RMSE = 1.535068
- R2 = -0.009601

## Interpretation

- best target: target_t
- second: target_cl
- hardest: target_nh3
- pipeline is working
- current results should be treated as baseline, not final model

## Update after model comparison

Best fixed baselines by target:
- target_t -> rf_shift_whitelist
- target_cl -> ridge
- target_nh3 -> rf_shift_whitelist

Metrics:
- target_t: MAE=0.387203, RMSE=0.475321, R2=0.036030
- target_cl: MAE=0.645223, RMSE=0.773226, R2=0.087208
- target_nh3: MAE=1.334134, RMSE=1.535068, R2=-0.009601
