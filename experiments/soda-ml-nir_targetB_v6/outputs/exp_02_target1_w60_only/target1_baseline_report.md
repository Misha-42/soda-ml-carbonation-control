# Target1 baseline report

## Dataset
- Data path: `C:\Users\user\Desktop\soda-ml-nir-main\soda-ml-nir-main\soda-ml-nir_targetB_v6\launch_target1_k1\dataset_target1_baseline_v1.csv`
- Rows total: 98
- Train rows: 78
- Test rows: 20
- Features: 769
- Target: `target_value`
- Time column: `target_timestamp_for_scada`

## Best model
- Model: gradient_boosting
- MAE: 3.349881
- RMSE: 4.444945
- R2: -0.033146

## Models compared
- ridge
- random_forest
- gradient_boosting

## Notes
- Использованы только признаки с префиксами `w60__` и `w120_30__`
- Использован time-based split без shuffle
- Ridge здесь выступает как линейный baseline
- RandomForest и GradientBoosting — основные нелинейные baseline-модели

## Full results

| model             |      mae |     rmse |          r2 |   train_rows |   test_rows |   n_features |
|:------------------|---------:|---------:|------------:|-------------:|------------:|-------------:|
| gradient_boosting |  3.34988 |  4.44495 |  -0.0331458 |           78 |          20 |          769 |
| random_forest     |  3.36505 |  4.11667 |   0.113823  |           78 |          20 |          769 |
| ridge             | 23.9758  | 27.7492  | -39.2652    |           78 |          20 |          769 |
