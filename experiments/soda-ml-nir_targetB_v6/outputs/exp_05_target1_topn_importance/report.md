# Отчёт: target1_exp05_topn_importance

## Датасет
- Путь: `C:\Users\user\Desktop\soda-ml-nir-main\soda-ml-nir-main\soda-ml-nir_targetB_v6\launch_target1_k1\dataset_target1_baseline_v1.csv`
- Число строк: 98
- Полное число признаков `w60__`: 769
- Целевая колонка: `target_value`
- Временная колонка: `target_timestamp_for_scada`

## Разбиение
- Размер train: 78
- Размер test: 20
- Метод: time_based_80_20_no_shuffle

## Базовый вариант: полный w60
- MAE: 3.394114
- RMSE: 4.188671
- R2: 0.082552
- delta RMSE vs exp_02: 0.072003
- delta R2 vs exp_02: -0.031271

## Варианты top-N
Чем меньше MAE/RMSE и выше R2, тем лучше.

### top_30
- Число признаков: 30
- MAE: 3.332403
- RMSE: 4.129147
- R2: 0.108442
- delta RMSE vs exp_02: 0.012479
- delta R2 vs exp_02: -0.005381

### top_50
- Число признаков: 50
- MAE: 3.270822
- RMSE: 4.056952
- R2: 0.139346
- delta RMSE vs exp_02: -0.059716
- delta R2 vs exp_02: 0.025523

### top_100
- Число признаков: 100
- MAE: 3.229928
- RMSE: 3.996800
- R2: 0.164679
- delta RMSE vs exp_02: -0.119868
- delta R2 vs exp_02: 0.050856

## Вывод
- Лучший вариант внутри exp_05 по RMSE: top_100 (RMSE=3.996800)
- Сравнение с exp_02 нужно проводить прежде всего по RMSE и R2.
- Если один из вариантов top-N близок к exp_02 или лучше него, его можно считать кандидатом на новый компактный baseline.