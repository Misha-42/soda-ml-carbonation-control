K1 TARGET1 LAUNCH BUNDLE

Канонический минимальный набор для быстрого запуска исследования по target1.

Главные входы:
1. dataset_target1LAB_baseline_v1.csv — lab-only основа
2. dataset_target1_baseline_v1.csv — текущий merged dataset
3. кк1.xlsx — узкий SCADA-файл по карбоколонне №1

Опциональные SCADA-источники:
- 6_min.xlsx
- 1_haur.xlsx

Контрольные файлы:
- dataset_target1LAB_baseline_v1_date_counts.csv
- dataset_target1LAB_baseline_v1_parts_summary.csv
- dataset_target1LAB_baseline_v1_preview.csv
- dataset_target1_baseline_v1_preview.csv

Справка:
- LAB66_ABBREVIATIONS.csv
- LAB66_DAYS_HOURS.csv
- LAB66_PARAMETER_REGISTRY.csv
- LAB66_SHEET_PARAMETER_COUNTS.csv

Рекомендуемый порядок:
1. Проверить dataset_target1LAB_baseline_v1.csv
2. Использовать dataset_target1_baseline_v1.csv как стартовый baseline-dataset
3. Если нужен более узкий вариант — пересобрать merged dataset только через кк1.xlsx
4. Затем считать baseline_results_target1_v1.csv

Статус готовности:
- lab слой: готов
- merged dataset: готов
- baseline results: еще не посчитаны в этом наборе