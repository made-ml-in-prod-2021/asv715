**Запуск обучения:**

`python ml_process/train_pipeline.py config_path`, где `config_path` - путь до конфигурационного файла в yml-формате

**Получение предсказаний модели:**

`python ml_process/predict.py config_path model_path test_sample_path`, где
`config_path` - путь до конфигурационного файла в yml-формате, `model_path` - путь до файла с обученной моделью в pkl-формате, `test_sample_path` - путь до файла с тестовой выборкой

**Запуск тестов:**

`pytest tests`

**Прогон линтера:**

`pylint ml_process`