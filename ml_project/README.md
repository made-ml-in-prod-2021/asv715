**Запуск обучения:**

`python ml_process/train_pipeline.py configs/train.logreg.conf.yml`, где `configs/train.logreg.conf.yml` - путь до конфигурационного файла в yml-формате

**Получение предсказаний модели:**

`python ml_process/predict.py configs/train.logreg.conf.yml models/model.pkl tests/test.csv`, где
`configs/train.logreg.conf.yml` - путь до конфигурационного файла в yml-формате, `models/model.pkl` - путь до файла с обученной моделью в pkl-формате, `tests/test.csv` - путь до файла с тестовой выборкой

**Запуск тестов:**

`pytest -v -p no:warnings tests`

**Прогон линтера:**

`pylint ml_process`