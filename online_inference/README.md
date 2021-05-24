**Загрузка образа с Docker Hub:**

`docker pull asv715/ml_prod:v1`

**Сборка образа из исходников:**

`docker build -t asv715/ml_prod:v1 .`

**Запуск:**

`docker run asv715/ml_prod:v1`

**Выполнение запросов (данные берутся из файла data/test.csv):**

`python make_request.py`

**Тесты:**

`pytest tests`