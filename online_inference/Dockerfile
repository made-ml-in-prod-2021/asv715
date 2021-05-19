FROM python:3.6
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY app/models/model.pkl /model.pkl
COPY app/inference.py /app.py

WORKDIR .

ENV PATH_TO_MODEL="/model.pkl"
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
EXPOSE 5000

CMD ["flask", "run"]