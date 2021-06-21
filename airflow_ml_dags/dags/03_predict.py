import os
from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["asv715@yandex.ru"],
    "email_on_retry": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "03_predict",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(5),
) as dag:
    predict = DockerOperator(
        image="airflow-predict",
        command="/data/processed/{{ ds }} {{ var.value.model_path }} /data/predictions/{{ ds }}",
        network_mode="bridge",
        task_id="docker-predict",
        do_xcom_push=False,
        volumes=["/home/sergey/development/airflow/data:/data"]
    )

