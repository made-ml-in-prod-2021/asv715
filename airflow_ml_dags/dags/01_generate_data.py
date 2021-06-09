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
    "01_generate_data",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(5),
) as dag:
    download = DockerOperator(
        image="airflow-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-generate-data",
        do_xcom_push=False,
        volumes=["/home/sergey/development/airflow/data:/data"]
    )