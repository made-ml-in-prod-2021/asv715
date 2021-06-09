import os
from datetime import timedelta
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
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
    "02_fit_model",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(5),
) as dag:
    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="/data/raw/{{ ds }} /data/processed/{{ ds }}",
        network_mode="bridge",
        task_id="docker-preprocess-data",
        do_xcom_push=False,
        volumes=["/home/sergey/development/airflow/data:/data"]
    )

    wait_for_preprocess = FileSensor(
        task_id="wait_for_preprocess",
        filepath="/opt/airflow/data/processed/{{ ds }}/train_data.csv",
        timeout=6000,
        poke_interval=10,
        retries=100,
    )

    split = DockerOperator(
        image="airflow-split",
        command="/data/processed/{{ ds }} /data/splitted/{{ ds }}",
        network_mode="bridge",
        task_id="docker-split-data",
        do_xcom_push=False,
        volumes=["/home/sergey/development/airflow/data:/data"]
    )

    wait_for_split = FileSensor(
        task_id="wait_for_split",
        filepath="/opt/airflow/data/splitted/{{ ds }}/train_data.csv",
        timeout=6000,
        poke_interval=10,
        retries=100,
    )

    fit = DockerOperator(
        image="airflow-fit",
        command="/data/splitted/{{ ds }} /data/models/{{ ds }}",
        network_mode="bridge",
        task_id="docker-fit-model",
        do_xcom_push=False,
        volumes=["/home/sergey/development/airflow/data:/data"]
    )

    wait_for_fit = FileSensor(
        task_id="wait_for_fit",
        filepath="/opt/airflow/data/models/{{ ds }}/model.pkl",
        timeout=6000,
        poke_interval=10,
        retries=100,
    )

    validate = DockerOperator(
        image="airflow-validate",
        command="/data/splitted/{{ ds }} /data/models/{{ ds }} /data/metrics/{{ ds }}",
        network_mode="bridge",
        task_id="docker-validate-model",
        do_xcom_push=False,
        volumes=["/home/sergey/development/airflow/data:/data"]
    )

    preprocess >> wait_for_preprocess >> split >> wait_for_split >> fit >> wait_for_fit >> validate
