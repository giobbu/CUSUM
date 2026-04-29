from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime
import os

HOST_PATH = os.getenv("PATH_DATA")


with DAG(
    dag_id="daily_detection", 
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False
    ) as dag:

    run_scheduled_detection = DockerOperator(
                                    task_id="run_backend",
                                    image="detection-backend",
                                    auto_remove=True,
                                    docker_url="unix:///var/run/docker.sock",
                                    network_mode="bridge",
                                    mounts=[
                                    {
                                        "source": HOST_PATH,
                                        "target": "/home/data",
                                        "type": "bind"
                                    }
                                        ]
                                    )