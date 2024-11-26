from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 11, 11),
}

dag = DAG(
    "weather_data_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
)

collect_data = BashOperator(
    task_id="collect_data",
    bash_command="python /opt/airflow/data_collection.py",
    dag=dag,
)

preprocess_data = BashOperator(
    task_id="preprocess_data",
    bash_command="python /opt/airflow/preprocessing.py",
    dag=dag,
)

training_model = BashOperator(
    task_id="training_model",
    bash_command="python /opt/airflow/train_model.py",
    dag=dag,
)

collect_data >> preprocess_data >> training_model