from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator

PROJECT_DIR = Variable.get(
    "wine_quality_project_dir", default_var="/opt/wine_quality_prediction"
)

default_args = {
    "owner": "arif",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="wine_quality_pipeline",
    default_args=default_args,
    description="End-to-end wine quality prediction pipeline (DVC-orchestrated)",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "wine-quality", "dvc"],
) as dag:
    data_ingestion = BashOperator(
        task_id="data_ingestion",
        bash_command=f"cd {PROJECT_DIR} && dvc repro data_ingestion",
    )

    data_validation = BashOperator(
        task_id="data_validation",
        bash_command=f"cd {PROJECT_DIR} && dvc repro data_validation",
    )

    data_transformation = BashOperator(
        task_id="data_transformation",
        bash_command=f"cd {PROJECT_DIR} && dvc repro data_transformation",
    )

    model_trainer = BashOperator(
        task_id="model_trainer",
        bash_command=f"cd {PROJECT_DIR} && dvc repro model_trainer",
    )

    model_evaluation = BashOperator(
        task_id="model_evaluation",
        bash_command=f"cd {PROJECT_DIR} && dvc repro model_evaluation",
    )

    model_registry = BashOperator(
        task_id="model_registry",
        bash_command=f"cd {PROJECT_DIR} && dvc repro model_registry",
    )

    model_packaging = BashOperator(
        task_id="model_packaging",
        bash_command=f"cd {PROJECT_DIR} && dvc repro model_packaging",
    )

    (
        data_ingestion
        >> data_validation
        >> data_transformation
        >> model_trainer
        >> model_evaluation
        >> model_registry
        >> model_packaging
    )
