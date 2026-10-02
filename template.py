import logging
import os
from pathlib import Path


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)


# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------

project_name = "wine_quality_prediction"


# -----------------------------------------------------------------------------
# Source package
# -----------------------------------------------------------------------------

src_folders = [
    f"src/{project_name}",
    f"src/{project_name}/config",
    f"src/{project_name}/constants",
    f"src/{project_name}/entity",
    f"src/{project_name}/logger",
    f"src/{project_name}/utils",
    f"src/{project_name}/database",
    f"src/{project_name}/components",
    f"src/{project_name}/pipeline",
]


# -----------------------------------------------------------------------------
# BentoML serving package
# -----------------------------------------------------------------------------

serving_folders = [
    "serving",
    "serving/schemas",
    "serving/services",
    "serving/persistence",
    "serving/metrics",
]


# -----------------------------------------------------------------------------
# Other project directories
# -----------------------------------------------------------------------------

other_folders = [
    "notebooks",

    # Configuration
    "config",

    # Tests
    "tests",
    "tests/unit",
    "tests/integration",
    "tests/serving",

    # Airflow
    "airflow",
    "airflow/dags",

    # Documentation
    "docs",
    "docs/images",

    # Deployment
    "deployment",
    "deployment/k8s",
    "deployment/k8s/base",

    # Docker
    "docker",

    # Monitoring
    "monitoring/prometheus",
    "monitoring/grafana/dashboards",
    "monitoring/grafana/provisioning/datasources",
    "monitoring/grafana/provisioning/dashboards",
]


# -----------------------------------------------------------------------------
# Python package __init__.py files
# -----------------------------------------------------------------------------

list_of_files = [
    f"{folder}/__init__.py"
    for folder in src_folders
]

list_of_files += [
    f"{folder}/__init__.py"
    for folder in serving_folders
]


# -----------------------------------------------------------------------------
# Application files
# -----------------------------------------------------------------------------

extra_files = [

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    f"src/{project_name}/config/configuration.py",

    "config/config.yaml",
    "config/schema.yaml",
    "config/params.yaml",


    # -------------------------------------------------------------------------
    # Entity
    # -------------------------------------------------------------------------

    f"src/{project_name}/entity/config_entity.py",


    # -------------------------------------------------------------------------
    # Logger
    # -------------------------------------------------------------------------

    f"src/{project_name}/logger/logger_config.py",
    f"src/{project_name}/logger/exception.py",


    # -------------------------------------------------------------------------
    # Utils
    # -------------------------------------------------------------------------

    f"src/{project_name}/utils/common.py",


    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------

    f"src/{project_name}/database/connection.py",
    f"src/{project_name}/database/operations.py",


    # -------------------------------------------------------------------------
    # Components
    # -------------------------------------------------------------------------

    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/model_registry.py",
    f"src/{project_name}/components/model_packaging.py",


    # -------------------------------------------------------------------------
    # Pipeline stages
    # -------------------------------------------------------------------------

    f"src/{project_name}/pipeline/stage_01_data_ingestion.py",
    f"src/{project_name}/pipeline/stage_02_data_validation.py",
    f"src/{project_name}/pipeline/stage_03_data_transformation.py",
    f"src/{project_name}/pipeline/stage_04_model_trainer.py",
    f"src/{project_name}/pipeline/stage_05_model_evaluation.py",
    f"src/{project_name}/pipeline/stage_06_model_registry.py",
    f"src/{project_name}/pipeline/stage_07_model_packaging.py",


    # -------------------------------------------------------------------------
    # Root files
    # -------------------------------------------------------------------------

    "requirements-dev.txt",
    "requirements.txt",
    "pyproject.toml",
    "main.py",
    "dvc.yaml",
    ".dvcignore",
    "Makefile",

    "docker-compose.airflow.yml",
    "docker-compose.dev.yml",
    "docker-compose.prod.yml",


    # -------------------------------------------------------------------------
    # Notebooks
    # -------------------------------------------------------------------------

    "notebooks/EDA.ipynb",
    "notebooks/Experiments.ipynb",


    # -------------------------------------------------------------------------
    # Tests
    # -------------------------------------------------------------------------

    "tests/unit/test_data_ingestion.py",
    "tests/unit/test_data_validation.py",
    "tests/unit/test_data_transformation.py",
    "tests/unit/test_model_trainer.py",
    "tests/unit/test_model_evaluation.py",
    "tests/unit/test_model_registry.py",
    "tests/unit/test_model_packaging.py",

    "tests/integration/test_pipeline_e2e.py",

    "tests/serving/test_quality_label.py",
    "tests/serving/test_schemas.py",
    "tests/serving/test_prediction_service.py",
    "tests/serving/test_prediction_repository.py",


    # -------------------------------------------------------------------------
    # Airflow
    # -------------------------------------------------------------------------

    "airflow/Dockerfile",
    "airflow/requirements-airflow.txt",
    "airflow/README.md",
    "airflow/dags/wine_quality_dag.py",


    # -------------------------------------------------------------------------
    # BentoML serving
    # -------------------------------------------------------------------------

    "serving/service.py",
    "serving/config.py",
    "serving/constants.py",
    "serving/requirements.txt",

    "serving/schemas/request_schema.py",
    "serving/schemas/response_schema.py",
    "serving/schemas/model_info_schema.py",
    "serving/schemas/predictions_schema.py",

    "serving/services/prediction_service.py",
    "serving/services/model_metadata.py",
    "serving/services/quality_label.py",

    "serving/persistence/prediction_repository.py",

    "serving/metrics/prediction_metrics.py",
    "serving/metrics/model_metrics.py",
    "serving/metrics/feature_metrics.py",


    # -------------------------------------------------------------------------
    # Docker
    # -------------------------------------------------------------------------

    "docker/Dockerfile.migrate",


    # -------------------------------------------------------------------------
    # Prometheus
    # -------------------------------------------------------------------------

    "monitoring/prometheus/prometheus.compose.yml",
    "monitoring/prometheus/prometheus.k8s.yml",
    "monitoring/prometheus/alert_rules.yml",


    # -------------------------------------------------------------------------
    # Grafana
    # -------------------------------------------------------------------------

    "monitoring/grafana/dashboards/wine-quality-overview.json",

    "monitoring/grafana/provisioning/datasources/prometheus.yml",
    "monitoring/grafana/provisioning/dashboards/dashboards.yml",


    # -------------------------------------------------------------------------
    # Kubernetes
    # -------------------------------------------------------------------------

    "deployment/k8s/base/namespace.yaml",
    "deployment/k8s/base/postgres.yaml",
    "deployment/k8s/base/migrate-job.yaml",
    "deployment/k8s/base/bento-deployment.yaml",
    "deployment/k8s/base/prometheus.yaml",
    "deployment/k8s/base/grafana.yaml",
    "deployment/k8s/base/kustomization.yaml",
    "deployment/k8s/base/secret.env.example",


    # -------------------------------------------------------------------------
    # GitHub CI/CD
    # -------------------------------------------------------------------------

    ".github/dependabot.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/docker.yml",
    ".github/workflows/model-training.yml",
    ".github/workflows/security.yml",
    ".github/workflows/deploy.yml",
    ".github/workflows/release.yml",
]


# -----------------------------------------------------------------------------
# Create directories
# -----------------------------------------------------------------------------

for folder in other_folders:
    os.makedirs(folder, exist_ok=True)
    logging.info(f"Creating folder: '{folder}'")


for folder in serving_folders:
    os.makedirs(folder, exist_ok=True)
    logging.info(f"Creating folder: '{folder}'")


# -----------------------------------------------------------------------------
# Create files
# -----------------------------------------------------------------------------

for filepath in list_of_files + extra_files:
    filepath = Path(filepath)

    if filepath.parent != Path("."):
        filepath.parent.mkdir(parents=True, exist_ok=True)

    if not filepath.exists():
        filepath.touch()
        logging.info(f"Creating empty file: '{filepath}'")

    elif filepath.stat().st_size == 0:
        logging.info(f"File already exists and is empty: '{filepath}'")

    else:
        logging.info(f"File already exists: '{filepath}'")


logging.info("Project structure created successfully.")
