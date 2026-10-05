# Airflow Orchestration

This folder contains the Apache Airflow setup used to orchestrate the Wine Quality Prediction pipeline.

## Version

* Apache Airflow: 3.3.1
* Python: 3.11
* Docker image: `apache/airflow:3.3.1-python3.11`

## Design

* DVC manages the ML pipeline and stage dependencies.
* Airflow handles scheduling, retries, and pipeline monitoring.
* Airflow tasks trigger individual DVC stages using `BashOperator`.
* Airflow runs in a separate Docker environment to avoid dependency conflicts with the main ML environment.

## Structure

```text
Wine-Quality-Prediction/
├── airflow/
│   ├── Dockerfile
│   ├── requirements-airflow.txt
│   ├── dags/
│   │   └── wine_quality_dag.py
│   └── README.md
│
├── docker-compose.airflow.yaml
```

## Run Airflow

From the project root:

```bash
docker compose -f docker-compose.airflow.yaml up -d
```

Open the Airflow UI:

```text
http://localhost:8080
```

To stop Airflow:

```bash
docker compose -f docker-compose.airflow.yaml down
```

## DAG

The `wine_quality_pipeline` DAG runs the following stages in order:

```text
Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Registry
```

The DAG is paused when created. Enable or trigger it from the Airflow UI.

## Notes

Airflow is used as the orchestration layer, while DVC manages the ML pipeline stages and dependencies. The pipeline logic remains inside the project's `src/` package.
