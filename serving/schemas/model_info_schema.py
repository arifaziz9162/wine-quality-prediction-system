from pydantic import BaseModel, ConfigDict


class ModelInfoResponse(BaseModel):
    """Returns the details and evaluation metrics of the deployed model."""

    status: str
    service: str
    service_version: str
    bento_model: str
    mlflow_version: str
    mlflow_run_id: str | None = None
    r2: float | None = None
    rmse: float | None = None
    mae: float | None = None

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "status": "Running",
                "service": "wine_quality_service",
                "service_version": "2.0.0",
                "bento_model": "wine_quality_model:latest",
                "mlflow_version": "7",
                "mlflow_run_id": "3f1c9a...",
                "r2": 0.4581,
                "rmse": 0.6212,
                "mae": 0.4803,
            }
        },
    )
