"""Metrics for tracking the model being served."""

import bentoml

MODEL_INFO = bentoml.metrics.Gauge(
    name="wine_model_info",
    documentation="Constant 1, labelled with the model versions being served.",
    labelnames=["bento_version", "mlflow_version"],
    multiprocess_mode="max",
)

MODEL_R2 = bentoml.metrics.Gauge(
    name="wine_model_r2",
    documentation="Test R2 of the served model, recorded at promotion time.",
    multiprocess_mode="max",
)


def publish_model_metrics(
    bento_version: str, mlflow_version: str | None, r2: float | None
) -> None:
    """Records the model version and R2 score in Prometheus."""

    MODEL_INFO.labels(
        bento_version=bento_version, mlflow_version=mlflow_version or "unknown"
    ).set(1)
    if r2 is not None:
        MODEL_R2.set(r2)
