"""Metrics for tracking model predictions and prediction errors."""

import bentoml

PREDICTIONS_TOTAL = bentoml.metrics.Counter(
    name="wine_predictions_total",
    documentation="Predictions served by quality label.",
    labelnames=["quality_label"],
)

PREDICTIONS_VALUE = bentoml.metrics.Histogram(
    name="wine_prediction_value",
    documentation="Distribution of predicted quality scores.",
    buckets=(3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0),
)

PREDICTION_ERRORS_TOTAL = bentoml.metrics.Counter(
    name="wine_prediction_errors_total",
    documentation="Errors raised while serving a prediction by stage.",
    labelnames=["stage"],
)

DB_SAVE_TOTAL = bentoml.metrics.Counter(
    name="wine_prediction_db_saves_total",
    documentation="Prediction history writes to PostgreSQL by status.",
    labelnames=["status"],
)
