import pandas as pd

from serving.config import settings
from serving.constants import FEATURE_COLUMNS
from serving.metrics import (
    PREDICTION_ERRORS_TOTAL,
    PREDICTIONS_VALUE,
    PREDICTIONS_TOTAL,
    observe_features,
)
from serving.persistence import PredictionRepository
from serving.schemas import (
    ModelInfoResponse,
    PredictionResponse,
    PredictionsResponse,
    WineInput,
)
from serving.services.model_metadata import ModelMetadata
from serving.services.quality_label import get_quality_label
from wine_quality_prediction.logger import ModelInferenceError, get_logger

logger = get_logger("prediction_service", "serving.log")


class PredictionService:
    """Handles model predictions and related serving operations."""

    def __init__(
        self,
        model,
        metadata: ModelMetadata,
        repository: PredictionRepository | None = None,
    ):
        self._model = model
        self._metadata = metadata
        self._repository = repository
        self._validate_model_features()

    def _validate_model_features(self) -> None:
        """Checks that the model uses the expected input features."""

        expected = getattr(self._model, "feature_names_in_", None)
        if expected is not None and tuple(expected) != FEATURE_COLUMNS:
            raise ModelInferenceError(
                f"Model features {tuple(expected)} do not match "
                f"Serving features {FEATURE_COLUMNS}"
            )

    def build_features(self, data: WineInput) -> pd.DataFrame:
        """Builds a DataFrame with features in the expected order."""

        values = data.model_dump()
        return pd.DataFrame(
            [[values[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS
        )

    def predict(self, data: WineInput) -> PredictionResponse:
        """Generates a prediction and records the prediction details."""

        try:
            features = self.build_features(data)
            prediction = round(float(self._model.predict(features)[0]), 2)
        except Exception as e:
            PREDICTION_ERRORS_TOTAL.labels(stage="inference").inc()
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise ModelInferenceError("Prediction failed") from e

        label = get_quality_label(prediction)
        PREDICTIONS_TOTAL.labels(quality_label=label.value).inc()
        PREDICTIONS_VALUE.observe(prediction)
        observe_features(data.model_dump())
        logger.info(f"Prediction made: '{prediction}' ({label.value})")

        if self._repository is not None:
            self._repository.save_async(features.iloc[0].tolist(), prediction)

        return PredictionResponse(
            predicted_quality=prediction,
            message=f"Predicted wine quality '{prediction}'",
            quality_label=label,
            model_version=self._metadata.distplay_version,
        )

    def model_info(self) -> ModelInfoResponse:
        """Returns information about the model currently being served."""

        meta = self._metadata
        return ModelInfoResponse(
            status="Running",
            service=settings.SERVICE_NAME,
            service_version=settings.APP_VERSION,
            bento_model=f"{meta.name}:{meta.bento_version}",
            mlflow_version=meta.mlflow_version,
            mlflow_run_id=meta.mlflow_run_id,
            r2=meta.r2,
            rmse=meta.rmse,
            mae=meta.mae,
        )

    def recent_predictions(self, limit: int) -> PredictionsResponse:
        """Returns the most recent stored predictions."""

        if self._repository is None:
            return PredictionsResponse(count=0, predictions=[])
        rows = self._repository.fetch_recent(limit)
        return PredictionsResponse(count=len(rows), predictions=rows)
