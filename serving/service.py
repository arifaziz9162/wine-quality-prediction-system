import bentoml
import joblib

from serving.config import settings
from serving.metrics import publish_model_metrics
from serving.persistence import PredictionRepository
from serving.schemas import (
    ModelInfoResponse,
    PredictionResponse,
    PredictionsRequest,
    PredictionsResponse,
    WineInput
)
from serving.services import ModelMetadata, PredictionService
from wine_quality_prediction.logger import get_logger

logger = get_logger("service", "service.log")


@bentoml.service(
    name=settings.SERVICE_NAME,
    traffic={
        "timeout": settings.REQUEST_TIMEOUT_SECONDS,
        "concurrency": settings.MAX_CONCURRENCY
    }
)
class WineQualityService:
    """Serves the wine quality model."""

    bento_model = bentoml.models.BentoModel(settings.BENTO_MODEL_TAG)

    def __init__(self) -> None:
        """Loads the model and initializes the prediction service."""

        model = joblib.load(
            self.bento_model.path_of(settings.MODEL_FILE_NAME)
        )

        metadata = ModelMetadata.from_bento_model(self.bento_model)

        repository = (
            PredictionRepository()
            if settings.DB_SAVE_ENABLED
            else None
        )

        self.prediction_service = PredictionService(
            model=model,
            metadata=metadata,
            repository=repository
        )

        publish_model_metrics(
            bento_version=metadata.bento_version,
            mlflow_version=metadata.mlflow_version,
            r2=metadata.r2
        )

        logger.info(
            f"Serving '{metadata.name}' "
            f"version '{metadata.distplay_version}'"
        )

    @bentoml.api(
        route="/api/v1/predict",
        input_spec=WineInput,
        output_spec=PredictionResponse
    )
    def predict(self, **params) -> PredictionResponse:
        """Predicts wine quality from input features."""

        return self.prediction_service.predict(
            WineInput(**params)
        )

    @bentoml.api(
        route="/api/v1/model_info",
        output_spec=ModelInfoResponse
    )
    def model_info(self) -> ModelInfoResponse:
        """Returns information about the model being served."""

        return self.prediction_service.model_info()

    @bentoml.api(
        route="/api/v1/predictions",
        input_spec=PredictionsRequest,
        output_spec=PredictionsResponse
    )
    def stored_predictions(self, **params) -> PredictionsResponse:
        """Returns the most recent stored predictions."""

        request = PredictionsRequest(**params)

        return self.prediction_service.recent_predictions(
            request.limit
        )
