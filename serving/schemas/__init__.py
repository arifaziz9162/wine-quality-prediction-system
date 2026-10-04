from .model_info_schema import ModelInfoResponse
from .predictions_schema import PredictionsRequest, PredictionsResponse
from .request_schema import WineInput
from .response_schema import PredictionResponse, QualityLabel

__all__ = [
    "ModelInfoResponse",
    "PredictionsRequest",
    "PredictionsResponse",
    "WineInput",
    "PredictionResponse",
    "QualityLabel",
]
