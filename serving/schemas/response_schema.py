from enum import Enum

from pydantic import BaseModel, ConfigDict


class QualityLabel(str, Enum):
    bad = "Bad"
    average = "Average"
    good = "Good"


class PredictionResponse(BaseModel):
    """Response schema for the prediction endpoint."""

    predicted_quality: float
    message: str
    quality_label: QualityLabel
    model_version: str

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "predicted_quality": 5.61,
                "message": "Predicted wine quality '5.61'",
                "quality_label": "Average",
                "model_version": "7",
            }
        },
    )
