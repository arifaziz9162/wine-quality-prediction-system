from pydantic import BaseModel, Field


class PredictionsRequest(BaseModel):
    """Parameters used to fetch stored predictions."""

    limit: int = Field(100, ge=1, le=500)


class PredictionsResponse(BaseModel):
    """Contains the stored predictions and their total counts."""

    count: int
    predictions: list[dict]
