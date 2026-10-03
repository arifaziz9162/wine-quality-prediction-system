from pydantic import BaseModel, ConfigDict, Field


class WineInput(BaseModel):
    """Input Schema for wine quality prediction."""

    fixed_acidity: float = Field(..., gt=0)
    volatile_acidity: float = Field(..., gt=0)
    citric_acid: float = Field(..., ge=0)
    residual_sugar: float = Field(..., gt=0)
    chlorides: float = Field(..., gt=0)
    free_sulfur_dioxide: float = Field(..., gt=0)
    total_sulfur_dioxide: float = Field(..., gt=0)
    density: float = Field(..., gt=0)
    ph: float = Field(..., gt=0)
    sulphates: float = Field(..., gt=0)
    alcohol: float = Field(..., gt=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "ph": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
            }
        }
    )
