import pytest
from pydantic import ValidationError

from serving.schemas import PredictionsRequest, WineInput

VALID = WineInput.model_config["json_schema_extra"]["example"]


def test_valid_payload_is_accepted():
    assert WineInput(**VALID).alcohol == 9.4


def test_missing_field_is_rejected():
    payload = {k: v for k, v in VALID.items() if k != "alcohol"}
    with pytest.raises(ValidationError):
        WineInput(**payload)


@pytest.mark.parametrize("field", ["fixed_acidity", "density", "alcohol"])
def test_non_positive_values_are_rejected(field):
    with pytest.raises(ValidationError):
        WineInput(**{**VALID, field: -1.0})


def test_citric_acid_may_be_zero():
    assert WineInput(**{**VALID, "citric_acid": 0.0}).citric_acid == 0.0


@pytest.mark.parametrize("limit", [0, -5, 501])
def test_predictions_limit_bounds(limit):
    with pytest.raises(ValidationError):
        PredictionsRequest(limit=limit)


def test_predictions_limit_defaults_to_100():
    assert PredictionsRequest().limit == 100
