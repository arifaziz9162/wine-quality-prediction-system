from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from serving.constants import FEATURE_COLUMNS
from serving.metrics import PREDICTION_ERRORS_TOTAL, PREDICTIONS_TOTAL
from serving.schemas import QualityLabel, WineInput
from serving.services import ModelMetadata, PredictionService
from wine_quality_prediction.logger import ModelInferenceError

VALID = WineInput.model_config["json_schema_extra"]["example"]


class FakeModel:
    """Stands in for the pickled XGBRegressor."""

    def __init__(self, value=5.6, feature_names=FEATURE_COLUMNS):
        self.value = value
        self.feature_names_in_ = np.array(feature_names)
        self.last_frame = None

    def predict(self, frame):
        self.last_frame = frame
        return np.array([self.value])


@pytest.fixture
def metadata():
    return ModelMetadata(
        name="wine_quality_model",
        bento_version="abc123",
        mlflow_version="7",
        mlflow_run_id="run-7",
        r2=0.4581,
        rmse=0.62,
        mae=0.48,
    )


def counter_value(counter, **labels):
    return counter.labels(**labels)._value.get()


class TestBuildFeatures:
    def test_columns_follow_training_order(self, metadata):
        service = PredictionService(FakeModel(), metadata)
        frame = service.build_features(WineInput(**VALID))

        assert tuple(frame.columns) == FEATURE_COLUMNS
        assert frame.shape == (1, 11)
        assert frame.loc[0, "ph"] == 3.51
        assert frame.loc[0, "alcohol"] == 9.4


class TestPredict:
    def test_returns_rounded_prediction_label_and_version(self, metadata):
        service = PredictionService(FakeModel(value=6.34567), metadata)
        response = service.predict(WineInput(**VALID))

        assert response.predicted_quality == 6.35
        assert response.quality_label == QualityLabel.average
        assert response.model_version == "7"

    def test_falls_back_to_bento_version_without_mlflow_label(self, metadata):
        metadata = ModelMetadata(**{**metadata.__dict__, "mlflow_version": None})
        response = PredictionService(FakeModel(), metadata).predict(WineInput(**VALID))

        assert response.model_version == "abc123"

    def test_increments_prediction_counter(self, metadata):
        service = PredictionService(FakeModel(value=7.5), metadata)
        before = counter_value(PREDICTIONS_TOTAL, quality_label="Good")

        service.predict(WineInput(**VALID))

        assert counter_value(PREDICTIONS_TOTAL, quality_label="Good") == before + 1

    def test_saves_to_repository_without_blocking(self, metadata):
        repository = MagicMock()
        service = PredictionService(FakeModel(value=5.0), metadata, repository)

        service.predict(WineInput(**VALID))

        repository.save_async.assert_called_once()
        features, prediction = repository.save_async.call_args.args
        assert prediction == 5.0
        assert len(features) == 11

    def test_repository_failure_does_not_break_the_response(self, metadata):
        repository = MagicMock()
        repository.save_async.return_value = None
        service = PredictionService(FakeModel(), metadata, repository)

        assert service.predict(WineInput(**VALID)).predicted_quality == 5.6

    def test_model_failure_raises_and_counts_error(self, metadata):
        model = FakeModel()
        model.predict = MagicMock(side_effect=RuntimeError("boom"))
        service = PredictionService(model, metadata)
        before = counter_value(PREDICTION_ERRORS_TOTAL, stage="inference")

        with pytest.raises(ModelInferenceError):
            service.predict(WineInput(**VALID))

        assert counter_value(PREDICTION_ERRORS_TOTAL, stage="inference") == before + 1


class TestStartupValidation:
    def test_rejects_model_trained_on_different_columns(self, metadata):
        wrong = FakeModel(feature_names=("a", "b", "c"))

        with pytest.raises(ModelInferenceError):
            PredictionService(wrong, metadata)

    def test_accepts_model_without_feature_names(self, metadata):
        model = SimpleNamespace(predict=lambda frame: np.array([5.0]))

        assert PredictionService(model, metadata) is not None


class TestModelInfo:
    def test_reports_mlflow_provenance(self, metadata):
        info = PredictionService(FakeModel(), metadata).model_info()

        assert info.status == "Running"
        assert info.bento_model == "wine_quality_model:abc123"
        assert info.mlflow_version == "7"
        assert info.r2 == 0.4581


class TestRecentPredictions:
    def test_returns_empty_when_history_disabled(self, metadata):
        result = PredictionService(FakeModel(), metadata).recent_predictions(10)

        assert result.count == 0
        assert result.predictions == []

    def test_returns_rows_from_repository(self, metadata):
        repository = MagicMock()
        repository.fetch_recent.return_value = [{"id": 1}, {"id": 2}]
        service = PredictionService(FakeModel(), metadata, repository)

        result = service.recent_predictions(10)

        assert result.count == 2
        repository.fetch_recent.assert_called_once_with(10)


class TestModelMetadata:
    def test_reads_labels_and_metadata_from_bento_model(self):
        bento_model = SimpleNamespace(
            tag=SimpleNamespace(name="wine_quality_model", version="v1"),
            info=SimpleNamespace(
                labels={"mlflow_version": "3"},
                metadata={"r2": 0.5, "rmse": 0.6, "mae": 0.4, "mlflow_run_id": "r"},
            ),
        )

        meta = ModelMetadata.from_bento_model(bento_model)

        assert meta.mlflow_version == "3"
        assert meta.r2 == 0.5
        assert meta.mlflow_run_id == "r"
