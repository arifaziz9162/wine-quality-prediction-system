import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from wine_quality_prediction.components import (
    DataTransformation,
    DataValidation,
    ModelEvaluation,
    ModelTrainer,
)
from wine_quality_prediction.entity import (
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
)
from wine_quality_prediction.logger.exception import DataValidationError

SCHEMA = {
    "fixed acidity": "float64",
    "volatile acidity": "float64",
    "citric acid": "float64",
    "residual sugar": "float64",
    "chlorides": "float64",
    "free sulfur dioxide": "float64",
    "total sulfur dioxide": "float64",
    "density": "float64",
    "pH": "float64",
    "sulphates": "float64",
    "alcohol": "float64",
    "quality": "int64",
}


def make_params(n_trials=2, cv=2):
    return SimpleNamespace(
        TRAINING=SimpleNamespace(
            test_size=0.2,
            random_state=42,
            development=SimpleNamespace(n_jobs=1, cv=cv),
            production=SimpleNamespace(n_jobs=1, cv=cv),
        ),
        BAYESIAN_OPTIMIZATION=SimpleNamespace(
            n_trials=n_trials, direction="maximize", scoring="r2"
        ),
        XGBOOST=SimpleNamespace(
            n_estimators=SimpleNamespace(low=10, high=20),
            max_depth=SimpleNamespace(low=2, high=4),
            learning_rate=SimpleNamespace(low=0.05, high=0.2),
            subsample=SimpleNamespace(low=0.8, high=1.0),
            colsample_bytree=SimpleNamespace(low=0.8, high=1.0),
            min_child_weight=SimpleNamespace(low=1, high=3),
            reg_alpha=SimpleNamespace(low=0.0, high=1.0),
            reg_lambda=SimpleNamespace(low=1, high=3),
        ),
    )


@pytest.fixture
def fixture_csv(tmp_path):
    rng = np.random.default_rng(7)
    n = 120
    df = pd.DataFrame(
        {
            "fixed acidity": rng.uniform(4.6, 15.9, n),
            "volatile acidity": rng.uniform(0.12, 1.58, n),
            "citric acid": rng.uniform(0.0, 1.0, n),
            "residual sugar": rng.uniform(0.9, 15.5, n),
            "chlorides": rng.uniform(0.012, 0.611, n),
            "free sulfur dioxide": rng.uniform(1, 72, n),
            "total sulfur dioxide": rng.uniform(6, 289, n),
            "density": rng.uniform(0.990, 1.004, n),
            "pH": rng.uniform(2.74, 4.01, n),
            "sulphates": rng.uniform(0.33, 2.0, n),
            "alcohol": rng.uniform(8.4, 14.9, n),
            "quality": rng.integers(3, 9, n),
        }
    )
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    return path


class TestFullPipelineE2E:
    def test_all_stages_run_and_hand_off_correctly(self, tmp_path, fixture_csv):
        # ---- Stage: data_validation ----
        validation_config = DataValidationConfig(
            root_dir=tmp_path,
            input_file=fixture_csv,
            schema=SCHEMA,
            status_file=tmp_path / "validation_status.json",
        )
        DataValidation(config=validation_config).run()

        validation_status = json.loads(validation_config.status_file.read_text())
        assert validation_status["status"] is True

        # ---- Stage: data_transformation ----
        transformation_config = DataTransformationConfig(
            root_dir=tmp_path,
            input_file=fixture_csv,
            train_file=tmp_path / "train.csv",
            test_file=tmp_path / "test.csv",
            status_file=tmp_path / "transformation_status.json",
        )
        DataTransformation(config=transformation_config, params=make_params()).run()

        transformation_status = json.loads(
            transformation_config.status_file.read_text()
        )
        assert transformation_status["status"] is True
        assert transformation_config.train_file.exists()
        assert transformation_config.test_file.exists()

        # ---- Stage: model_trainer ----
        trainer_config = ModelTrainerConfig(
            root_dir=tmp_path,
            train_file=transformation_config.train_file,
            model_file=tmp_path / "model.joblib",
            status_file=tmp_path / "trainer_status.json",
            target_column="quality",
        )
        ModelTrainer(config=trainer_config, params=make_params(n_trials=2, cv=2)).run()

        trainer_status = json.loads(trainer_config.status_file.read_text())
        assert trainer_status["status"] is True
        assert trainer_config.model_file.exists()

        # ---- Stage: model_evaluation (mocking DagsHub/MLflow network calls) ----
        evaluation_config = ModelEvaluationConfig(
            root_dir=tmp_path,
            model_file=trainer_config.model_file,
            test_file=transformation_config.test_file,
            metrics_file=tmp_path / "metrics.json",
            status_file=tmp_path / "evaluation_status.json",
            target_column="quality",
        )

        with (
            patch("wine_quality_prediction.components.model_evaluation.dagshub.init"),
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.set_experiment"
            ),
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.start_run"
            ) as mock_start_run,
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.log_metrics"
            ),
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.log_params"
            ),
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.log_artifact"
            ),
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.set_tags"
            ),
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.xgboost.log_model"
            ),
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.active_run"
            ) as mock_active_run,
        ):
            mock_active_run.return_value.info.run_id = "test-run-id"
            mock_start_run.return_value.__enter__ = lambda self: None
            mock_start_run.return_value.__exit__ = lambda *a: None

            metrics, y_pred, y_test = ModelEvaluation(config=evaluation_config).run()

        evaluation_status = json.loads(evaluation_config.status_file.read_text())
        assert evaluation_status["stage"] == "model_evaluation"
        assert evaluation_status["status"] is True
        assert evaluation_config.metrics_file.exists()
        assert set(metrics.keys()) == {"r2", "rmse", "mae"}
        assert len(y_pred) == len(y_test)

    def test_pipeline_halts_cleanly_on_bad_input_data(self, tmp_path):
        bad_df = pd.DataFrame({"fixed acidity": [7.4, 7.8], "quality": [5, 6]})
        bad_path = tmp_path / "bad.csv"
        bad_df.to_csv(bad_path, index=False)

        validation_config = DataValidationConfig(
            root_dir=tmp_path,
            input_file=bad_path,
            schema=SCHEMA,
            status_file=tmp_path / "validation_status.json",
        )
        with pytest.raises(DataValidationError):
            DataValidation(config=validation_config).run()

        status = json.loads(validation_config.status_file.read_text())
        assert status["stage"] == "data_validation"
        assert status["status"] is False
