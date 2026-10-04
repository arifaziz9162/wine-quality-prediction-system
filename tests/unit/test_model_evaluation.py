import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from wine_quality_prediction.components import ModelEvaluation
from wine_quality_prediction.entity import ModelEvaluationConfig
from wine_quality_prediction.logger import ModelEvaluationError
from wine_quality_prediction.utils import save_bin


@pytest.fixture
def evaluation_config(tmp_path):
    status_dir = tmp_path / "model_evaluation"
    status_dir.mkdir(parents=True, exist_ok=True)
    return ModelEvaluationConfig(
        root_dir=tmp_path,
        model_file=tmp_path / "model.joblib",
        test_file=tmp_path / "test.csv",
        metrics_file=tmp_path / "metrics.json",
        status_file=status_dir / "status.json",
        target_column="quality",
    )


@pytest.fixture
def test_df():
    rng = np.random.default_rng(42)
    n = 30
    return pd.DataFrame(
        {
            "fixed_acidity": rng.uniform(6, 9, n),
            "volatile_acidity": rng.uniform(0.3, 0.8, n),
            "alcohol": rng.uniform(9, 13, n),
            "quality": rng.integers(4, 8, n),
        }
    )


@pytest.fixture
def trained_model(test_df):
    x_train = test_df.drop(columns=["quality"])
    y_train = test_df["quality"]
    model = XGBRegressor(objective="reg:squarederror", n_jobs=1, random_state=42)

    model.fit(x_train, y_train)

    return model


@pytest.fixture
def mock_dagshub_mlflow():
    with (
        patch("wine_quality_prediction.components.model_evaluation.dagshub.init"),
        patch(
            "wine_quality_prediction.components.model_evaluation.mlflow.set_experiment"
        ),
    ):
        yield


class TestInit:
    def test_target_column_read_from_schema(
        self, evaluation_config, mock_dagshub_mlflow
    ):
        evaluation = ModelEvaluation(config=evaluation_config)

        assert evaluation.target_column == "quality"

    def test_target_column_reflects_config(
        self, evaluation_config, mock_dagshub_mlflow
    ):
        config = ModelEvaluationConfig(
            root_dir=evaluation_config.root_dir,
            model_file=evaluation_config.model_file,
            test_file=evaluation_config.test_file,
            metrics_file=evaluation_config.metrics_file,
            status_file=evaluation_config.status_file,
            target_column="rating",
        )
        evaluation = ModelEvaluation(config=config)

        assert evaluation.target_column == "rating"


class TestLoadModel:
    def test_loads_model_successfully(
        self, evaluation_config, trained_model, mock_dagshub_mlflow
    ):
        save_bin(trained_model, evaluation_config.model_file)
        evaluation = ModelEvaluation(config=evaluation_config)

        model = evaluation.load_model()

        assert hasattr(model, "predict")

    def test_raises_on_missing_model_file(self, evaluation_config, mock_dagshub_mlflow):
        evaluation = ModelEvaluation(config=evaluation_config)

        with pytest.raises(ModelEvaluationError):
            evaluation.load_model()


class TestLoadTestData:
    def test_loads_test_csv_successfully(
        self, evaluation_config, test_df, mock_dagshub_mlflow
    ):
        test_df.to_csv(evaluation_config.test_file, index=False)
        evaluation = ModelEvaluation(config=evaluation_config)

        loaded = evaluation.load_test_data()

        assert loaded.shape == test_df.shape

    def test_raises_on_missing_test_file(self, evaluation_config, mock_dagshub_mlflow):
        evaluation = ModelEvaluation(config=evaluation_config)

        with pytest.raises(ModelEvaluationError):
            evaluation.load_test_data()


class TestPrepareTestData:
    def test_splits_features_and_target(
        self, evaluation_config, test_df, mock_dagshub_mlflow
    ):
        evaluation = ModelEvaluation(config=evaluation_config)
        x_test, y_test = evaluation.prepare_test_data(test_df)

        assert "quality" not in x_test.columns
        assert y_test.name == "quality"
        assert len(x_test) == len(y_test) == len(test_df)

    def test_raises_if_target_column_missing(
        self, evaluation_config, mock_dagshub_mlflow
    ):
        evaluation = ModelEvaluation(config=evaluation_config)
        df = pd.DataFrame({"fixed acidity": [1, 2, 3]})

        with pytest.raises(ModelEvaluationError):
            evaluation.prepare_test_data(df)


class TestEvaluate:
    def test_computes_correct_metrics_for_perfect_predictions(
        self, evaluation_config, mock_dagshub_mlflow
    ):
        evaluation = ModelEvaluation(config=evaluation_config)

        y_test = pd.Series([5.0, 6.0, 7.0, 5.0])
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([5.0, 6.0, 7.0, 5.0])
        x_test = pd.DataFrame({"a": [1, 2, 3, 4]})

        metrics, y_pred = evaluation.evaluate(mock_model, x_test, y_test)

        assert metrics["r2"] == 1.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0

    def test_metrics_match_manual_calculation(
        self, evaluation_config, mock_dagshub_mlflow
    ):
        evaluation = ModelEvaluation(config=evaluation_config)

        y_test = pd.Series([3.0, 5.0, 7.0])
        y_pred_values = np.array([2.0, 5.0, 9.0])
        mock_model = MagicMock()
        mock_model.predict.return_value = y_pred_values
        x_test = pd.DataFrame({"a": [1, 2, 3]})

        metrics, y_pred = evaluation.evaluate(mock_model, x_test, y_test)

        ss_res = np.sum((y_test.values - y_pred_values) ** 2)
        ss_tot = np.sum((y_test.values - y_test.values.mean()) ** 2)
        expected_r2 = 1 - (ss_res / ss_tot)

        expected_rmse = np.sqrt(np.mean((y_test.values - y_pred_values) ** 2))
        expected_mae = np.mean(np.abs(y_test.values - y_pred_values))

        assert metrics["r2"] == pytest.approx(expected_r2, abs=1e-4)
        assert metrics["rmse"] == pytest.approx(expected_rmse, abs=1e-4)
        assert metrics["mae"] == pytest.approx(expected_mae, abs=1e-4)

    def test_saves_metrics_to_json_file(
        self, evaluation_config, test_df, trained_model, mock_dagshub_mlflow
    ):
        evaluation = ModelEvaluation(config=evaluation_config)
        x_test = test_df.drop(columns=["quality"])
        y_test = test_df["quality"]

        evaluation.evaluate(trained_model, x_test, y_test)

        assert evaluation_config.metrics_file.exists()
        saved_metrics = json.loads(evaluation_config.metrics_file.read_text())
        assert set(saved_metrics.keys()) == {"r2", "rmse", "mae"}

    def test_returns_predictions_matching_test_set_length(
        self, evaluation_config, test_df, trained_model, mock_dagshub_mlflow
    ):
        evaluation = ModelEvaluation(config=evaluation_config)
        x_test = test_df.drop(columns=["quality"])
        y_test = test_df["quality"]

        _, y_pred = evaluation.evaluate(trained_model, x_test, y_test)

        assert len(y_pred) == len(y_test)


class TestLogToMlflow:
    def test_logs_params_metrics_and_model(
        self, evaluation_config, trained_model, test_df, mock_dagshub_mlflow
    ):
        evaluation = ModelEvaluation(config=evaluation_config)
        metrics = {"r2": 0.5, "rmse": 0.6, "mae": 0.47}

        with (
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.start_run"
            ) as mock_start_run,
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.set_tags"
            ) as mock_set_tags,
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.log_params"
            ) as mock_log_params,
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.log_metrics"
            ) as mock_log_metrics,
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.xgboost.log_model"
            ) as mock_log_model,
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.log_artifact"
            ),
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.active_run"
            ) as mock_active_run,
        ):
            mock_active_run.return_value.info.run_id = "test-run-id"
            mock_start_run.return_value.__enter__ = lambda self: None
            mock_start_run.return_value.__exit__ = lambda *a: None

            x_test, _ = evaluation.prepare_test_data(test_df)

            evaluation.log_to_mlflow(trained_model, metrics, x_test)

            mock_set_tags.assert_called_once()
            mock_log_params.assert_called_once()
            mock_log_metrics.assert_called_once_with(metrics)
            mock_log_model.assert_called_once()

    def test_raises_wrapped_error_on_mlflow_failure(
        self, evaluation_config, trained_model, test_df, mock_dagshub_mlflow
    ):
        evaluation = ModelEvaluation(config=evaluation_config)
        x_test, y_test = evaluation.prepare_test_data(test_df)
        metrics, _ = evaluation.evaluate(trained_model, x_test, y_test)

        with patch(
            "wine_quality_prediction.components.model_evaluation.mlflow.start_run",
            side_effect=Exception("Mlflow failure"),
        ):
            with pytest.raises(ModelEvaluationError):
                evaluation.log_to_mlflow(trained_model, metrics, x_test)


class TestRun:
    def test_run_end_to_end_writes_success_status(
        self, evaluation_config, test_df, trained_model
    ):
        save_bin(trained_model, evaluation_config.model_file)
        test_df.to_csv(evaluation_config.test_file, index=False)

        with (
            patch("wine_quality_prediction.components.model_evaluation.dagshub.init"),
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.set_experiment"
            ),
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.start_run"
            ) as mock_start_run,
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.set_tags"
            ),
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.log_params"
            ),
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.log_metrics"
            ),
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.xgboost.log_model"
            ),
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.log_artifact"
            ),
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.active_run"
            ) as mock_active_run,
        ):
            mock_active_run.return_value.info.run_id = "test-run-id"
            mock_start_run.return_value.__enter__ = lambda self: None
            mock_start_run.return_value.__exit__ = lambda *a: None

            evaluation = ModelEvaluation(config=evaluation_config)
            metrics, y_pred, y_test = evaluation.run()

        status = json.loads(evaluation_config.status_file.read_text())
        assert status["stage"] == "model_evaluation"
        assert status["status"] is True
        assert set(metrics.keys()) == {"r2", "rmse", "mae"}
        assert len(y_pred) == len(y_test)

    def test_run_writes_failure_status_on_missing_model_file(
        self, evaluation_config, test_df
    ):
        test_df.to_csv(evaluation_config.test_file, index=False)

        with (
            patch("wine_quality_prediction.components.model_evaluation.dagshub.init"),
            patch(
                "wine_quality_prediction.components.model_evaluation.mlflow.set_experiment"
            ),
        ):
            evaluation = ModelEvaluation(config=evaluation_config)
            with pytest.raises(ModelEvaluationError):
                evaluation.run()

        status = json.loads(evaluation_config.status_file.read_text())
        assert status["stage"] == "model_evaluation"
        assert status["status"] is False
