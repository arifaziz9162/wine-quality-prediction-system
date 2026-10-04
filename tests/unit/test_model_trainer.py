import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wine_quality_prediction.components import ModelTrainer
from wine_quality_prediction.entity import ModelTrainerConfig
from wine_quality_prediction.logger import ModelTrainingError


def make_params(n_trials=3, cv=2, n_jobs=1):
    return SimpleNamespace(
        TRAINING=SimpleNamespace(
            random_state=42,
            development=SimpleNamespace(n_jobs=n_jobs, cv=cv),
            production=SimpleNamespace(n_jobs=n_jobs, cv=cv),
        ),
        BAYESIAN_OPTIMIZATION=SimpleNamespace(
            n_trials=n_trials, direction="maximize", scoring="r2"
        ),
        XGBOOST=SimpleNamespace(
            n_estimators=SimpleNamespace(low=50, high=100),
            max_depth=SimpleNamespace(low=2, high=5),
            learning_rate=SimpleNamespace(low=0.01, high=0.1),
            subsample=SimpleNamespace(low=0.6, high=0.9),
            colsample_bytree=SimpleNamespace(low=0.6, high=0.9),
            min_child_weight=SimpleNamespace(low=1, high=5),
            reg_alpha=SimpleNamespace(low=0.0, high=1.0),
            reg_lambda=SimpleNamespace(low=1, high=5),
        ),
    )


@pytest.fixture
def trainer_config(tmp_path):
    status_dir = tmp_path / "model_trainer"
    status_dir.mkdir(parents=True, exist_ok=True)
    return ModelTrainerConfig(
        root_dir=tmp_path,
        train_file=tmp_path / "train.csv",
        model_file=tmp_path / "model.joblib",
        status_file=status_dir / "status.json",
        target_column="quality",
    )


@pytest.fixture
def train_df():
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame(
        {
            "fixed acidity": rng.uniform(6, 9, n),
            "volatile acidity": rng.uniform(0.3, 0.8, n),
            "alcohol": rng.uniform(9, 13, n),
            "quality": rng.integers(4, 8, n),
        }
    )


class TestInit:
    def test_target_column_read_from_config(self, trainer_config):
        trainer = ModelTrainer(config=trainer_config, params=make_params())

        assert trainer.target_column == "quality"


class TestPrepareTrainData:
    def test_splits_features_and_target(self, trainer_config, train_df):
        trainer = ModelTrainer(config=trainer_config, params=make_params())
        x_train, y_train = trainer.prepare_train_data(train_df)

        assert "quality" not in x_train.columns
        assert y_train.name == "quality"
        assert len(x_train) == len(y_train) == len(train_df)

    def test_raises_if_target_column_missing(self, trainer_config):
        trainer = ModelTrainer(config=trainer_config, params=make_params())
        df = pd.DataFrame({"fixed acidity": [1, 2, 3]})

        with pytest.raises(ModelTrainingError):
            trainer.prepare_train_data(df)


class TestGetTrainingParams:
    def test_defaults_to_development(self, trainer_config, monkeypatch):
        monkeypatch.delenv("ENV", raising=False)
        params = make_params(n_jobs=4, cv=3)
        trainer = ModelTrainer(config=trainer_config, params=params)
        params = trainer.get_training_params()

        assert params.n_jobs == 4
        assert params.cv == 3

    def test_uses_production_when_env_set(self, trainer_config, monkeypatch):
        monkeypatch.setenv("ENV", "production")
        params_obj = make_params()
        params_obj.TRAINING.production = SimpleNamespace(n_jobs=2, cv=5)
        trainer = ModelTrainer(config=trainer_config, params=params_obj)
        params = trainer.get_training_params()

        assert params.n_jobs == 2
        assert params.cv == 5


class TestOptimize:
    def test_returns_valid_param_dict_within_configured_bounds(
        self, trainer_config, train_df
    ):
        params = make_params(n_trials=3, cv=2)
        trainer = ModelTrainer(config=trainer_config, params=params)
        x_train, y_train = trainer.prepare_train_data(train_df)
        best_params = trainer.optimize(x_train, y_train)

        assert isinstance(best_params, dict)
        expected_keys = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
        }
        assert set(best_params.keys()) == expected_keys

        assert 50 <= best_params["n_estimators"] <= 100
        assert 2 <= best_params["max_depth"] <= 5
        assert 0.01 <= best_params["learning_rate"] <= 0.1
        assert 0.6 <= best_params["subsample"] <= 0.9
        assert 0.6 <= best_params["colsample_bytree"] <= 0.9
        assert 1 <= best_params["min_child_weight"] <= 5
        assert 0.0 <= best_params["reg_alpha"] <= 1.0
        assert 1 <= best_params["reg_lambda"] <= 5

    def test_optimize_is_deterministic_with_same_random_state(
        self, trainer_config, train_df
    ):
        params = make_params(n_trials=3, cv=2)
        trainer1 = ModelTrainer(config=trainer_config, params=params)
        trainer2 = ModelTrainer(config=trainer_config, params=params)
        x_train, y_train = trainer1.prepare_train_data(train_df)

        result1 = trainer1.optimize(x_train, y_train)
        result2 = trainer2.optimize(x_train, y_train)

        assert result1 == result2


class TestTrainModel:
    def test_trains_and_save_model(self, trainer_config, train_df):
        params = make_params(n_trials=2, cv=2)
        trainer = ModelTrainer(config=trainer_config, params=params)
        x_train, y_train = trainer.prepare_train_data(train_df)
        best_params = trainer.optimize(x_train, y_train)
        model = trainer.train_model(x_train, y_train, best_params)

        assert hasattr(model, "predict")
        assert trainer_config.model_file.exists()

    def test_trained_model_can_predict(self, trainer_config, train_df):
        params = make_params(n_trials=2, cv=2)
        trainer = ModelTrainer(config=trainer_config, params=params)
        x_train, y_train = trainer.prepare_train_data(train_df)
        best_params = trainer.optimize(x_train, y_train)
        model = trainer.train_model(x_train, y_train, best_params)
        preds = model.predict(x_train.head(5))

        assert len(preds) == 5


class TestRun:
    def test_run_end_to_end_writes_success_status(self, trainer_config, train_df):
        train_df.to_csv(trainer_config.train_file, index=False)
        params = make_params(n_trials=2, cv=2)
        trainer = ModelTrainer(config=trainer_config, params=params)
        trainer.run()

        status = json.loads(trainer_config.status_file.read_text())
        assert status["stage"] == "model_trainer"
        assert status["status"] is True
        assert trainer_config.model_file.exists()

    def test_run_writes_failure_status_on_missing_train_file(self, trainer_config):
        params = make_params(n_trials=2, cv=2)
        trainer = ModelTrainer(config=trainer_config, params=params)

        with pytest.raises(ModelTrainingError):
            trainer.run()

        status = json.loads(trainer_config.status_file.read_text())
        assert status["stage"] == "model_trainer"
        assert status["status"] is False
