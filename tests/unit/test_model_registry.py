import json
from unittest.mock import MagicMock, patch

import pytest

from wine_quality_prediction.components import ModelRegistry
from wine_quality_prediction.entity import ModelRegistryConfig
from wine_quality_prediction.logger import ModelRegistryError


@pytest.fixture
def registry_config(tmp_path):
    status_dir = tmp_path / "model_registry"
    status_dir.mkdir(parents=True, exist_ok=True)

    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"fake-model-bytes")

    test_file = tmp_path / "test.csv"
    test_file.write_text("quality\n5\n6\n")

    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps({"r2": 0.5, "rmse": 0.6, "mae": 0.47}))

    return ModelRegistryConfig(
        root_dir=tmp_path,
        model_file=model_file,
        test_file=test_file,
        metrics_file=metrics_file,
        model_name="WineQualityModel",
        production_threshold=0.45,
        status_file=status_dir / "status.json",
    )


class TestLoadModel:
    def test_raises_on_missing_model_file(self, registry_config):
        registry_config.model_file.unlink()
        registry = ModelRegistry(config=registry_config)

        with pytest.raises(ModelRegistryError):
            registry.load_model()


class TestLoadMetrics:
    def test_loads_metrics_successfully(self, registry_config):
        registry = ModelRegistry(config=registry_config)
        metrics = registry.load_metrics()

        assert metrics["r2"] == 0.5

    def test_raises_on_missing_metrics_file(self, registry_config):
        registry_config.metrics_file.unlink()
        registry = ModelRegistry(config=registry_config)

        with pytest.raises(ModelRegistryError):
            registry.load_metrics()


class TestGetCurrentProductionR2:
    def test_returns_r2_when_production_model_exists(self, registry_config):
        registry = ModelRegistry(config=registry_config)

        with patch(
            "wine_quality_prediction.components.model_registry.MlflowClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            champion_version = MagicMock(run_id="run-1")
            mock_client.get_latest_versions.return_value = [champion_version]
            mock_client.get_run.return_value.data.metrics = {"r2": 0.62}
            mock_client_cls.return_value = mock_client

            r2 = registry.get_current_production_r2()

            assert r2 == 0.62
            mock_client.get_latest_versions.assert_called_once_with(
                "WineQualityModel", stages=["Production"]
            )

    def test_returns_none_when_no_production_model_exists(self, registry_config):
        registry = ModelRegistry(config=registry_config)

        with patch(
            "wine_quality_prediction.components.model_registry.MlflowClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_latest_versions.return_value = []
            mock_client_cls.return_value = mock_client

            r2 = registry.get_current_production_r2()

            assert r2 is None

    def test_returns_none_on_client_failure(self, registry_config):
        registry = ModelRegistry(config=registry_config)

        with patch(
            "wine_quality_prediction.components.model_registry.MlflowClient"
        ) as mock_client_cls:
            mock_client_cls.side_effect = RuntimeError("mlflow unreachable")

            r2 = registry.get_current_production_r2()

            assert r2 is None


class TestPromoteModel:
    def test_promotes_to_production_when_score_meets_threshold(self, registry_config):
        registry = ModelRegistry(config=registry_config)

        with patch(
            "wine_quality_prediction.components.model_registry.MlflowClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_latest_versions.return_value = []
            mock_client_cls.return_value = mock_client

            registry.promote_model(version="1", r2_score=0.5)

            calls = mock_client.transition_model_version_stage.call_args_list
            stages_set = [call.kwargs["stage"] for call in calls]

            assert "Staging" in stages_set
            assert "Production" in stages_set

    def test_promotes_to_production_when_score_exactly_at_threshold(
        self, registry_config
    ):
        registry = ModelRegistry(config=registry_config)

        with patch(
            "wine_quality_prediction.components.model_registry.MlflowClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_latest_versions.return_value = []
            mock_client_cls.return_value = mock_client

            registry.promote_model(version="1", r2_score=0.45)

            calls = mock_client.transition_model_version_stage.call_args_list
            stages_set = [call.kwargs["stage"] for call in calls]

            assert "Production" in stages_set

    def test_stays_in_staging_when_score_below_threshold(self, registry_config):
        registry = ModelRegistry(config=registry_config)

        with patch(
            "wine_quality_prediction.components.model_registry.MlflowClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_latest_versions.return_value = []
            mock_client_cls.return_value = mock_client

            registry.promote_model(version="1", r2_score=0.30)

            calls = mock_client.transition_model_version_stage.call_args_list
            stages_set = [call.kwargs["stage"] for call in calls]

            assert "Staging" in stages_set
            assert "Production" not in stages_set

    def test_always_transitions_to_staging_first(self, registry_config):
        registry = ModelRegistry(config=registry_config)

        with patch(
            "wine_quality_prediction.components.model_registry.MlflowClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_latest_versions.return_value = []
            mock_client_cls.return_value = mock_client

            registry.promote_model(version="1", r2_score=0.0)

            first_call = mock_client.transition_model_version_stage.call_args_list[0]
            assert first_call.kwargs["stage"] == "Staging"

    def test_raises_wrapped_error_on_client_failure(self, registry_config):
        registry = ModelRegistry(config=registry_config)

        with patch(
            "wine_quality_prediction.components.model_registry.MlflowClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_latest_versions.return_value = []
            mock_client.transition_model_version_stage.side_effect = RuntimeError(
                "mlflow down"
            )
            mock_client_cls.return_value = mock_client

            with pytest.raises(ModelRegistryError):
                registry.promote_model(version="1", r2_score=0.5)

    def test_does_not_promote_when_challenger_worse_than_champion(
        self, registry_config
    ):
        registry = ModelRegistry(config=registry_config)

        with patch(
            "wine_quality_prediction.components.model_registry.MlflowClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            champion_version = MagicMock(run_id="run-1")
            mock_client.get_latest_versions.return_value = [champion_version]
            mock_client.get_run.return_value.data.metrics = {"r2": 0.7}
            mock_client_cls.return_value = mock_client

            # 0.5 clears the 0.45 threshold but is worse than the 0.7 champion
            registry.promote_model(version="2", r2_score=0.5)

            calls = mock_client.transition_model_version_stage.call_args_list
            stages = [call.kwargs["stage"] for call in calls]

            assert "Staging" in stages
            assert "Production" not in stages

    def test_promotes_when_challenger_beats_champion(self, registry_config):
        registry = ModelRegistry(config=registry_config)

        with patch(
            "wine_quality_prediction.components.model_registry.MlflowClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            champion_version = MagicMock(run_id="run-1")
            mock_client.get_latest_versions.return_value = [champion_version]
            mock_client.get_run.return_value.data.metrics = {"r2": 0.5}
            mock_client_cls.return_value = mock_client

            registry.promote_model(version="2", r2_score=0.65)

            calls = mock_client.transition_model_version_stage.call_args_list
            stages = [call.kwargs["stage"] for call in calls]
            production_call = next(
                call for call in calls if call.kwargs["stage"] == "Production"
            )

            assert "Production" in stages
            assert production_call.kwargs["archive_existing_versions"] is True

    def test_promotes_when_no_existing_champion(self, registry_config):
        registry = ModelRegistry(config=registry_config)

        with patch(
            "wine_quality_prediction.components.model_registry.MlflowClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_latest_versions.return_value = []
            mock_client_cls.return_value = mock_client

            registry.promote_model(version="1", r2_score=0.5)

            calls = mock_client.transition_model_version_stage.call_args_list
            stages = [call.kwargs["stage"] for call in calls]

            assert "Production" in stages


class TestGetLatestVersion:
    def test_returns_highest_version_number(self, registry_config):
        registry = ModelRegistry(config=registry_config)

        v1 = MagicMock(version="1")
        v2 = MagicMock(version="3")
        v3 = MagicMock(version="2")

        with patch(
            "wine_quality_prediction.components.model_registry.MlflowClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.search_model_versions.return_value = [v1, v2, v3]
            mock_client_cls.return_value = mock_client
            latest = registry.get_latest_version()

            assert latest == "3"

    def test_raises_when_no_versions_found(self, registry_config):
        registry = ModelRegistry(config=registry_config)

        with patch(
            "wine_quality_prediction.components.model_registry.MlflowClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.search_model_versions.return_value = []
            mock_client_cls.return_value = mock_client

            with pytest.raises(ModelRegistryError):
                registry.get_latest_version()


class TestRun:
    def test_run_writes_success_status(self, registry_config):
        registry = ModelRegistry(config=registry_config)

        with (
            patch.object(registry, "initialize_mlflow"),
            patch.object(registry, "load_model", return_value=MagicMock()),
            patch.object(registry, "register_model", return_value="run-123"),
            patch.object(registry, "get_latest_version", return_value="1"),
            patch.object(registry, "promote_model"),
        ):
            registry.run()

        status = json.loads(registry_config.status_file.read_text())
        assert status["stage"] == "model_registry"
        assert status["status"] is True

    def test_run_writes_failure_status_on_error(self, registry_config):
        registry = ModelRegistry(config=registry_config)

        with patch.object(
            registry, "initialize_mlflow", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(ModelRegistryError):
                registry.run()

        status = json.loads(registry_config.status_file.read_text())
        assert status["stage"] == "model_registry"
        assert status["status"] is False
