import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from wine_quality_prediction.components import ModelPackaging
from wine_quality_prediction.entity import ModelPackagingConfig
from wine_quality_prediction.logger import ModelPackagingError

MODULE = "wine_quality_prediction.components.model_packaging"


class FakeNotFound(Exception):
    pass


class FakeBentoModel:
    """Minimal stand-in for the object bentoml.models.create() yields."""

    def __init__(self, directory: Path, tag="wine_quality_model:new123"):
        self._directory = directory
        self.tag = tag

    def path_of(self, name):
        return str(self._directory / name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_bentoml(tmp_path, existing=None):
    """Fake `bentoml` module. `existing=None` means the store has no such model."""

    if existing is None:
        list_model = MagicMock(side_effect=FakeNotFound("no Models"))
    else:
        list_model = MagicMock(return_value=existing)

    return SimpleNamespace(
        exceptions=SimpleNamespace(NotFound=FakeNotFound),
        models=SimpleNamespace(
            list=list_model,
            create=MagicMock(return_value=FakeBentoModel(tmp_path)),
        ),
    )


def existing_model(mlflow_version, tag="wine_quality_model:old999"):
    return SimpleNamespace(
        tag=tag, info=SimpleNamespace(labels={"mlflow_version": mlflow_version})
    )


@pytest.fixture
def packaging_config(tmp_path):
    return ModelPackagingConfig(
        root_dir=tmp_path,
        mlflow_model_name="WineQualityModel",
        mlflow_model_stage="Production",
        bento_model_name="wine_quality_model",
        info_file=tmp_path / "info.json",
        status_file=tmp_path / "status.json",
    )


@pytest.fixture
def version():
    return SimpleNamespace(version="7", run_id="run-7")


class TestInitializeMlflow:
    def test_respects_existing_tracking_uri(self, packaging_config, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:/tmp/x")
        monkeypatch.delenv("DAGSHUB_TOKEN", raising=False)

        with patch(f"{MODULE}.mlflow.set_tracking_uri") as set_uri:
            ModelPackaging(packaging_config).initialize_mlflow()

        set_uri.assert_not_called()

    def test_points_mlflow_at_dagshub_with_token(self, packaging_config, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.setenv("DAGSHUB_TOKEN", "tok")
        monkeypatch.setenv("DAGSHUB_USERNAME", "arif")
        monkeypatch.setenv("DAGSHUB_REPO_OWNER", "TechArif")
        monkeypatch.setenv("DAGSHUB_REPO_NAME", "wine-quality-prediction-system")

        with patch(f"{MODULE}.mlflow.set_tracking_uri") as set_uri:
            ModelPackaging(packaging_config).initialize_mlflow()

        set_uri.assert_called_once_with(
            "https://dagshub.com/TechArif/wine-quality-prediction-system.mlflow"
        )
        import os

        assert os.environ["MLFLOW_TRACKING_USERNAME"] == "arif"
        assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "tok"

    def test_raises_without_any_credentials(self, packaging_config, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.delenv("DAGSHUB_TOKEN", raising=False)

        with pytest.raises(ModelPackagingError):
            ModelPackaging(packaging_config).initialize_mlflow()


class TestGetProductionVersion:
    def test_returns_the_production_entry(self, packaging_config, version):
        with patch(f"{MODULE}.MlflowClient") as client_cls:
            client_cls.return_value.get_latest_versions.return_value = [version]

            found = ModelPackaging(packaging_config).get_production_version()

        assert found.version == "7"
        client_cls.return_value.get_latest_versions.assert_called_once_with(
            "WineQualityModel", stages=["Production"]
        )

    def test_raises_when_nothing_is_in_production(self, packaging_config):
        with patch(f"{MODULE}.MlflowClient") as client_cls:
            client_cls.return_value.get_latest_versions.return_value = []

            with pytest.raises(ModelPackagingError):
                ModelPackaging(packaging_config).get_production_version()


class TestGetMetrics:
    def test_keeps_only_known_metrics_as_floats(self, packaging_config):
        with patch(f"{MODULE}.MlflowClient") as client_cls:
            run = client_cls.return_value.get_run.return_value
            run.data.metrics = {"r2": 0.45, "rmse": 0.6, "mae": 0.4, "other": 9}

            metrics = ModelPackaging(packaging_config).get_metrics("run-7")

        assert metrics == {"r2": 0.45, "rmse": 0.6, "mae": 0.4}


class TestFindExisting:
    def test_returns_none_for_a_brand_new_store(self, packaging_config, tmp_path):
        packaging = ModelPackaging(packaging_config)

        assert packaging.find_existing(make_bentoml(tmp_path), "7") is None

    def test_matches_on_mlflow_version_label(self, packaging_config, tmp_path):
        bentoml = make_bentoml(
            tmp_path, existing=[existing_model("6"), existing_model("7", "m:hit")]
        )

        found = ModelPackaging(packaging_config).find_existing(bentoml, "7")

        assert found.tag == "m:hit"

    def test_returns_none_when_no_label_matches(self, packaging_config, tmp_path):
        bentoml = make_bentoml(tmp_path, existing=[existing_model("6")])

        assert ModelPackaging(packaging_config).find_existing(bentoml, "7") is None


class TestPackage:
    def test_creates_bento_model_with_labels_and_metadata(
        self, packaging_config, version, tmp_path
    ):
        bentoml = make_bentoml(tmp_path)
        metrics = {"r2": 0.4581, "rmse": 0.62, "mae": 0.48}

        with patch(f"{MODULE}._import_bentoml", return_value=bentoml):
            result = ModelPackaging(packaging_config).package(
                {"fake": "model"}, version, metrics
            )

        kwargs = bentoml.models.create.call_args.kwargs
        assert kwargs["name"] == "wine_quality_model"
        assert kwargs["labels"]["mlflow_version"] == "7"
        assert kwargs["labels"]["mlflow_stage"] == "Production"
        assert kwargs["metadata"]["r2"] == 0.4581
        assert kwargs["metadata"]["mlflow_run_id"] == "run-7"
        assert (tmp_path / "model.joblib").exists()
        assert result.tag == "wine_quality_model:new123"

    def test_reuses_model_already_packaged_from_same_mlflow_version(
        self, packaging_config, version, tmp_path
    ):
        bentoml = make_bentoml(tmp_path, existing=[existing_model("7")])

        with patch(f"{MODULE}._import_bentoml", return_value=bentoml):
            result = ModelPackaging(packaging_config).package({}, version, {})

        bentoml.models.create.assert_not_called()
        assert result.tag == "wine_quality_model:old999"


class TestRun:
    def test_success_writes_info_for_ci_and_success_status(
        self, packaging_config, version, tmp_path
    ):
        packaging = ModelPackaging(packaging_config)
        bento_model = SimpleNamespace(tag="wine_quality_model:new123")

        with (
            patch.object(packaging, "initialize_mlflow"),
            patch.object(packaging, "get_production_version", return_value=version),
            patch.object(packaging, "get_metrics", return_value={"r2": 0.46}),
            patch.object(packaging, "load_model", return_value=object()),
            patch.object(packaging, "package", return_value=bento_model),
        ):
            packaging.run()

        info = json.loads(packaging_config.info_file.read_text())
        assert info["bento_model"] == "wine_quality_model:new123"
        assert info["mlflow_version"] == "7"
        assert info["r2"] == 0.46
        status = json.loads(packaging_config.status_file.read_text())
        assert status == {**status, "stage": "model_packaging", "status": True}

    def test_failure_writes_failure_status_and_raises(self, packaging_config):
        packaging = ModelPackaging(packaging_config)

        with patch.object(packaging, "initialize_mlflow", side_effect=RuntimeError):
            with pytest.raises(ModelPackagingError):
                packaging.run()

        status = json.loads(packaging_config.status_file.read_text())
        assert status["status"] is False
        assert not packaging_config.info_file.exists()
