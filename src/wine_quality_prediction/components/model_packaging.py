import os

import joblib
import mlflow
import mlflow.xgboost
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from wine_quality_prediction.entity import ModelPackagingConfig
from wine_quality_prediction.logger import ModelPackagingError, get_logger
from wine_quality_prediction.utils import save_json, save_status

load_dotenv()

logger = get_logger("model_packaging", "model_packaging.log")

MODEL_FILE_NAME = "model.joblib"


def _import_bentoml():
    """Import bentoml on demand see module docstring."""

    import bentoml  # type: ignore[import-not-found]

    return bentoml


class ModelPackaging:
    """Turn the current Prouduction model into a versioned BentoML model."""

    def __init__(self, config: ModelPackagingConfig):
        self.config = config

    def initialize_mlflow(self) -> None:
        """Point MLflow to DagsHub using plain environment variables."""

        try:
            if os.getenv("MLFLOW_TRACKING_URI"):
                logger.info("Using MLFLOW_TRACKING_URI fro the environment")
                return

            token = os.getenv("DAGSHUB_TOKEN")
            if not token:
                raise ModelPackagingError(
                    "Set MLFLOW_TRACKING_URI, or DAGSHUB_TOKEN for DagsHub"
                )

            owner = os.getenv("DAGSHUB_REPO_OWNER", "TechArif")
            repo = os.getenv("DAGSHUB_REPO_NAME", "wine-quality-prediction-system")

            os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv(
                "DAGSHUB_USERNAME", owner
            )
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            mlflow.set_tracking_uri(f"https://dagshub.com/{owner}/{repo}.mlflow")
            logger.info(f"MLflow tracking set to DagsHub repo '{owner}/{repo}'")

        except Exception as e:
            logger.error(f"MLflow initialization failed: {e}", exc_info=True)
            raise ModelPackagingError("Failed to initialize MLflow") from e

    def get_production_version(self):
        """Return the registry entry currently in the configured stage."""

        try:
            client = MlflowClient()
            versions = client.get_latest_versions(
                self.config.mlflow_model_name, stages=[self.config.mlflow_model_stage]
            )
            if not versions:
                raise ValueError(
                    f"No '{self.config.mlflow_model_name}' version in stage "
                    f"'{self.config.mlflow_model_stage}'. Nothing to package."
                )
            version = versions[0]
            logger.info(
                f"Found {self.config.mlflow_model_stage} model - "
                f"version='{version.version}' run_id='{version.run_id}'"
            )
            return version

        except Exception as e:
            logger.error(f"Failed to find production model: {e}", exc_info=True)
            raise ModelPackagingError("Failed to find production model") from e

    def get_metrics(self, run_id: str) -> dict:
        """Read the evaluation metrics logged for the model's run."""

        try:
            metrics = MlflowClient().get_run(run_id).data.metrics
            return {k: float(metrics[k]) for k in ("r2", "rmse", "mae") if k in metrics}

        except Exception as e:
            logger.error(f"Failed to read run metrics: {e}", exc_info=True)
            raise ModelPackagingError("Failed to read run metrics") from e

    def load_model(self, version):
        try:
            uri = f"models:/{self.config.mlflow_model_name}/{version}"

            logger.info(f"Loading MLflow model from URI: {uri}")

            model = mlflow.xgboost.load_model(uri)

            logger.info(
                f"Successfully loaded MLflow model version {version}"
            )

            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise ModelPackagingError("Failed to load model from MLflow") from e

    def find_existing(self, bentoml, mlflow_version: str):
        """Return the BentoML model already built from this MLflow version if any."""

        try:
            candidates = bentoml.models.list(self.config.bento_model_name)
        except bentoml.exceptions.NotFound:
            return None

        for existing in candidates:
            if (existing.info.labels or {}).get("mlflow_version") == mlflow_version:
                return existing

        return None

    def package(self, model, version, metrics: dict):
        """Save the model into the BentoML model store once per MLflow version."""

        try:
            bentoml = _import_bentoml()
            mlflow_version = str(version.version)

            existing = self.find_existing(bentoml, mlflow_version)
            if existing is not None:
                logger.info(
                    f"MLflow version '{mlflow_version}' already packaged as "
                    f"'{existing.tag}', reusing it"
                )
                return existing

            labels = {
                "mlflow_model": self.config.mlflow_model_name,
                "mlflow_version": mlflow_version,
                "mlflow_stage": self.config.mlflow_model_stage,
            }
            metadata = {**metrics, "mlflow_run_id": version.run_id}

            with bentoml.models.create(
                name=self.config.bento_model_name, labels=labels, metadata=metadata
            ) as bento_model:
                joblib.dump(model, bento_model.path_of(MODEL_FILE_NAME))

            return bento_model

        except ModelPackagingError:
            raise
        except Exception as e:
            logger.error(f"Packaging failed: {e}", exc_info=True)
            raise ModelPackagingError("Failed to package model") from e

    def save_info(self, bento_model, version, metrics: dict) -> None:
        """Write a small JSON file that CI reads to tag the container image."""

        save_json(
            self.config.info_file,
            {
                "bento_model": str(bento_model.tag),
                "mlflow_version": str(version.version),
                "mlflow_run_id": version.run_id,
                **metrics,
            },
        )

    def run(self):
        "Run the model packaging process."

        try:
            self.initialize_mlflow()
            version = self.get_production_version()
            metrics = self.get_metrics(version.run_id)
            model = self.load_model(version.version)
            bento_model = self.package(model, version, metrics)
            self.save_info(bento_model, version, metrics)
            save_status(self.config.status_file, "model_packaging", True)
            logger.info("Model packaging completed successfully")

        except Exception as e:
            save_status(self.config.status_file, "model_packaging", False)
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise ModelPackagingError("Model packaging process failed") from e
