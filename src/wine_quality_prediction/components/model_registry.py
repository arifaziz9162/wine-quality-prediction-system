import json
import os

import dagshub
import mlflow
import mlflow.xgboost
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from wine_quality_prediction.entity import ModelRegistryConfig
from wine_quality_prediction.logger import ModelRegistryError, get_logger
from wine_quality_prediction.utils import load_bin, save_status

load_dotenv()

logger = get_logger("model_registry", "model_registry.log")


class ModelRegistry:
    """Register and promote the final trained model."""

    def __init__(self, config: ModelRegistryConfig):
        """Initialize model registry with configuration."""

        self.config = config

    def initialize_mlflow(self) -> None:
        """Initialize DagsHub MLflow tracking."""

        try:
            token = os.getenv("DAGSHUB_TOKEN")

            if not token:
                raise ModelRegistryError(
                    "DAGSHUB_TOKEN is not configured in the environment"
                )

            dagshub.auth.add_app_token(token)

            dagshub.init(
                repo_owner="TechArif",
                repo_name="wine-quality-prediction-system",
                mlflow=True,
            )

            mlflow.set_experiment("Wine Quality Prediction")

            logger.info("DagsHub MLflow initialized successfully")

        except Exception as e:
            logger.error(f"MLflow initialization failed: {e}", exc_info=True)
            raise ModelRegistryError("Failed to initialize MLflow") from e

    def load_model(self):
        """Load the final trained model."""

        try:
            if not self.config.model_file.exists():
                raise FileNotFoundError(f"Model not found: '{self.config.model_file}'")
            model = load_bin(self.config.model_file)
            logger.info(f"Trained model loaded '{self.config.model_file}'")
            return model

        except Exception as e:
            logger.error(f"Model loading failed: {e}", exc_info=True)
            raise ModelRegistryError("Failed to load model") from e

    def load_test_data(self) -> pd.DataFrame:
        """Load test dataset for model signature inference."""

        try:
            test = pd.read_csv(self.config.test_file)
            logger.info(f"Test data loaded for signature - shape: '{test.shape}'")
            return test

        except Exception as e:
            logger.error(f"Test data load failed: {e}", exc_info=True)
            raise ModelRegistryError("Failed to load test data") from e

    def load_metrics(self) -> dict:
        """Load model evaluation metrics."""

        try:
            if not self.config.metrics_file.exists():
                raise FileNotFoundError(
                    f"Metrics file not found: '{self.config.metrics_file}'"
                )
            with open(self.config.metrics_file) as f:
                metrics = json.load(f)
            logger.info(f"Evaluation metrics loaded: '{metrics}'")
            return metrics

        except Exception as e:
            logger.error(f"Metrics loading failed: {e}", exc_info=True)
            raise ModelRegistryError("Failed to load evaluation metrics") from e

    def register_model(self, model, metrics: dict, x_test: pd.DataFrame) -> str:
        """Register the final model in MLflow."""

        try:
            with mlflow.start_run(run_name="model_registry"):
                mlflow.set_tags(
                    {
                        "project": "wine-quality-prediction-system",
                        "stage": "Model Registry",
                        "model": type(model).__name__,
                        "framework": "scikit-learn",
                    }
                )

                if hasattr(model, "get_params"):
                    for key, value in model.get_params().items():
                        mlflow.log_param(key, value)
                mlflow.log_metrics(metrics)

                signature = infer_signature(x_test, model.predict(x_test))

                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="model",
                    registered_model_name=self.config.model_name,
                    signature=signature,
                    input_example=x_test.iloc[:1],
                )

                if self.config.metrics_file.exists():
                    mlflow.log_artifact(str(self.config.metrics_file))

                active_run = mlflow.active_run()
                if active_run is None:
                    raise ValueError("No active MLflow run found")
                run_id = active_run.info.run_id
                logger.info(f"Model registered successfully - run_id='{run_id}'")
                return run_id

        except Exception as e:
            logger.error(f"Model registration failed: {e}", exc_info=True)
            raise ModelRegistryError("Failed to register model") from e

    def get_latest_version(self) -> str:
        """Get the latest registered model version."""
        try:
            client = MlflowClient()

            versions = client.search_model_versions(
                filter_string=f"name='{self.config.model_name}'"
            )
            if not versions:
                raise ValueError("No registered model versions found")
            latest_version = max(
                versions, key=lambda model_version: int(model_version.version)
            )
            logger.info(f"Latest model version: '{latest_version.version}'")
            return latest_version.version

        except Exception as e:
            logger.error(f"Failed to get latest model version: {e}", exc_info=True)
            raise ModelRegistryError("Failed to get latest model version") from e

    def get_current_production_r2(self) -> float | None:
        """Get the R2 score of the currently deployed Production model, if any."""

        try:
            client = MlflowClient()
            versions = client.get_latest_versions(
                self.config.model_name, stages=["Production"]
            )

            if not versions:
                logger.info("No existing Production model found.")
                return None

            current = versions[0]
            run = client.get_run(current.run_id)
            r2 = run.data.metrics.get("r2")

            logger.info(
                f"Current production model - "
                f"version='{current.version}' r2='{r2}'"
            )
            return r2

        except Exception as e:
            logger.error(
                f"Failed to get current production R2: {e}",
                exc_info=True,
            )
            return None

    def promote_model(self, version: str, r2_score: float) -> None:
        """Promote the registered model to Production only if it beats the incumbent."""

        try:
            client = MlflowClient()

            production_threshold = float(self.config.production_threshold)
            current_production_r2 = self.get_current_production_r2()

            logger.info(f"Production threshold: '{production_threshold}'")
            logger.info(f"Challenger R2: '{r2_score:.4f}'")
            logger.info(f"Champion R2: '{current_production_r2}'")

            client.transition_model_version_stage(
                name=self.config.model_name, version=version, stage="Staging"
            )
            logger.info(f"Model version '{version}' moved to STAGING")

            meets_threshold = r2_score >= production_threshold
            beats_champion = (
                current_production_r2 is None or r2_score > current_production_r2
            )

            if meets_threshold and beats_champion:
                client.transition_model_version_stage(
                    name=self.config.model_name,
                    version=version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                logger.info(f"Model version '{version}' promoted to PRODUCTION")
            elif not meets_threshold:
                logger.warning(
                    f"Model version '{version}' below threshold, stays in STAGING"
                )
            else:
                logger.info(
                    f"Model version '{version}' did not beat champion "
                    f"(r2={r2_score:.4f} vs {current_production_r2:.4f}), "
                    f"stays in STAGING"
                )

        except Exception as e:
            logger.error(f"Model promotion failed: {e}", exc_info=True)
            raise ModelRegistryError("Failed to promote model") from e

    def run(self) -> None:
        """Run the model registry process."""
        try:
            self.initialize_mlflow()
            model = self.load_model()
            test = self.load_test_data()
            metrics = self.load_metrics()

            # hardcoded on purpose, just used for the mlflow signature here
            x_test = test.drop(columns=["quality"])

            run_id = self.register_model(model, metrics, x_test)
            logger.info(f"MLflow run completed - run_id: '{run_id}'")
            version = self.get_latest_version()
            self.promote_model(version=version, r2_score=float(metrics["r2"]))
            save_status(self.config.status_file, "model_registry", True)
            logger.info(f"Model version '{version}' registered successfully")
            logger.info("Model registry process completed successfully")

        except Exception as e:
            save_status(self.config.status_file, "model_registry", False)
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise ModelRegistryError("Model registry process failed") from e
