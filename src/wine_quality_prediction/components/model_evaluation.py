import dagshub
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from wine_quality_prediction.entity import ModelEvaluationConfig
from wine_quality_prediction.logger import ModelEvaluationError, get_logger
from wine_quality_prediction.utils import get_size, load_bin, save_json, save_status

load_dotenv()

logger = get_logger("model_evaluation", "model_evaluation.log")


class ModelEvaluation:
    """Evaluate the trained model and track its results."""

    def __init__(self, config: ModelEvaluationConfig):
        """Initialize model evaluation with configuration."""

        self.config = config
        self.target_column = config.target_column

        dagshub.init(
            repo_owner="TechArif",
            repo_name="wine-quality-prediction-system",
            mlflow=True,
        )

        mlflow.set_experiment("Wine Quality Prediction")

    def load_model(self):
        """Load the trained model from artifacts directory."""

        try:
            file_path = self.config.model_file
            logger.info(f"Loading model: '{file_path}'")
            model = load_bin(file_path)
            logger.info("Trained model loaded successfully.")
            return model

        except Exception as e:
            logger.error(
                f"Model load failed - path='{file_path}': {e}",
                exc_info=True,
            )
            raise ModelEvaluationError("Failed to load model") from e

    def load_test_data(self) -> pd.DataFrame:
        """Load the test dataset."""

        try:
            file_path = self.config.test_file

            logger.info(f"Reading test file: '{file_path}'")
            logger.info(f"Test file size: '{get_size(file_path)}'")
            test = pd.read_csv(file_path)
            logger.info(f"Test data loaded successfully - shape: '{test.shape}'")
            logger.info(f"Test columns: '{test.columns.tolist()}'")
            return test

        except Exception as e:
            logger.error(
                f"Test data load failed - path='{self.config.test_file}': {e}",
                exc_info=True,
            )
            raise ModelEvaluationError("Failed to load test data") from e

    def prepare_test_data(self, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Separate features and target from testing data."""
        try:
            if self.target_column not in test.columns:
                raise ValueError(
                    f"Target column '{self.target_column}' not found in testing data"
                )

            x_test = test.drop(columns=[self.target_column])
            y_test = test[self.target_column]

            logger.info(f"Test features shape: '{x_test.shape}'")
            logger.info(f"Test target shape: '{y_test.shape}'")
            return x_test, y_test

        except Exception as e:
            logger.error(f"Test data preperation failed: {e}", exc_info=True)
            raise ModelEvaluationError("Failed to prepare test data") from e

    def evaluate(
        self, model, x_test: pd.DataFrame, y_test: pd.Series
    ) -> tuple[dict, np.ndarray]:
        """Calculate evaluation metrics for the trained model."""
        try:
            y_pred = model.predict(x_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            metrics = {
                "r2": round(float(r2), 4),
                "rmse": round(float(rmse), 4),
                "mae": round(float(mae), 4),
            }

            logger.info(f"R2 Score: '{metrics['r2']}'")
            logger.info(f"RMSE: '{metrics['rmse']}'")
            logger.info(f"MAE: '{metrics['mae']}'")

            save_json(self.config.metrics_file, metrics)
            logger.info(f"Metrics saved: '{self.config.metrics_file}'")
            return metrics, y_pred

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}", exc_info=True)
            raise ModelEvaluationError("Failed to evaluate model") from e

    def log_to_mlflow(self, model, metrics: dict, x_test: pd.DataFrame) -> None:
        """Log model, metrics and artifacts to MLflow."""

        try:
            with mlflow.start_run(run_name="model_evaluation"):
                mlflow.set_tags(
                    {
                        "project": "wine-quality-prediction-system",
                        "stage": "Model Evaluation",
                        "model": type(model).__name__,
                        "framework": "scikit-learn",
                    }
                )

                if hasattr(model, "get_params"):
                    mlflow.log_params(model.get_params())

                mlflow.log_metrics(metrics)

                signature = infer_signature(x_test, model.predict(x_test))

                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=x_test.iloc[:1],
                )

                if self.config.metrics_file.exists():
                    mlflow.log_artifact(str(self.config.metrics_file))

                run_id = mlflow.active_run().info.run_id
                logger.info(f"MLflow run_id: '{run_id}'")

        except Exception as e:
            logger.error(f"MLflow logging failed: {e}", exc_info=True)
            raise ModelEvaluationError("Failed to log evaluation results") from e

    def run(self):
        """Run the model evaluation process."""

        try:
            model = self.load_model()
            test = self.load_test_data()
            x_test, y_test = self.prepare_test_data(test)
            metrics, y_pred = self.evaluate(model, x_test, y_test)
            self.log_to_mlflow(model, metrics, x_test)
            save_status(self.config.status_file, "model_evaluation", True)
            logger.info("Model evaluation completed successfully")
            return metrics, y_pred, y_test

        except Exception as e:
            save_status(self.config.status_file, "model_evaluation", False)
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise ModelEvaluationError("Model evaluation process failed") from e
