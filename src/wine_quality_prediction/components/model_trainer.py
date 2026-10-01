import os

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from wine_quality_prediction.entity import ModelTrainerConfig
from wine_quality_prediction.logger import ModelTrainingError, get_logger
from wine_quality_prediction.utils import get_size, save_bin, save_status

logger = get_logger("model_trainer", "model_trainer.log")


class ModelTrainer:
    """Find the best XGBoost parameters and train the final model."""

    def __init__(self, config: ModelTrainerConfig, params):
        """Initialize model trainer with configuration."""

        self.config = config
        self.params = params
        self.target_column = config.target_column

    def load_data(self) -> pd.DataFrame:
        """Load training data."""

        try:
            file_path = self.config.train_file
            logger.info(f"Reading train_file: '{file_path}'")
            logger.info(f"Train file size: '{get_size(file_path)}'")
            train = pd.read_csv(file_path)
            logger.info(f"Training data loaded successfully - shape='{train.shape}'")
            logger.info(f"Training columns: '{train.columns.tolist()}'")
            return train

        except Exception as e:
            logger.error(
                f"Training data loaded failed - path='{file_path}': {e}",
                exc_info=True,
            )
            raise ModelTrainingError("Failed to load training data") from e

    def prepare_train_data(self, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Separate features and target from training data."""

        try:
            if self.target_column not in train.columns:
                raise ValueError(
                    f"Target column '{self.target_column}' not found in training data"
                )

            x_train = train.drop(columns=[self.target_column])
            y_train = train[self.target_column]

            logger.info(f"Features shape: '{x_train.shape}'")
            logger.info(f"Target shape: '{y_train.shape}'")
            return x_train, y_train

        except Exception as e:
            logger.error(f"Training data preperation failed: {e}", exc_info=True)
            raise ModelTrainingError("Failed to prepare training data") from e

    def get_training_params(self):
        """Get environment specific training parameters."""

        environment = os.getenv("ENV", "development").lower()

        if environment == "production":
            training = self.params.TRAINING.production
        else:
            training = self.params.TRAINING.development

        logger.info(f"Training environment: '{environment}'")
        return training

    def optimize(self, x_train: pd.DataFrame, y_train: pd.Series):
        """Find the best XGBoost parameters using Optuna."""

        try:
            optimization = self.params.BAYESIAN_OPTIMIZATION
            xgb_params = self.params.XGBOOST
            training = self.get_training_params()

            n_trials = int(optimization.n_trials)
            cv = int(training.cv)
            scoring = optimization.scoring
            random_state = int(self.params.TRAINING.random_state)
            n_jobs = int(training.n_jobs)

            logger.info(
                f"Starting Bayesian optimization - trials='{n_trials}' cv='{cv}' "
                f"scoring='{scoring}' random_state='{random_state}' n_jobs='{n_jobs}'"
            )

            def objective(trial):
                model = XGBRegressor(
                    n_estimators=trial.suggest_int(
                        "n_estimators",
                        int(xgb_params.n_estimators.low),
                        int(xgb_params.n_estimators.high),
                    ),
                    max_depth=trial.suggest_int(
                        "max_depth",
                        int(xgb_params.max_depth.low),
                        int(xgb_params.max_depth.high),
                    ),
                    learning_rate=trial.suggest_float(
                        "learning_rate",
                        float(xgb_params.learning_rate.low),
                        float(xgb_params.learning_rate.high),
                    ),
                    subsample=trial.suggest_float(
                        "subsample",
                        float(xgb_params.subsample.low),
                        float(xgb_params.subsample.high),
                    ),
                    colsample_bytree=trial.suggest_float(
                        "colsample_bytree",
                        float(xgb_params.colsample_bytree.low),
                        float(xgb_params.colsample_bytree.high),
                    ),
                    min_child_weight=trial.suggest_int(
                        "min_child_weight",
                        int(xgb_params.min_child_weight.low),
                        int(xgb_params.min_child_weight.high),
                    ),
                    reg_alpha=trial.suggest_float(
                        "reg_alpha",
                        float(xgb_params.reg_alpha.low),
                        float(xgb_params.reg_alpha.high),
                    ),
                    reg_lambda=trial.suggest_int(
                        "reg_lambda",
                        int(xgb_params.reg_lambda.low),
                        int(xgb_params.reg_lambda.high),
                    ),
                    random_state=random_state,
                    n_jobs=1,
                )

                scores = cross_val_score(
                    model, x_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs
                )
                return scores.mean()

            sampler = optuna.samplers.TPESampler(seed=random_state)

            study = optuna.create_study(
                direction=optimization.direction, sampler=sampler
            )

            study.optimize(objective, n_trials=n_trials)

            logger.info(f"Optimization completed - best_score='{study.best_value}'")
            logger.info(f"Best parameters: '{study.best_params}'")

            return study.best_params

        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}", exc_info=True)
            raise ModelTrainingError("Failed to optimize model parameters") from e

    def train_model(
        self, x_train: pd.DataFrame, y_train: pd.Series, best_params: dict
    ) -> XGBRegressor:
        """Train the final XGBoost model."""

        try:
            training = self.get_training_params()
            random_state = int(self.params.TRAINING.random_state)
            n_jobs = int(training.n_jobs)
            logger.info(f"Trainig final XGBoost model with parameters: '{best_params}'")
            model = XGBRegressor(
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=n_jobs,
                **best_params,
            )
            model.fit(x_train, y_train)
            save_bin(model, self.config.model_file)
            logger.info(f"Final model saved: '{self.config.model_file}'")
            return model

        except Exception as e:
            logger.error(f"Final model training failed: {e}", exc_info=True)
            raise ModelTrainingError("Failed to train final model") from e

    def run(self):
        """Run the model training process."""

        try:
            train = self.load_data()
            x_train, y_train = self.prepare_train_data(train)
            best_params = self.optimize(x_train, y_train)
            self.train_model(x_train, y_train, best_params)
            save_status(self.config.status_file, "model_trainer", True)
            logger.info("Model training completed successfully")

        except Exception as e:
            save_status(self.config.status_file, "model_trainer", False)
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise ModelTrainingError("Model training process failed") from e
