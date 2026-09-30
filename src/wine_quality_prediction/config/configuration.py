from pathlib import Path

from wine_quality_prediction.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    SCHEMA_FILE_PATH,
)
from wine_quality_prediction.entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelPackagingConfig,
    ModelRegistryConfig,
    ModelTrainerConfig,
)
from wine_quality_prediction.utils import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            output_file=Path(config.output_file),
            database_table=config.database_table,
            kaggle_dataset=config.kaggle_dataset,
            status_file=Path(config.status_file),
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])
        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            input_file=Path(config.input_file),
            schema=self.schema.COLUMNS,
            status_file=Path(config.status_file),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            input_file=Path(config.input_file),
            train_file=Path(config.train_file),
            test_file=Path(config.test_file),
            status_file=Path(config.status_file),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        create_directories([config.root_dir])
        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_file=Path(config.train_file),
            model_file=Path(config.model_file),
            status_file=Path(config.status_file),
            target_column=self.schema.TARGET_COLUMN.name,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            model_file=Path(config.model_file),
            test_file=Path(config.test_file),
            metrics_file=Path(config.metrics_file),
            status_file=Path(config.status_file),
            target_column=self.schema.TARGET_COLUMN.name,
        )

    def get_model_registry_config(self) -> ModelRegistryConfig:
        config = self.config.model_registry
        create_directories([config.root_dir])
        return ModelRegistryConfig(
            root_dir=Path(config.root_dir),
            model_file=Path(config.model_file),
            test_file=Path(config.test_file),
            metrics_file=Path(config.metrics_file),
            model_name=config.model_name,
            production_threshold=float(config.production_threshold),
            status_file=Path(config.status_file),
        )

    def get_model_packaging_config(self) -> ModelPackagingConfig:
        config = self.config.model_packaging
        create_directories([config.root_dir])
        return ModelPackagingConfig(
            root_dir=Path(config.root_dir),
            mlflow_model_name=config.mlflow_model_name,
            mlflow_stage_name=config.mlflow_stage_name,
            bento_model_name=config.bento_model_name,
            info_file=Path(config.info_file),
            status_file=Path(config.status_file),
        )
