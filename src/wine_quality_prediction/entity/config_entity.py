from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    output_file: Path
    database_table: str
    kaggle_dataset: str
    status_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    input_file: Path
    schema: dict
    status_file: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    input_file: Path
    train_file: Path
    test_file: Path
    status_file: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_file: Path
    model_file: Path
    status_file: Path
    target_column: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model_file: Path
    test_file: Path
    metrics_file: Path
    status_file: Path
    target_column: str


@dataclass(frozen=True)
class ModelRegistryConfig:
    root_dir: Path
    model_file: Path
    test_file: Path
    metrics_file: Path
    model_name: str
    production_threshold: float
    status_file: Path
