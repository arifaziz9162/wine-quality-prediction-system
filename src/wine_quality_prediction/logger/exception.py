class WineQualityBaseError(Exception):
    """Base error for wine quality prediction project."""

    pass


class DataIngestionError(WineQualityBaseError):
    """Raised when data ingestion fails."""

    pass


class DataValidationError(WineQualityBaseError):
    """Raised when data validation fails."""

    pass


class DataTransformationError(WineQualityBaseError):
    """Raised when data transformation fails."""

    pass


class ModelTrainingError(WineQualityBaseError):
    """Raised when model training fails."""

    pass


class ModelEvaluationError(WineQualityBaseError):
    """Raised when model evaluation fails."""

    pass


class ModelRegistryError(WineQualityBaseError):
    """Raised when model registry fails."""

    pass


class ModelInferenceError(WineQualityBaseError):
    """Raised when prediction/inference fails."""

    pass


class DatabaseError(WineQualityBaseError):
    """Raised when any database operation fails."""

    pass
