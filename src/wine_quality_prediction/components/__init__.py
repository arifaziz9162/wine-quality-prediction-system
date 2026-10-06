"""Component package exports with lazy imports."""

from importlib import import_module

__all__ = [
    "DataIngestion",
    "DataValidation",
    "DataTransformation",
    "ModelTrainer",
    "ModelEvaluation",
    "ModelRegistry",
    "ModelPackaging",
]

_COMPONENTS = {
    "DataIngestion": ".data_ingestion",
    "DataValidation": ".data_validation",
    "DataTransformation": ".data_transformation",
    "ModelTrainer": ".model_trainer",
    "ModelEvaluation": ".model_evaluation",
    "ModelRegistry": ".model_registry",
    "ModelPackaging": ".model_packaging",
}


def __getattr__(name: str):
    """Load a component only when it is accessed."""
    if name not in _COMPONENTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_COMPONENTS[name], package=__name__)
    component = getattr(module, name)

    globals()[name] = component
    return component
