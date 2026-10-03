from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMetadata:
    """Stores information about the model being served."""

    name: str
    bento_version: str
    mlflow_version: str | None
    mlflow_run_id: str | None
    r2: float | None
    rmse: float | None
    mae: float | None

    @property
    def distplay_version(self) -> str:
        """Return the MLflow version when available, otherwise the BentoML version."""

        return self.mlflow_version or self.bento_version

    @classmethod
    def from_bento_model(cls, bento_model) -> "ModelMetadata":
        labels = bento_model.info.labels or {}
        metadata = bento_model.info.metadata or {}
        return cls(
            name=bento_model.tag.name,
            bento_version=bento_model.tag.version,
            mlflow_version=labels.get("mlflow_version"),
            mlflow_run_id=metadata.get("mlflow_run_id"),
            r2=metadata.get("r2"),
            rmse=metadata.get("rmse"),
            mae=metadata.get("mae"),
        )
