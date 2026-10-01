import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Serving configuration, read from environment variables."""

    SERVICE_NAME: str = "wine_quality_service"
    APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")

    # BentoML model sotre reference, created by the packaging stage
    BENTO_MODEL_TAG: str = os.getenv("BENTO_MODEL_TAG", "wine_quality_model:latest")
    MODEL_FILE_NAME: str = "model.joblib"

    # Traffic
    REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", 30))
    MAX_CONCURRENCY: int = int(os.getenv("MAX_CONCURRENCY", 32))

    # Prediction history (PostgreSQL)
    DB_SAVE_ENABLED: bool = os.getenv("DB_SAVE_ENABLED", "true").lower()
    DB_SAVE_WORKERS: int = int(os.getenv("DB_SAVE_WORKERS", 2))
    PREDICTIONS_MAX_LIMIT: int = 500


settings = Settings()
