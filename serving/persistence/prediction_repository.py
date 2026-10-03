import json
from concurrent.futures import Future, ThreadPoolExecutor

from serving.config import settings
from serving.metrics import DB_SAVE_TOTAL
from wine_quality_prediction.database import DatabaseOperations
from wine_quality_prediction.logger import DatabaseError, get_logger

logger = get_logger("prediction_repository", "serving.log")


class PredictionRepository:
    """Handles saving and retrieving prediction history."""

    def __init__(self, workers: int | None = None):
        self._executor = ThreadPoolExecutor(
            max_workers=workers or settings.DB_SAVE_WORKERS,
            thread_name_prefix="prediction-db",
        )

    def save(self, features: list[float], prediction: float) -> None:
        """Saves a prediction to PostgreSQL."""

        db = DatabaseOperations()
        try:
            db.save_prediction(features, prediction)
        finally:
            db.close_connection()

    def save_async(self, features: list[float], prediction: float) -> Future:
        """Saves a prediction in the background."""

        future = self._executor.submit(self.save, features, prediction)
        future.add_done_callback(self._on_saved)
        return future

    @staticmethod
    def _on_saved(future: Future) -> None:
        """Updates the database save metric after the task finishes."""

        error = future.exception()
        if error is None:
            DB_SAVE_TOTAL.labels(status="success").inc()
            return
        DB_SAVE_TOTAL.labels(status="failure").inc()
        logger.warning(f"Prediction generated but DB save failed: '{error}'")

    def fetch_recent(self, limit: int) -> list[dict]:
        """ "Returns the most recent stored predictions."""

        try:
            db = DatabaseOperations()
            frame = db.fetch_predictions().tail(limit)
            return json.loads(frame.to_json(orient="records", date_format="iso"))
        except DatabaseError:
            raise
        finally:
            db.close_connection()

    def close(self) -> None:
        """Shut down the background database worker."""
        self._executor.shutdown(wait=False)
