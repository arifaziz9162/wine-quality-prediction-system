import json
from unittest.mock import patch

import pandas as pd
import pytest

from serving.metrics import DB_SAVE_TOTAL
from serving.persistence import PredictionRepository

PATH = "serving.persistence.prediction_repository.DatabaseOperations"


def saves(status):
    return DB_SAVE_TOTAL.labels(status=status)._value.get()


@pytest.fixture
def repository():
    repo = PredictionRepository(workers=1)
    yield repo
    repo.close()


def test_save_writes_and_always_closes_connection(repository):
    with patch(PATH) as db_cls:
        db = db_cls.return_value

        repository.save([1.0, 2.0], 5.5)

        db.save_prediction.assert_called_once_with([1.0, 2.0], 5.5)
        db.close_connection.assert_called_once()


def test_save_closes_connection_even_when_write_fails(repository):
    with patch(PATH) as db_cls:
        db = db_cls.return_value
        db.save_prediction.side_effect = RuntimeError("db down")

        with pytest.raises(RuntimeError):
            repository.save([1.0], 5.5)

        db.close_connection.assert_called_once()


def test_save_async_counts_success(repository):
    before = saves("success")
    with patch(PATH):
        repository.save_async([1.0], 5.5).result(timeout=5)

    assert saves("success") == before + 1


def test_save_async_failure_is_swallowed_and_counted(repository):
    before = saves("failure")
    with patch(PATH) as db_cls:
        db_cls.return_value.save_prediction.side_effect = RuntimeError("db down")

        future = repository.save_async([1.0], 5.5)
        future.exception(timeout=5)

    assert saves("failure") == before + 1


def test_fetch_recent_returns_tail_as_json_safe_records(repository):
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "predicted_quality": [5.0, 6.0, 7.0],
            "created_at": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        }
    )
    with patch(PATH) as db_cls:
        db = db_cls.return_value
        db.fetch_predictions.return_value = frame

        rows = repository.fetch_recent(2)

        json.dumps(rows)  # must be serialisable
        assert [r["id"] for r in rows] == [2, 3]
        db.close_connection.assert_called_once()
