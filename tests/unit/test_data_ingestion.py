import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from wine_quality_prediction.components import DataIngestion
from wine_quality_prediction.entity import DataIngestionConfig
from wine_quality_prediction.logger import DataIngestionError


@pytest.fixture
def ingestion_config(tmp_path):
    status_dir = tmp_path / "data_ingestion"
    status_dir.mkdir(parents=True, exist_ok=True)
    return DataIngestionConfig(
        root_dir=tmp_path / "data_ingestion",
        output_file=tmp_path / "data_ingestion" / "WineQT.csv",
        database_table="wine_data",
        kaggle_dataset="uciml/red-wine-quality-cortez-et-al-2009",
        status_file=status_dir / "status.json",
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "fixed acidity": [7.4, 7.8],
            "volatile acidity": [0.7, 0.88],
            "quality": [5, 5],
        }
    )


class TestDownloadFile:
    def test_skips_download_if_file_exists(self, ingestion_config, tmp_path):
        ingestion_config.root_dir.mkdir(parents=True, exist_ok=True)
        ingestion_config.output_file.write_text("dummy, data\n1, 2\n")

        ingestion = DataIngestion(config=ingestion_config)

        with patch(
            "wine_quality_prediction.components.data_ingestion.KaggleApi"
        ) as mock_kaggle:
            ingestion.download_file()
            mock_kaggle.assert_not_called()

    def test_raises_when_kaggle_credentials_missing(
        self, ingestion_config, monkeypatch
    ):
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

        ingestion = DataIngestion(config=ingestion_config)

        with pytest.raises(DataIngestionError):
            ingestion.download_file()

    def test_downloads_and_renames_file(self, ingestion_config, monkeypatch, tmp_path):
        monkeypatch.setenv("KAGGLE_USERNAME", "fake_user")
        monkeypatch.setenv("KAGGLE_API_TOKEN", "fake_api_token")

        ingestion = DataIngestion(config=ingestion_config)

        def fake_download_files(dataset, path, unzip):
            downloaded = Path(path) / "winequality-red.csv"
            downloaded.write_text(
                "fixed acidity,volatile acidity,quality,\n7.4,0.7,5\n"
            )

        with patch(
            "wine_quality_prediction.components.data_ingestion.KaggleApi"
        ) as mock_kaggle_cls:
            mock_api = MagicMock()
            mock_api.dataset_download_files.side_effect = fake_download_files
            mock_kaggle_cls.return_value = mock_api

            ingestion.download_file()

            mock_api.authenticate.assert_called_once()
            assert ingestion_config.output_file.exists()

    def test_raises_when_downloaded_file_missing(self, ingestion_config, monkeypatch):
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

        ingestion = DataIngestion(config=ingestion_config)

        with patch(
            "wine_quality_prediction.components.data_ingestion.KaggleApi"
        ) as mock_kaggle_cls:
            mock_api = MagicMock()
            mock_kaggle_cls.return_value = mock_api

            with pytest.raises(DataIngestionError):
                ingestion.download_file()


class TestLoadCSV:
    def test_loads_csv_successfully(self, ingestion_config, sample_df):
        ingestion_config.root_dir.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(ingestion_config.output_file, index=False)

        ingestion = DataIngestion(config=ingestion_config)
        df = ingestion.load_csv()

        assert df.shape == sample_df.shape
        assert list(df.columns) == list(sample_df.columns)

    def test_raises_on_missing_file(self, ingestion_config):
        ingestion = DataIngestion(config=ingestion_config)

        with pytest.raises(DataIngestionError):
            ingestion.load_csv()


class TestStoreInDatabase:
    def test_skips_insert_if_data_already_exists(self, ingestion_config, sample_df):
        ingestion = DataIngestion(config=ingestion_config)

        with patch(
            "wine_quality_prediction.components.data_ingestion.DatabaseOperations"
        ) as mock_db_cls:
            mock_db = MagicMock()
            mock_db.fetch_data.return_value = sample_df
            mock_db_cls.return_value = mock_db

            ingestion.store_in_database(sample_df)

            mock_db.insert_dataframe.assert_not_called()
            mock_db.close_connection.assert_called_once()

    def test_inserts_when_database_empty(self, ingestion_config, sample_df):
        ingestion = DataIngestion(config=ingestion_config)

        with patch(
            "wine_quality_prediction.components.data_ingestion.DatabaseOperations"
        ) as mock_db_cls:
            mock_db = MagicMock()
            mock_db.fetch_data.return_value = pd.DataFrame()
            mock_db_cls.return_value = mock_db

            ingestion.store_in_database(sample_df)

            mock_db.insert_dataframe.assert_called_once_with(sample_df)
            mock_db.close_connection.assert_called_once()

    def test_closes_connection_even_on_failure(self, ingestion_config, sample_df):
        ingestion = DataIngestion(config=ingestion_config)

        with patch(
            "wine_quality_prediction.components.data_ingestion.DatabaseOperations"
        ) as mock_db_cls:
            mock_db = MagicMock()
            mock_db.fetch_data.side_effect = RuntimeError("connection dropped")
            mock_db_cls.return_value = mock_db

            with pytest.raises(DataIngestionError):
                ingestion.store_in_database(sample_df)

            mock_db.close_connection.assert_called_once()


class TestRun:
    def test_run_end_to_end_writes_on_success_status(self, ingestion_config, sample_df):
        ingestion_config.root_dir.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(ingestion_config.output_file, index=False)

        ingestion = DataIngestion(config=ingestion_config)

        with patch(
            "wine_quality_prediction.components.data_ingestion.DatabaseOperations"
        ) as mock_db_cls:
            mock_db = MagicMock()
            mock_db.fetch_data.return_value = sample_df
            mock_db_cls.return_value = mock_db
            ingestion.run()

        status = json.loads(ingestion_config.status_file.read_text())
        assert status["stage"] == "data_ingestion"
        assert status["status"] is True

    def test_run_writes_on_failue_status_on_error(self, ingestion_config, sample_df):
        ingestion_config.root_dir.mkdir(parents=True, exist_ok=True)

        ingestion = DataIngestion(config=ingestion_config)

        with patch(
            "wine_quality_prediction.components.data_ingestion.DataIngestion.download_file"
        ):
            with pytest.raises(DataIngestionError):
                ingestion.run()

        status = json.loads(ingestion_config.status_file.read_text())
        assert status["stage"] == "data_ingestion"
        assert status["status"] is False
