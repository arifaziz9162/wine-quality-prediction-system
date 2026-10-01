import os

import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

from wine_quality_prediction.database import DatabaseOperations
from wine_quality_prediction.entity import DataIngestionConfig
from wine_quality_prediction.logger import DataIngestionError, get_logger
from wine_quality_prediction.utils import get_size, save_status

load_dotenv()

logger = get_logger("data_ingestion", "data_ingestion.log")


class DataIngestion:
    """Handles data downloading from kaggle and storing raw data into database."""

    def __init__(self, config: DataIngestionConfig):
        """Initialize data ingestion with configuration"""

        self.config = config

    def download_file(self) -> None:
        """Download the dataset from kaggle."""

        try:
            file_path = self.config.output_file

            if not os.path.exists(file_path):
                logger.info(
                    f"Downloading dataset from kaggle: '{self.config.kaggle_dataset}' "
                )
                os.makedirs(self.config.root_dir, exist_ok=True)

                kaggle_username = os.getenv("KAGGLE_USERNAME")
                kaggle_api_token = os.getenv("KAGGLE_API_TOKEN")

                if not kaggle_username or not kaggle_api_token:
                    raise ValueError("KAGGLE_USERNAME and KAGGLE_API_TOKEN must be set")

                api = KaggleApi()
                api.authenticate()

                api.dataset_download_files(
                    dataset=self.config.kaggle_dataset,
                    path=str(self.config.root_dir),
                    unzip=True,
                )

                downloaded_file = os.path.join(
                    self.config.root_dir, "winequality-red.csv"
                )

                if not os.path.exists(downloaded_file):
                    raise FileNotFoundError(
                        f"Downloaded dataset not found: '{downloaded_file}'"
                    )

                os.replace(downloaded_file, file_path)

                logger.info(f"Dataset downloaded successfully to: '{file_path}'")
                logger.info(f"File size: '{get_size(file_path)}'")
            else:
                logger.info(f"Dataset already exists at: '{file_path}'")

        except Exception as e:
            logger.error(
                f"Download failed - "
                f"kaggle_dataset='{self.config.kaggle_dataset}' "
                f"destination='{file_path}': {e}",
                exc_info=True,
            )
            raise DataIngestionError("Failed to download data from kaggle") from e

    def load_csv(self) -> pd.DataFrame:
        """Load raw data from CSV file."""

        try:
            file_path = self.config.output_file
            logger.info(f"Reading file: '{file_path}'")
            logger.info(f"File size: '{get_size(file_path)}'")
            df = pd.read_csv(file_path)
            logger.info(f"CSV loaded successfully with shape: {df.shape}")
            logger.info(f"Columns found: {df.columns.tolist()}")
            return df

        except Exception as e:
            logger.error(
                f"CSV load failed - path='{self.config.output_file}': {e}",
                exc_info=True,
            )
            raise DataIngestionError("Failed to load CSV") from e

    def store_in_database(self, df: pd.DataFrame):
        """Store raw dataframe into database table."""

        db = DatabaseOperations()
        try:
            existing = db.fetch_data()

            if len(existing) > 0:
                logger.info("Data already in database skipping insert")
                return
            logger.info("Inserting raw data into database")
            db.insert_dataframe(df)
            logger.info("Data inserted successfully.")

        except Exception as e:
            logger.error(
                f"Database insert failed - table='{self.config.database_table}' "
                f"rows='{len(df)}': {e}",
                exc_info=True,
            )
            raise DataIngestionError("Failed to store data in database") from e
        finally:
            db.close_connection()

    def run(self):
        """Run the data ingestion process."""

        try:
            self.download_file()
            df = self.load_csv()
            self.store_in_database(df)
            save_status(self.config.status_file, "data_ingestion", True)
            logger.info("Data ingestion completed successfully")

        except Exception as e:
            save_status(self.config.status_file, "data_ingestion", False)
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise DataIngestionError("Data ingestion process failed") from e
