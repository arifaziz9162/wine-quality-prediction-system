import os 
import urllib.request
import pandas as pd
from wine_quality_prediction.database import DatabaseOperations
from wine_quality_prediction.logger import get_logger, log_exceptions
from wine_quality_prediction.utils.common import get_size
from wine_quality_prediction.entity.config_entity import DataIngestionConfig

logger = get_logger("data_ingestion", "data_ingestion.log")

class DataIngestion:
    """Handles data loading from CSV and storing raw data into database. """

    def __init__(self, config: DataIngestionConfig):
        """Initialize with ingestion config."""
        self.config = config

    @log_exceptions
    def download_data(self):
        """Download dataset from Github raw URL."""
        if not os.path.exists(self.config.source_file):
            logger.info(f"Downloading dataset from: '{self.config.dataset_url}'")

            urllib.request.urlretrieve(
                self.config.dataset_url,
                self.config.source_file
            )
            logger.info(f"Dataset downloaded successfully to: '{self.config.source_file}'")
            logger.info(f"File size: '{get_size(self.config.source_file)}'")

    @log_exceptions
    def load_csv(self) -> pd.DataFrame:
        """Load raw data from CSV file."""
        file_path = self.config.source_file

        logger.info(f"Reading file: '{file_path}'")
        logger.info(f"File size: '{get_size(file_path)}'")

        df = pd.read_csv(file_path)

        logger.info(f"CSV loaded successfully with shape: '{df.shape}'")
        logger.info(f"Columns found: {df.columns.tolist()}")
        return df

    @log_exceptions
    def store_in_database(self, df: pd.DataFrame):
        """Store raw dataframe into database table."""
        db = DatabaseOperations()

        try:
            logger.info("Creating table if not exists")
            db.create_table()

            existing = db.fetch_data()
            if len(existing) > 0:
                logger.info("Data already in database skipping insert")
                return 

            logger.info("Inserting raw data into database")
            db.insert_dataframe(df)

            logger.info("Data inserted successfully.")
        finally:
            db.close_connection()

    @log_exceptions
    def run(self):
        """Execute full data ingestion pipeline."""
        self.download_data()
        df = self.load_csv()
        self.store_in_database(df)
        logger.info("Data ingestion completed successfully")