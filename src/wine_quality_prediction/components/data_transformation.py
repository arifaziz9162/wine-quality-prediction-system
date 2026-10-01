import pandas as pd
from sklearn.model_selection import train_test_split

from wine_quality_prediction.entity import DataTransformationConfig
from wine_quality_prediction.logger import DataTransformationError, get_logger
from wine_quality_prediction.utils import get_size, save_status

logger = get_logger("data_transformation", "data_transformation.log")


class DataTransformation:
    """Clean, split, scale, and save the training data."""

    def __init__(self, config: DataTransformationConfig, params):
        """Initialize data transforamtion with configuration."""

        self.config = config
        self.params = params

    def load_data(self) -> pd.DataFrame:
        """Load input dataset from CSV file."""

        try:
            file_path = self.config.input_file
            logger.info(f"Reading file: '{file_path}'")
            logger.info(f"File size: '{get_size(file_path)}'")
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully - shape: '{df.shape}'")
            logger.info(f"Columns found: '{df.columns.tolist()}'")
            return df

        except Exception as e:
            logger.error(
                f"Load data failed - path='{file_path}': {e}",
                exc_info=True,
            )
            raise DataTransformationError("Failed to load data") from e

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names and prepare dataset ."""

        try:
            df = df.copy()
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            logger.info(f"Columns after normalization: '{df.columns.tolist()}'")
            return df

        except Exception as e:
            logger.error(f"Data cleaning failed: {e}", exc_info=True)
            raise DataTransformationError("Failed to clean data") from e

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates rows from the datasets."""

        try:
            duplicate_count = df.duplicated().sum()
            if duplicate_count == 0:
                logger.info("No duplicate rows found")
                return df
            original_shape = df.shape
            df = df.drop_duplicates().reset_index(drop=True)
            logger.info(
                f"Removed '{duplicate_count}' duplicate rows"
                f"Shape changed from '{original_shape}' to '{df.shape}'"
            )
            return df

        except Exception as e:
            logger.error(f"Duplicate removal failed: {e}", exc_info=True)
            raise DataTransformationError("Failed to remove duplicates rows") from e

    def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataset into train and test sets."""

        try:
            test_size = self.params.TRAINING.test_size
            random_state = self.params.TRAINING.random_state
            train, test = train_test_split(
                df, test_size=test_size, random_state=random_state
            )
            logger.info(f"Train dataset shape: '{train.shape}'")
            logger.info(f"Test dataset shape: '{test.shape}'")
            return train, test

        except Exception as e:
            logger.error(f"Data split failed: {e}", exc_info=True)
            raise DataTransformationError("Failed to split data") from e

    def save_data(self, train: pd.DataFrame, test: pd.DataFrame):
        """Save train and test datasets to CSV files."""

        try:
            train.to_csv(self.config.train_file, index=False)
            test.to_csv(self.config.test_file, index=False)

            logger.info(f"Train file saved: '{self.config.train_file}'")
            logger.info(f"Test file saved: '{self.config.test_file}'")

            logger.info(f"Train file size: '{get_size(self.config.train_file)}'")
            logger.info(f"Test file size: '{get_size(self.config.test_file)}'")

        except Exception as e:
            logger.error(f"Data save failed: {e}", exc_info=True)
            raise DataTransformationError("Failed to save transformed data") from e

    def run(self):
        """Run the data transformation process."""

        try:
            df = self.load_data()
            df = self.clean_data(df)
            df = self.remove_duplicates(df)
            train, test = self.split_data(df)
            self.save_data(train, test)
            save_status(self.config.status_file, "data_transformation", True)
            logger.info("Data transformation completed successfully")

        except Exception as e:
            save_status(self.config.status_file, "data_transformation", False)
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise DataTransformationError("Data transformation process failed") from e
