import pandas as pd

from wine_quality_prediction.entity import DataValidationConfig
from wine_quality_prediction.logger import DataValidationError, get_logger
from wine_quality_prediction.utils import get_size, save_status

logger = get_logger("data_validation", "data_validation.log")


class DataValidation:
    """Handles dataset validation checks."""

    def __init__(self, config: DataValidationConfig):
        """Initialize data validation with configuration."""

        self.config = config

    def load_data(self) -> pd.DataFrame:
        """Load input dataset from CSV file."""

        try:
            file_path = self.config.input_file
            logger.info(f"Reading file: '{file_path}'")
            logger.info(f"File size: '{get_size(file_path)}'")
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully with shape: {df.shape}'")
            logger.info(f"Columns found: '{df.columns.tolist()}'")
            return df

        except Exception as e:
            logger.error(
                f"Load data failed - path='{self.config.input_file}': {e}",
                exc_info=True,
            )
            raise DataValidationError("Failed to load data") from e

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Check that all expected columns are present."""

        try:
            expected_columns = list(self.config.schema.keys())
            missing_columns = [
                column for column in expected_columns if column not in df.columns
            ]
            if missing_columns:
                logger.error(
                    f"Schema validation failed - missing columns: '{missing_columns}'"
                )
                return False
            logger.info("Schema validation passed.")
            return True

        except Exception as e:
            logger.error(f"Schema validation error: {e}", exc_info=True)
            raise DataValidationError("Schema validation failed") from e

    def validate_dtypes(self, df: pd.DataFrame) -> bool:
        """Check data types of expected columns."""

        try:
            all_passed = True

            for column, expected_dtype in self.config.schema.items():
                if column not in df.columns:
                    logger.error(
                        f"Dtype validation failed - missing column: '{column}'"
                    )
                    all_passed = False
                    continue

                actual_dtype = str(df[column].dtype)

                if actual_dtype != expected_dtype:
                    logger.error(
                        f"Dtype mismatch - column='{column}' "
                        f"expected='{expected_dtype}' got='{actual_dtype}'"
                    )
                    all_passed = False

            if all_passed:
                logger.info("Data type validation passed.")

            return all_passed

        except Exception as e:
            logger.error(f"Data type validation error: {e}", exc_info=True)
            raise DataValidationError("Data type validation failed") from e

    def validate_missing_values(self, df: pd.DataFrame) -> bool:
        """Check that the dataset contains no missing values."""

        try:
            null_count = df.isnull().sum()
            columns_with_nulls = null_count[null_count > 0]
            if len(columns_with_nulls) > 0:
                logger.error(
                    f"Missing value validation failed - "
                    f"null values found:\n'{columns_with_nulls}'"
                )
                return False
            logger.info("Missing value validation passed.")
            return True

        except Exception as e:
            logger.error(f"Missing value validation error: {e}", exc_info=True)
            raise DataValidationError("Missing value validation failed") from e

    def validate_value_range(self, df: pd.DataFrame) -> bool:
        """Check value ranges for selected columns."""

        try:
            all_passed = True
            ranges = {
                "quality": (0, 10),
                "pH": (0, 14),
                "alcohol": (0, 100),
                "density": (0, 2),
            }
            for column, (min_val, max_val) in ranges.items():
                if column in df.columns:
                    out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
                    if len(out_of_range) > 0:
                        logger.error(
                            f"Range validation failed - col='{column}' "
                            f"range='{(min_val, max_val)}' "
                            f"violations='{len(out_of_range)}'"
                        )
                        all_passed = False
            if all_passed:
                logger.info("Value range validation passed.")
            return all_passed

        except Exception as e:
            logger.error(f"Value range validation error: {e}", exc_info=True)
            raise DataValidationError("Value range validation failed") from e

    def validate_duplicates(self, df: pd.DataFrame) -> bool:
        """Check duplicate rows in the dataset."""

        try:
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                logger.warning(
                    f"Duplicate rows found - '{duplicate_count}' duplicate rows"
                )
                # Duplicates are handled during data transformation
                return True
            logger.info("Duplicate validation passed.")
            return True

        except Exception as e:
            logger.error(f"Duplicate validation error: {e}", exc_info=True)
            raise DataValidationError("Duplicate validation failed") from e

    def run(self):
        """Run the data validation process."""

        try:
            df = self.load_data()
            schema_check = self.validate_schema(df)
            dtype_check = self.validate_dtypes(df)
            missing_check = self.validate_missing_values(df)
            range_check = self.validate_value_range(df)
            duplicate_check = self.validate_duplicates(df)
            overall_status = all(
                [schema_check, dtype_check, missing_check, range_check, duplicate_check]
            )
            save_status(self.config.status_file, "data_validation", overall_status)
            if overall_status:
                logger.info("Data validation completed successfully.")
            else:
                logger.error("Data validation failed.")

        except Exception as e:
            save_status(self.config.status_file, "data_validation", False)
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise DataValidationError("Data validation process failed") from e

        if not overall_status:
            raise DataValidationError(
                "Validation checks failed, stopping the pipeline here"
            )
