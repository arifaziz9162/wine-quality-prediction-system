import pandas as pd

from wine_quality_prediction.logger import DatabaseError, get_logger

from .connection import get_connection, get_engine

# Create logger
logger = get_logger("db_operations", "database.log")


class DatabaseOperations:
    """Handles database operations for wine_data."""

    def __init__(self):
        """Initialize DB connection and cursor."""

        self.conn = get_connection()
        self.engine = get_engine()
        self.cur = self.conn.cursor()

    def insert_dataframe(self, df):
        """Insert raw dataframe rows into wine_data table."""

        try:
            query = """
                INSERT INTO wine_data (
                    fixed_acidity,
                    volatile_acidity,
                    citric_acid,
                    residual_sugar,
                    chlorides,
                    free_sulfur_dioxide,
                    total_sulfur_dioxide,
                    density, pH,
                    sulphates,
                    alcohol,
                    quality
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            data = [tuple(row) for row in df.itertuples(index=False, name=None)]
            self.cur.executemany(query, data)
            self.conn.commit()
            logger.info(f"Inserted {len(df)} rows into wine_data.")

        except Exception as e:
            logger.error(f"Failed to inserted rows - {len(df)}: {e}", exc_info=True)
            raise DatabaseError("Failed to insert dataframe") from e

    def fetch_data(self):
        """Fetch all data from wine_data table."""

        try:
            return pd.read_sql("SELECT * FROM wine_data", self.engine)

        except Exception as e:
            logger.error(f"Failed to fetch data from wine_data: {e}", exc_info=True)
            raise DatabaseError("Failed to fetch data") from e

    def clear_table(self):
        """Delete all data from wine_data table."""

        try:
            self.cur.execute("DELETE FROM wine_data")
            self.conn.commit()
            logger.info("Table wine_data cleared.")

        except Exception as e:
            logger.error(f"Failed to clear wine_table: {e}", exc_info=True)
            raise DatabaseError("Failed to clear table") from e

    def save_prediction(self, input_data, prediction):
        """Insert a prediction result into predictions table."""

        try:
            self.cur.execute(
                """
                INSERT INTO predictions (
                    fixed_acidity,
                    volatile_acidity,
                    citric_acid,
                    residual_sugar,
                    chlorides,
                    free_sulfur_dioxide,
                    total_sulfur_dioxide,
                    density,
                    pH,
                    sulphates,
                    alcohol,
                    predicted_quality
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (*input_data, prediction),
            )
            self.conn.commit()
            logger.info("Prediction saved successfully.")

        except Exception as e:
            logger.error(f"Failed to save prediction: {e}", exc_info=True)
            raise DatabaseError("Failed to save prediction") from e

    def fetch_predictions(self):
        """Fetch all data from predictions table."""

        try:
            return pd.read_sql("SELECT * FROM predictions", self.engine)

        except Exception as e:
            logger.error(f"Failed to fetch prediction: {e}", exc_info=True)
            raise DatabaseError("Failed to fetch prediction") from e

    def close_connection(self):
        """Close database connection and engine."""

        try:
            self.cur.close()
            self.conn.close()
            self.engine.dispose()
            logger.info("Database connection closed.")

        except Exception as e:
            logger.error(f"Failed to close database connection: {e}", exc_info=True)
            raise DatabaseError("Failed to close database connection") from e
