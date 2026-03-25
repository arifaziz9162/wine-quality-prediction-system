import pandas as pd
from .connection import get_connection, get_engine

class DatabaseOperations:
    """Handles database operations for wine_data."""

    def __init__(self):
        """Initialize DB connection and cursor."""
        self.conn = get_connection()
        self.engine = get_engine()
        self.cur = self.conn.cursor()

    def create_table(self):
        """Create wine_data table if not exists."""
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS wine_data (
                id SERIAL PRIMARY KEY,
                fixed_acidity FLOAT,
                volatile_acidity FLOAT,
                citric_acid FLOAT,
                residual_sugar FLOAT,
                chlorides FLOAT,
                free_sulfur_dioxide FLOAT,
                total_sulfur_dioxide FLOAT,
                density FLOAT,
                pH FLOAT,
                sulphates FLOAT,
                alcohol FLOAT,
                quality INT,
                id INT
            );
            """
        )
        self.conn.commit()

    def insert_dataframe(self, df):
        """Insert raw dataframe rows into wine_data table."""
        query = """
            INSERT INTO wine_data (
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
                quality,
                id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        data = [tuple(row) for row in df.itertuples(index=False, name=None)]
        self.cur.executemany(query, data)
        self.conn.commit()
        
    def fetch_data(self):
        """Fetch all data from wine_data table."""
        return pd.read_sql("SELECT * FROM wine_data", self.engine)

    def clear_table(self):
        """Delete all data from wine_data table."""
        self.cur.execute("DELETE FROM wine_data")
        self.conn.commit()

    def create_prediction_table(self):
        """Create predictions table if not exists."""
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                fixed_acidity FLOAT,
                volatile_acidity FLOAT,
                citric_acid FLOAT,
                residual_sugar FLOAT,
                chlorides FLOAT,
                free_sulfur_dioxide FLOAT,
                total_sulfur_dioxide FLOAT,
                density FLOAT,
                pH FLOAT,
                sulphates FLOAT,
                alcohol FLOAT,
                predicted_quality FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.commit()

    def save_prediction(self, input_data, prediction):
        """Insert a prediction result into predictions table."""
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
            (*input_data, prediction)
        )
        self.conn.commit()

    def fetch_predictions(self):
        """Fetch all data from predictions table."""
        return pd.read_sql("SELECT * FROM predictions", self.engine)

    def close_connection(self):
        """Close database connection and engine."""
        self.cur.close()
        self.conn.close()
        self.engine.dispose()