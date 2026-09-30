import os
from urllib.parse import quote_plus

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

from wine_quality_prediction.logger import DatabaseError, get_logger

# Load env variables
load_dotenv()

# Create logger
logger = get_logger("db_connection", "database.log")


def get_connection():
    """Create and return a PostgreSQL connection."""

    try:
        conn = psycopg2.connect(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
        )
        logger.info("Connected to database successfully.")
        return conn

    except Exception as e:
        logger.error(
            f"Database connection failed - "
            f"host='{os.getenv('DB_HOST')}' "
            f"db='{os.getenv('DB_NAME')}': {e}",
            exc_info=True,
        )
        raise DatabaseError("Failed to connect to database") from e


def get_engine():
    """Create and return a SQLALchemy engine for pandas operations."""

    try:
        engine = create_engine(
            f"postgresql+psycopg2://"
            f"{os.getenv('DB_USER')}:"
            f"{quote_plus(os.getenv('DB_PASSWORD'))}@"
            f"{os.getenv('DB_HOST')}:"
            f"{os.getenv('DB_PORT')}/"
            f"{os.getenv('DB_NAME')}"
        )
        logger.info("SQLALchemy engine created successfully.")
        return engine

    except Exception as e:
        logger.error(
            f"SQLALchemy engine creation failed - "
            f"host='{os.getenv('DB_HOST')}' "
            f"db='{os.getenv('DB_NAME')}': {e}",
            exc_info=True,
        )
        raise DatabaseError("Failed to create SQLALchemy engine") from e
