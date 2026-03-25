import os
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from wine_quality_prediction.logger import get_logger, log_exceptions, DatabaseException

# Load env variables
load_dotenv()

# Create logger
logger = get_logger("db_connection", "database.log")

@log_exceptions
def get_connection():
    """Create and return a PostgreSQL connection."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        logger.info("Connected to database successfully.")
        return conn
    except Exception as e:
        raise DatabaseException("Failed to connect database", e) from e
    
@log_exceptions
def get_engine():
    """Create and return a SQLALchemy engine for pandas operations."""
    try:
        engine = create_engine(
            f"postgresql+psycopg2://"
            f"{os.getenv('DB_USER')}:{quote_plus(os.getenv('DB_PASSWORD'))}"
            f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}"
            f"/{os.getenv('DB_NAME')}"
        )
        logger.info("SQLALchemy engine created successfully.")
        return engine
    except Exception as e:
        raise DatabaseException("Failed to create SQLALchemy engine", e) from e