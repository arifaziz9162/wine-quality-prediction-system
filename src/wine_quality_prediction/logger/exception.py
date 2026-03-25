from .logger_config import get_logger
from typing import Callable

logger = get_logger("exception", "exception.log")

class CustomException(Exception):
    """
    Custom exception class for the wine-quality-prediction project.
    Wraps the original exception and logs it automatically.
    """

    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception

        # log the exception automatically
        if original_exception:
            logger.error(f"{message} | Original: {repr(original_exception)}", exc_info=True)
        else:
            logger.error(message)

    def __str__(self):
        if self.original_exception:
            return f"{super().__str__()} | Original: {repr(self.original_exception)}"
        return super().__str__()
    
class DataValidationException(CustomException):
    """Raised when input data fails validation checks."""
    pass


class ModelTrainingException(CustomException):
    """Raised when model training fails."""
    pass


class ModelInferenceException(CustomException):
    """Raised when prediction/inference fails."""
    pass

class DatabaseException(CustomException):
    """Raised when any database related operation fails."""
    pass

def log_exceptions(func: Callable):
    """
    Decorator to automatically catch and log exceptions using CustomException.
    Any exception raised inside the decorator fuction will be logged. 
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CustomException:
            # Already logged
            raise
        except Exception as e:
            # Wrap and log
            raise CustomException(f"Error occurred in function '{func.__name__}'", e) from e
    return wrapper