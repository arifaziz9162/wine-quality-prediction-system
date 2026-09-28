import json
import os
import joblib
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any
from box import ConfigBox
from ensure import ensure_annotations
from wine_quality_prediction.logger import WineQualityBaseError, get_logger

logger = get_logger("common", "common.log")


@ensure_annotations
def read_yaml(path: str | Path) -> ConfigBox:
    """Read a YAML file and return as Configbox."""

    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"YAML file does not exist: '{path}'")

        with open(path) as f:
            content = yaml.safe_load(f)
            if content is None:
                raise ValueError(f"YAML file is empty: '{path}'")
            logger.info(f"YAML file loaded successfully: '{path}'")

            return ConfigBox(content)

    except Exception as e:
        logger.error(f"Failed to read YAML file '{path}': {e}", exc_info=True)
        raise WineQualityBaseError(f"Failed to read YAML: '{path}' ") from e


def create_directories(paths: list[str | Path] | None = None, verbose: bool = True):
    """Create directories if they don't exist."""

    try:
        if not paths:
            return
        for path in paths:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at: '{path}'")

    except Exception as e:
        logger.error(f"Failed to create directory '{path}': {e}", exc_info=True)
        raise WineQualityBaseError(f"Failed to create directories: '{path}' ") from e


@ensure_annotations
def save_json(path: str | Path, data: dict):
    """Save dictionary as JSON file."""

    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON file saved at: '{path}'")

    except Exception as e:
        logger.error(f"Failed to save JSON file '{path}': {e}", exc_info=True)
        raise WineQualityBaseError(f"Failed to save JSON: '{path}' ") from e


@ensure_annotations
def load_json(path: str | Path) -> ConfigBox:
    """Load JSON file as ConfigBox."""

    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file does not exist: '{path}'")
        with open(path) as f:
            content = json.load(f)
        logger.info(f"JSON file loaded successfully: '{path}'")

        return ConfigBox(content)

    except Exception as e:
        logger.error(f"Failed to load JSON file '{path}': {e}", exc_info=True)
        raise WineQualityBaseError(f"Failed to load JSON: '{path}' ") from e


def save_bin(data: Any, path: str | Path):
    """Save binary object using joblib."""

    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(value=data, filename=path)
        logger.info(f"Binary file saved at: '{path}'")

    except Exception as e:
        logger.error(f"Failed to save binary file '{path}': {e}", exc_info=True)
        raise WineQualityBaseError(f"Failed to save binary: '{path}' ") from e


def load_bin(path: str | Path) -> Any:
    """Load binary object using joblib."""

    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Binary file does not exist: '{path}'")
        data = joblib.load(path)
        logger.info(f"Binary file loaded successfully: '{path}'")

        return data

    except Exception as e:
        logger.error(f"Failed to load binary file '{path}': {e}", exc_info=True)
        raise WineQualityBaseError(f"Failed to load binary: '{path}' ") from e


@ensure_annotations
def save_status(path: str | Path, stage: str, status: bool):
    """Save pipeline stage status."""

    try:
        path = Path(path)
        save_json(
            path,
            {
                "stage": stage,
                "status": status,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
            },
        )
        logger.info(
            f"Status file saved at: '{path}' "
            f"for stage: '{stage}' "
            f"with status: '{status}'"
        )

    except Exception as e:
        logger.error(f"Failed to save status file '{path}': {e}", exc_info=True)
        raise WineQualityBaseError(f"Failed to save status: '{path}' ") from e


@ensure_annotations
def get_size(path: str | Path) -> str:
    """Get file size in KB."""

    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist at: '{path}'")
        size_in_kb = round(os.path.getsize(path) / 1024)

        return f"~{size_in_kb} KB"

    except Exception as e:
        logger.error(f"Failed to get file size '{path}': {e}", exc_info=True)
        raise WineQualityBaseError(f"Failed to get size: '{path}' ") from e
