import os
import yaml 
import json 
import joblib
from ensure import ensure_annotations
from typing import Any, Union, List
from box import ConfigBox
from pathlib import Path
from wine_quality_prediction.logger import get_logger, CustomException, log_exceptions

logger = get_logger("common", "common.log")

@ensure_annotations
@log_exceptions
def read_yaml(path: Union[str, Path]) -> ConfigBox:
    """Read a YAML file and return as Configbox"""
    path = Path(path)
    if not path.exists():
        raise CustomException(f"YAML file does not exist: '{path}'")    
    with open(path) as f:
        content = yaml.safe_load(f)
        if content is None:
            raise CustomException(f"YAML file is empty: '{path}'")            
        logger.info(f"YAML file {path} loaded successfully")
        return ConfigBox(content)
    
@ensure_annotations
@log_exceptions
def create_directories(paths: List[Union[str, Path]] = None, verbose : bool = True):
    """Create directories if they don't exist"""
    if not paths:
        return 
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: '{path}'")

@ensure_annotations
@log_exceptions
def save_json(path: Union[str, Path], data: dict):
    """Save dictionary as JSON file"""
    path = Path(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: '{path}'")

@ensure_annotations
@log_exceptions
def load_json(path: Union[str, Path]) -> ConfigBox:
    """Load JSON file as ConfigBox"""
    path = Path(path)
    if not path.exists():
        raise CustomException(f"JSON file does not exist: '{path}'")     
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully: '{path}'")
    return ConfigBox(content)

@ensure_annotations
@log_exceptions
def save_bin(data : Any, path: Union[str, Path]):
    """Save binary object using joblib"""
    path = Path(path)
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: '{path}'")

@ensure_annotations
@log_exceptions
def get_size(path: Union[str, Path]) -> str:
    """Get file size in KB"""
    path = Path(path)
    if not path.exists():
        raise CustomException(f"File does not exist at: '{path}'")
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~{size_in_kb} KB"