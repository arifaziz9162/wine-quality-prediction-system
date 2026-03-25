import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s')

project_name = "wine_quality_prediction"

# Folders inside src
src_folders = [
    f"src/{project_name}",
    f"src/{project_name}/components",
    f"src/{project_name}/utils",
    f"src/{project_name}/config",
    f"src/{project_name}/pipeline",
    f"src/{project_name}/entity",
    f"src/{project_name}/constants",
    f"src/{project_name}/logger",
    f"src/{project_name}/database"
]

# Folders outside src
other_folders = [
    "config",
    "research",
    "data",
    "logs"
]

# __init__.py only inside src folders
list_of_files = [f"{folder}/__init__.py" for folder in src_folders]

# Extra files
extra_files = [
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/logger/logger_config.py",
    f"src/{project_name}/logger/exception.py",
    f"src/{project_name}/database/connection.py",
    f"src/{project_name}/database/operations.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/pipeline/stage_01_data_ingestion.py",


    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",

    "research/trails.ipynb",
    "research/test.yaml",
    "research/Experiments.ipynb",
    "research/01_data_ingestion.ipynb"
]

list_of_files += extra_files

# Create outside folders
for folder in other_folders:
    os.makedirs(folder, exist_ok=True)
    logging.info(f"Creating folder: '{folder}'")

# Create files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: '{filedir}' for file: '{filename}'")

    if (not filepath.exists()) or (filepath.stat().st_size == 0):
        with open(filepath, "w"):
            pass
        logging.info(f"Creating empty file: '{filepath}'")

    else:
        logging.info(f"File already exists: '{filepath}'")