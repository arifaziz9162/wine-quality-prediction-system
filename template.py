import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
    f"src/{project_name}/database",
]

# Folders outside src
other_folders = [
    "config",
    "notebooks",
    "backend",
    "frontend",
    "reports",
    "reports/figures",
    "docs",
    "docs/images",
]

# __init__.py only inside src folders
list_of_files = [f"{folder}/__init__.py" for folder in src_folders]

# Extra files
extra_files = [
    # utils
    f"src/{project_name}/utils/common.py",
    # config
    f"src/{project_name}/config/configuration.py",
    # entity
    f"src/{project_name}/entity/config_entity.py",
    # logger
    f"src/{project_name}/logger/logger_config.py",
    f"src/{project_name}/logger/exception.py",
    # database
    f"src/{project_name}/database/connection.py",
    f"src/{project_name}/database/operations.py",
    # components
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/model_registry.py",
    # pipeline
    f"src/{project_name}/pipeline/stage_01_data_ingestion.py",
    f"src/{project_name}/pipeline/stage_02_data_validation.py",
    f"src/{project_name}/pipeline/stage_03_data_transformation.py",
    f"src/{project_name}/pipeline/stage_04_model_trainer.py",
    f"src/{project_name}/pipeline/stage_05_model_evaluation.py",
    f"src/{project_name}/pipeline/stage_06_model_registry.py",
    # config files
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    # root files
    "main.py",
    "requirements-dev.txt",
    "requirements.txt",
    "pyproject.toml",
    "dvc.yaml",
    ".dvcignore",
    "docker-compose.dev.yml",
    "docker-compose.prod.yml",
    # notebooks
    "notebooks/Experiments.ipynb",
    "notebooks/01_data_ingestion.ipynb",
    "notebooks/02_data_validation.ipynb",
    "notebooks/03_data_transformation.ipynb",
    "notebooks/04_model_trainer.ipynb",
    "notebooks/05_model_evaluation.ipynb",
    # reports
    "reports/evalution_report.ipynb",
    # backend
    "backend/__init__.py",
    "backend/app.py",
    "backend/api/__init__.py",
    "backend/api/v1/__init__.py",
    "backend/api/v1/routes.py",
    "backend/core/__init__.py",
    "backend/core/config.py",
    "backend/core/exception_handler.py",
    "backend/services/__init__.py",
    "backend/services/prediction_service.py",
    "backend/models/__init__.py",
    "backend/models/model_loader.py",
    "backend/schemas/__init__.py",
    "backend/schemas/request_schema.py",
    "backend/schemas/response_schema.py",
    "backend/schemas/health_schema.py",
    "backend/tests/test_prediction.py",
    "backend/Dockerfile.dev",
    "backend/Dockerfile",
    "backend/.dockerignore",
    # frontend
    "frontend/templates/index.html",
    "frontend/templates/result.html",
    "frontend/static/css/main.css",
    "frontend/static/css/layout.css",
    "frontend/static/css/forms.css",
    "frontend/static/css/responsive.css",
    "frontend/static/css/animations.css",
    "frontend/static/js/app.js",
    "frontend/static/js/api.js",
    "frontend/static/js/ui.js",
    "frontend/static/js/validation.js",
    "frontend/static/js/config.js",
    "frontend/static/assets/images/logo.png",
    "frontend/static/assets/images/background.jpg",
    "frontend/static/assets/icons/about.txt",
    "frontend/static/assets/icons/android-chrome-192x192.png",
    "frontend/static/assets/icons/android-chrome-512x512.png",
    "frontend/static/assets/icons/apple-touch-icon.png",
    "frontend/static/assets/icons/favicon-16x16.png",
    "frontend/static/assets/icons/favicon-32x32.png",
    "frontend/static/assets/icons/favicon.ico",
    "frontend/static/assets/icons/site.webmanifest",
    "frontend/nginx.conf",
    "frontend/Dockerfile.dev",
    "frontend/Dockerfile",
    "frontend/.dockerignore",
    # github CI/CD
    ".github/dependabot.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/docker.yml",
    ".github/workflows/model-training.yml",
    ".github/workflows/security.yml",
    ".github/workflows/deploy.yml",
    ".github/workflows/release.yml",
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
