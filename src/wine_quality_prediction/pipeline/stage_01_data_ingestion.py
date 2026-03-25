from wine_quality_prediction.config.configuration import ConfigurationManager
from wine_quality_prediction.components.data_ingestion import DataIngestion
from wine_quality_prediction.logger import get_logger

logger = get_logger("pipeline", "pipeline.log")
STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def run(self):
        logger.info(f">>>>>>> {STAGE_NAME} started <<<<<<<")
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.run()
        logger.info(f">>>>>>> {STAGE_NAME} completed <<<<<<<")