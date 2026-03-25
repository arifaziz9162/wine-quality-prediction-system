from wine_quality_prediction.logger import get_logger
from wine_quality_prediction.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

logger = get_logger("main", "main.log")

try:
    logger.info("========== Pipeline Started ==========")

    logger.info(">>>>>>> Data Ingestion Stage Started <<<<<<<")
    DataIngestionTrainingPipeline().run()
    logger.info(">>>>>>> Data Ingestion Stage Completed <<<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e