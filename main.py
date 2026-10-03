from wine_quality_prediction.logger import get_logger
from wine_quality_prediction.pipeline.stage_01_data_ingestion import (
    DataIngestionPipeline,
)
from wine_quality_prediction.pipeline.stage_02_data_validation import (
    DataValidationPipeline,
)
from wine_quality_prediction.pipeline.stage_03_data_transformation import (
    DataTransformationPipeline,
)
from wine_quality_prediction.pipeline.stage_04_model_trainer import (
    ModelTrainingPipeline,
)
from wine_quality_prediction.pipeline.stage_05_model_evaluation import (
    ModelEvaluationPipeline,
)
from wine_quality_prediction.pipeline.stage_06_model_registry import (
    ModelRegistryPipeline,
)
from wine_quality_prediction.pipeline.stage_07_model_packaging import (
    ModelPackagingPipeline,
)

logger = get_logger("main", "main.log")

if __name__ == "__main__":
    try:
        logger.info("========== Pipeline Started ==========")

        logger.info("Data Ingestion Stage Started")
        DataIngestionPipeline().run()
        logger.info("Data Ingestion Stage Completed\n")

        logger.info("Data Validation Stage Started")
        DataValidationPipeline().run()
        logger.info("Data Validation Stage Completed\n")

        logger.info("Data Transformation Stage Started")
        DataTransformationPipeline().run()
        logger.info("Data Transformation Stage Completed\n")

        logger.info("Model Trainer Stage Started")
        ModelTrainingPipeline().run()
        logger.info("Model Trainer Stage Completed\n")

        logger.info("Model Evaluation Stage Started")
        ModelEvaluationPipeline().run()
        logger.info("Model Evaluation Stage Completed\n")

        logger.info("Model Registry Stage Started")
        ModelRegistryPipeline().run()
        logger.info("Model Registry Stage Completed\n")

        logger.info("Model Packaging Stage Started")
        ModelPackagingPipeline().run()
        logger.info("Model Packaging Stage completed\n")

        logger.info("========== Pipeline completed ==========")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise
