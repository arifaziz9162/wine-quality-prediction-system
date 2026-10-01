from wine_quality_prediction.components import DataTransformation
from wine_quality_prediction.config import ConfigurationManager
from wine_quality_prediction.logger import get_logger

logger = get_logger("data_transformation_pipeline", "pipeline.log")
STAGE_NAME = "Data Transformation Stage"


class DataTransformationPipeline:
    """Pipeline for transforming training data."""

    def run(self):
        """Execute data transformation pipeline."""

        try:
            logger.info(f">>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<")
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(
                config=data_transformation_config, params=config.params
            )
            data_transformation.run()
            logger.info(f">>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<")

        except Exception as e:
            logger.error(f"{STAGE_NAME} failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    pipeline = DataTransformationPipeline()
    pipeline.run()
