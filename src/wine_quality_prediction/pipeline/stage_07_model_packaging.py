from wine_quality_prediction.components import ModelPackaging
from wine_quality_prediction.config import ConfigurationManager
from wine_quality_prediction.logger import get_logger

logger = get_logger("model_packaging_pipeline", "pipeline.log")

STAGE_NAME = "Model Packaging Stage"


class ModelPackagingPipeline:
    """Pipeline for packaging the Production model for BentoML."""

    def __init__(self):
        self.config = ConfigurationManager()
        self.model_packaging_config = self.config.get_model_packaging_config()

    def run(self):
        try:
            logger.info(f">>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<")

            model_packaging = ModelPackaging(config=self.model_packaging_config)

            model_packaging.run()

            logger.info(f">>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<")

        except Exception as e:
            logger.error(f"{STAGE_NAME} failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    pipeline = ModelPackagingPipeline()
    pipeline.run()
