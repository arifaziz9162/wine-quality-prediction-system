from wine_quality_prediction.components import ModelRegistry
from wine_quality_prediction.config import ConfigurationManager
from wine_quality_prediction.logger import get_logger

logger = get_logger("model_registry_pipeline", "pipeline.log")
STAGE_NAME = "Model Registry Stage"


class ModelRegistryPipeline:
    """Pipeline for registry machine learning model."""

    def run(self):
        """Execute model registry pipeline."""

        try:
            logger.info(f">>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<")
            config = ConfigurationManager()
            model_registry_config = config.get_model_registry_config()
            model_registry = ModelRegistry(config=model_registry_config)
            model_registry.run()
            logger.info(f">>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<")

        except Exception as e:
            logger.error(f"{STAGE_NAME} failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    pipeline = ModelRegistryPipeline()
    pipeline.run()
