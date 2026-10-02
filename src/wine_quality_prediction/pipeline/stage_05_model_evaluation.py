from wine_quality_prediction.components import ModelEvaluation
from wine_quality_prediction.config import ConfigurationManager
from wine_quality_prediction.logger import get_logger

logger = get_logger("model_evaluation_pipeline", "pipeline.log")
STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationPipeline:
    """Pipeline for evaluating machine learning model."""

    def run(self):
        """Execute model evaluation pipeline."""

        try:
            logger.info(f">>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<")
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.run()
            logger.info(f">>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<")

        except Exception as e:
            logger.error(f"{STAGE_NAME} failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    pipeline = ModelEvaluationPipeline()
    pipeline.run()
