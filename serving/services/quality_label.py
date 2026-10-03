from serving.constants import AVERAGE_QUALITY_MIN, GOOD_QUALITY_MIN
from serving.schemas import QualityLabel


def get_quality_label(score: float) -> QualityLabel:
    """Convert a numeric score to a quality label."""

    if score >= GOOD_QUALITY_MIN:
        return QualityLabel.good
    if score >= AVERAGE_QUALITY_MIN:
        return QualityLabel.average
    return QualityLabel.bad
