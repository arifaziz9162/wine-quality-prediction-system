import pytest

from serving.schemas import QualityLabel
from serving.services import get_quality_label


@pytest.mark.parametrize(
    "score, expected",
    [
        (9.0, QualityLabel.good),
        (7.0, QualityLabel.good),
        (6.99, QualityLabel.average),
        (5.0, QualityLabel.average),
        (4.99, QualityLabel.bad),
        (0.0, QualityLabel.bad),
    ],
)
def test_label_boundaries(score, expected):
    assert get_quality_label(score) == expected
