"""Metrics for tracking distribution of incoming feature values."""

from collections.abc import Mapping

import bentoml

from serving.constants import FEATURE_COLUMNS

FEATURE_BUCKETS: dict[str, tuple[float, ...]] = {
    "fixed_acidity": (5, 6, 7, 8, 9, 10, 12, 14, 16),
    "volatile_acidity": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.2, 1.6),
    "citric_acid": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
    "residual_sugar": (1.5, 2, 2.5, 3, 4, 6, 10, 16),
    "chlorides": (0.04, 0.06, 0.08, 0.1, 0.15, 0.3, 0.6),
    "free_sulfur_dioxide": (5, 10, 15, 20, 30, 40, 55, 75),
    "total_sulfur_dioxide": (20, 40, 60, 80, 100, 150, 200, 290),
    "density": (0.992, 0.994, 0.996, 0.997, 0.998, 1.0, 1.004),
    "ph": (3.0, 3.2, 3.3, 3.4, 3.5, 3.7, 4.1),
    "sulphates": (0.45, 0.55, 0.65, 0.75, 0.9, 1.2, 2.0),
    "alcohol": (9, 9.5, 10, 10.5, 11, 12, 13, 15),
}

FEATURE_HISTOGRAMS = {
    feature: bentoml.metrics.Histogram(
        name=f"wine_input_{feature}",
        documentation=f"Distribution of incoming '{feature}' values.",
        buckets=FEATURE_BUCKETS[feature],
    )
    for feature in FEATURE_COLUMNS
}


def observe_features(values: Mapping[str, float]) -> None:
    """Records the feature values from a prediction request."""

    for feature, histogram in FEATURE_HISTOGRAMS.items():
        histogram.observe(float(values[feature]))
