"""Constants used by the serving package."""

# Feature names used by the trained model.
# Keep the order the same as the training data.
FEATURE_COLUMNS: tuple[str, ...] = (
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
)

# Minimum score for good and average quality.
GOOD_QUALITY_MIN = 7
AVERAGE_QUALITY_MIN = 5
