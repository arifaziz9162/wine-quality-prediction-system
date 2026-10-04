import json

import pandas as pd
import pytest

from wine_quality_prediction.components import DataValidation
from wine_quality_prediction.entity import DataValidationConfig
from wine_quality_prediction.logger import DataValidationError

SCHEMA = {"fixed acidity": "float64", "volatile acidity": "float64", "quality": "int64"}


@pytest.fixture
def validation_config(tmp_path):
    status_dir = tmp_path / "data_validation"
    status_dir.mkdir(parents=True, exist_ok=True)
    return DataValidationConfig(
        root_dir=tmp_path,
        input_file=tmp_path / "WineQT.csv",
        schema=SCHEMA,
        status_file=status_dir / "status.json",
    )


def make_df(**overrides):
    base = {
        "fixed acidity": [7.4, 7.8, 7.8],
        "volatile acidity": [0.7, 0.88, 0.76],
        "quality": [5, 5, 6],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestValidateSchema:
    def test_passes_when_all_columns_present(self, validation_config):
        validation = DataValidation(config=validation_config)
        result = validation.validate_schema(make_df())
        assert result is True

    def test_fails_when_column_missing(self, validation_config):
        validation = DataValidation(config=validation_config)
        df = make_df().drop(columns=["quality"])
        result = validation.validate_schema(df)
        assert result is False


class TestValidateDtypes:
    def test_passes_when_dtypes_match(self, validation_config):
        validation = DataValidation(config=validation_config)
        result = validation.validate_dtypes(make_df())
        assert result is True

    def test_fails_when_dtype_mismatch(self, validation_config):
        validation = DataValidation(config=validation_config)
        df = make_df(quality=["5", "5", "6"])
        result = validation.validate_dtypes(df)
        assert result is False

    def test_does_not_crash_when_first_column_missing(self, validation_config):
        validation = DataValidation(config=validation_config)
        df = make_df().drop(columns=["fixed acidity"])
        result = validation.validate_dtypes(df)
        assert result is False

    def test_skips_missing_column_without_corrupting_later_checks(
        self, validation_config
    ):
        validation = DataValidation(config=validation_config)
        df = make_df().drop(columns=["fixed acidity"])
        result = validation.validate_dtypes(df)
        assert result is False


class TestValidateMissingValues:
    def test_passes_with_no_nulls(self, validation_config):
        validation = DataValidation(config=validation_config)
        result = validation.validate_missing_values(make_df())
        assert result is True

    def test_fails_with_nulls(self, validation_config):
        validation = DataValidation(config=validation_config)
        df = make_df()
        df.loc[0, "quality"] = None
        result = validation.validate_missing_values(df)
        assert result is False


class TestValidateValueRange:
    def test_passes_within_range(self, validation_config):
        validation = DataValidation(config=validation_config)
        df = make_df()
        df["pH"] = [3.51, 3.2, 3.26]
        df["alcohol"] = [9.4, 9.8, 9.8]
        df["density"] = [0.9978, 0.9968, 0.997]
        result = validation.validate_value_range(df)
        assert result is True

    def test_fails_when_quality_out_of_range(self, validation_config):
        validation = DataValidation(config=validation_config)
        df = make_df()
        df["quality"] = [5, 5, 15]
        result = validation.validate_value_range(df)
        assert result is False

    def test_fails_when_pH_out_of_range(self, validation_config):
        validation = DataValidation(config=validation_config)
        df = make_df()
        df["pH"] = [3.2, 3.3, 20.0]
        result = validation.validate_value_range(df)
        assert result is False

    def test_ignores_columns_not_present(self, validation_config):
        validation = DataValidation(config=validation_config)
        df = make_df()
        result = validation.validate_value_range(df)
        assert result is True


class TestValidateDuplicates:
    def test_warns_but_passes_with_duplicates(self, validation_config):
        validation = DataValidation(config=validation_config)
        df = make_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        result = validation.validate_duplicates(df)
        assert result is True

    def test_passes_with_duplicates(self, validation_config):
        validation = DataValidation(config=validation_config)
        df = make_df()
        result = validation.validate_duplicates(df)
        assert result is True


class TestRun:
    def test_run_writes_success_status_for_clean_data(self, validation_config):
        make_df().to_csv(validation_config.input_file, index=False)
        validation = DataValidation(config=validation_config)
        validation.run()

        status = json.loads(validation_config.status_file.read_text())
        assert status["stage"] == "data_validation"
        assert status["status"] is True

    def test_run_writes_failure_status_for_bad_schema(self, validation_config):
        bad_df = make_df().drop(columns=["quality"])
        bad_df.to_csv(validation_config.input_file, index=False)
        validation = DataValidation(config=validation_config)
        with pytest.raises(DataValidationError):
            validation.run()

        status = json.loads(validation_config.status_file.read_text())
        assert status["stage"] == "data_validation"
        assert status["status"] is False

    def test_run_raises_on_missing_input_file(self, validation_config):
        validation = DataValidation(config=validation_config)
        with pytest.raises(DataValidationError):
            validation.run()
