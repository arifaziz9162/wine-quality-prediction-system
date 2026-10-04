import json

import pandas as pd
import pytest

from wine_quality_prediction.components import DataTransformation
from wine_quality_prediction.entity import DataTransformationConfig
from wine_quality_prediction.logger import DataTransformationError


class FakeTrainingParams:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state


class FakeParams:
    def __init__(self, test_size=0.2, random_state=42):
        self.TRAINING = FakeTrainingParams(test_size, random_state)


@pytest.fixture
def transformation_config(tmp_path):
    status_dir = tmp_path / "data_transformation"
    status_dir.mkdir(parents=True, exist_ok=True)
    return DataTransformationConfig(
        root_dir=tmp_path,
        input_file=tmp_path / "WineQT.csv",
        train_file=tmp_path / "train.csv",
        test_file=tmp_path / "test.csv",
        status_file=status_dir / "status.json",
    )


@pytest.fixture
def raw_df():
    rows = [
        {
            "fixed acidity": 7.0 + i * 0.1,
            "volatile acidity": 0.5,
            "quality": 5 + (i % 3),
        }
        for i in range(95)
    ]
    df = pd.DataFrame(rows)
    duplicate_rows = df.iloc[:5].copy()
    return pd.concat([df, duplicate_rows], ignore_index=True)


class TestCleanData:
    def test_normalizes_column_names(self, transformation_config, raw_df):
        transformation = DataTransformation(
            config=transformation_config, params=FakeParams()
        )
        cleaned = transformation.clean_data(raw_df)

        assert list(cleaned.columns) == ["fixed_acidity", "volatile_acidity", "quality"]

    def test_does_not_mutate_original_df(self, transformation_config, raw_df):
        transformation = DataTransformation(
            config=transformation_config, params=FakeParams()
        )
        original_columns = list(raw_df.columns)
        transformation.clean_data(raw_df)

        assert list(raw_df.columns) == original_columns


class TestRemoveDuplicates:
    def test_removes_exact_duplicates(self, transformation_config, raw_df):
        transformation = DataTransformation(transformation_config, params=FakeParams())
        cleaned = transformation.clean_data(raw_df)
        deduped = transformation.remove_duplicates(cleaned)

        assert deduped.duplicated().sum() == 0
        assert len(deduped) == 95

    def test_noop_when_no_duplicates(self, transformation_config):
        transformation = DataTransformation(
            config=transformation_config, params=FakeParams()
        )
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = transformation.remove_duplicates(df)

        assert len(result) == 3


class TestSplitData:
    def test_split_ratio_matches_params(self, transformation_config, raw_df):
        transformation = DataTransformation(
            config=transformation_config, params=FakeParams()
        )

        cleaned = transformation.remove_duplicates(transformation.clean_data(raw_df))
        train, test = transformation.split_data(cleaned)
        total = len(cleaned)

        assert len(test) == pytest.approx(total * 0.2, abs=1)
        assert len(train) + len(test) == total

    def test_split_is_reproducible_with_same_random_state(
        self, transformation_config, raw_df
    ):
        transformation_1 = DataTransformation(
            config=transformation_config, params=FakeParams()
        )
        transformation_2 = DataTransformation(
            config=transformation_config, params=FakeParams()
        )

        cleaned = transformation_1.remove_duplicates(
            transformation_1.clean_data(raw_df)
        )

        train1, test1 = transformation_1.split_data(cleaned)
        train2, test2 = transformation_2.split_data(cleaned)

        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_no_row_overlap_between_train_and_test(self, transformation_config, raw_df):
        params = FakeParams(test_size=0.25, random_state=1)
        transformation = DataTransformation(config=transformation_config, params=params)

        cleaned = transformation.remove_duplicates(
            transformation.clean_data(raw_df).reset_index(drop=True)
        )
        train, test = transformation.split_data(cleaned)

        assert set(train.index).isdisjoint(set(test.index))

    def test_different_test_size_changes_split(self, transformation_config, raw_df):
        transformation_small = DataTransformation(
            config=transformation_config,
            params=FakeParams(test_size=0.1, random_state=42),
        )

        transformation_large = DataTransformation(
            config=transformation_config,
            params=FakeParams(test_size=0.4, random_state=42),
        )

        cleaned = transformation_small.remove_duplicates(
            transformation_small.clean_data(raw_df)
        )

        _, test_small = transformation_small.split_data(cleaned)
        _, test_large = transformation_large.split_data(cleaned)

        assert len(test_large) > len(test_small)


class TestSaveData:
    def test_saves_train_and_test_csvs(self, transformation_config, raw_df):
        transformation = DataTransformation(
            config=transformation_config, params=FakeParams()
        )

        cleaned = transformation.remove_duplicates(transformation.clean_data(raw_df))
        train, test = transformation.split_data(cleaned)

        transformation.save_data(train, test)

        assert transformation_config.train_file.exists()
        assert transformation_config.test_file.exists()
        assert len(pd.read_csv(transformation_config.train_file)) == len(train)
        assert len(pd.read_csv(transformation_config.test_file)) == len(test)


class TestRun:
    def test_run_end_to_end_writes_success_status(self, transformation_config, raw_df):
        raw_df.to_csv(transformation_config.input_file, index=False)
        transformation = DataTransformation(
            config=transformation_config, params=FakeParams()
        )
        transformation.run()

        status = json.loads(transformation_config.status_file.read_text())
        assert status["stage"] == "data_transformation"
        assert status["status"] is True
        assert transformation_config.train_file.exists()
        assert transformation_config.test_file.exists()

    def test_run_writes_failure_status_on_missing_input_file(
        self, transformation_config
    ):
        transformation = DataTransformation(
            config=transformation_config, params=FakeParams()
        )

        with pytest.raises(DataTransformationError):
            transformation.run()

        status = json.loads(transformation_config.status_file.read_text())
        assert status["stage"] == "data_transformation"
        assert status["status"] is False
