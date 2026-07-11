"""Data validation tests: schema, nulls, value ranges."""
import pandas as pd
import pytest

EXPECTED_COLUMNS = {
    "sepal_length": "float64",
    "sepal_width": "float64",
    "petal_length": "float64",
    "petal_width": "float64",
    "target": "int64",
}

# Reasonable bounds for IRIS measurements (cm), with a little headroom
VALUE_RANGES = {
    "sepal_length": (3.0, 9.0),
    "sepal_width": (1.5, 5.5),
    "petal_length": (0.5, 8.0),
    "petal_width": (0.0, 3.5),
}

VALID_TARGETS = {0, 1, 2}


@pytest.fixture(params=["data/train.csv", "data/eval.csv"])
def dataset(request):
    return request.param, pd.read_csv(request.param)


def test_file_not_empty(dataset):
    path, df = dataset
    assert len(df) > 0, f"{path} is empty"


def test_expected_columns_present(dataset):
    path, df = dataset
    missing = set(EXPECTED_COLUMNS.keys()) - set(df.columns)
    assert not missing, f"{path} missing columns: {missing}"


def test_column_dtypes(dataset):
    path, df = dataset
    for col, expected_dtype in EXPECTED_COLUMNS.items():
        actual_dtype = str(df[col].dtype)
        assert actual_dtype == expected_dtype, (
            f"{path}: column '{col}' has dtype {actual_dtype}, expected {expected_dtype}"
        )


def test_no_missing_values(dataset):
    path, df = dataset
    nulls = df[list(EXPECTED_COLUMNS.keys())].isnull().sum()
    assert nulls.sum() == 0, f"{path} has missing values:\n{nulls[nulls > 0]}"


def test_no_duplicate_rows(dataset):
    path, df = dataset
    dupes = df.duplicated().sum()
    assert dupes == 0, f"{path} has {dupes} duplicate rows"


@pytest.mark.parametrize("col,bounds", VALUE_RANGES.items())
def test_value_ranges(dataset, col, bounds):
    path, df = dataset
    low, high = bounds
    out_of_range = df[(df[col] < low) | (df[col] > high)]
    assert len(out_of_range) == 0, (
        f"{path}: column '{col}' has {len(out_of_range)} values outside [{low}, {high}]"
    )


def test_target_values_valid(dataset):
    path, df = dataset
    invalid = set(df["target"].unique()) - VALID_TARGETS
    assert not invalid, f"{path}: target has invalid values: {invalid}"


def test_train_eval_no_overlap():
    """A leakage check: train and eval shouldn't share the same original record (by row_id)."""
    train_df = pd.read_csv("data/train.csv")
    eval_df = pd.read_csv("data/eval.csv")
    overlap = set(train_df["row_id"]) & set(eval_df["row_id"])
    assert not overlap, f"Found {len(overlap)} overlapping row_ids between train and eval: {overlap}"