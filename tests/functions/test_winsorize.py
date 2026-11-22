"""Tests for the winsorize function."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

import janitor  # noqa: F401


@pytest.fixture
def df():
    """DataFrame with one outlier column, one clean column and a text column."""
    return pd.DataFrame(
        {
            "amount": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000],
            "score": [10.0, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "id": list("abcdefghij"),
        }
    )


@pytest.mark.functions
def test_winsorize_caps_both_tails(df):
    """Values outside the quantiles are replaced by the quantile values."""
    result = df.winsorize(columns="amount", lower=0.1, upper=0.9)

    low, high = df["amount"].quantile([0.1, 0.9])
    assert result["amount"].min() == pytest.approx(low)
    assert result["amount"].max() == pytest.approx(high)
    assert result["amount"].max() == pytest.approx(108.1)


@pytest.mark.functions
def test_winsorize_middle_values_unchanged(df):
    """Values already inside the quantile range are left as they were."""
    result = df.winsorize(columns="amount", lower=0.1, upper=0.9)

    assert_series_equal(
        result["amount"].iloc[1:9],
        df["amount"].iloc[1:9].astype(float),
    )


@pytest.mark.functions
def test_winsorize_default_selects_numeric_columns(df):
    """With no columns given, all numeric columns are capped; text is untouched."""
    result = df.winsorize(lower=0.1, upper=0.9)

    assert result["amount"].max() == pytest.approx(108.1)
    assert result["score"].min() == pytest.approx(19.0)
    assert_series_equal(result["id"], df["id"])


@pytest.mark.functions
def test_winsorize_does_not_mutate_input(df):
    """The input DataFrame is unchanged and a new object is returned."""
    original = df.copy()

    result = df.winsorize()

    assert result is not df
    assert_frame_equal(df, original)


@pytest.mark.functions
def test_winsorize_suffix_creates_new_column(df):
    """A suffix keeps the original column and adds the capped one."""
    result = df.winsorize(columns=["amount"], upper=0.9, suffix="_capped")

    assert_series_equal(result["amount"], df["amount"])
    assert "amount_capped" in result.columns
    assert result["amount_capped"].max() == pytest.approx(108.1)


@pytest.mark.functions
def test_winsorize_full_range_is_noop(df):
    """lower=0 and upper=1 leave the values unchanged."""
    result = df.winsorize(columns="score", lower=0, upper=1)

    assert_series_equal(result["score"], df["score"])


@pytest.mark.functions
def test_winsorize_preserves_missing_values():
    """Missing values stay missing and are ignored when computing quantiles."""
    data = pd.DataFrame({"x": [1.0, np.nan, 2.0, 3.0, 100.0]})

    result = data.winsorize(lower=0, upper=0.75)

    assert pd.isna(result.loc[1, "x"])
    assert result["x"].max() == pytest.approx(data["x"].quantile(0.75))


@pytest.mark.functions
def test_winsorize_all_missing_column():
    """An all-missing column is returned unchanged instead of raising."""
    data = pd.DataFrame({"x": [np.nan, np.nan]})

    result = data.winsorize()

    assert result["x"].isna().all()


@pytest.mark.functions
def test_winsorize_empty_dataframe():
    """An empty DataFrame is returned as an empty DataFrame."""
    data = pd.DataFrame({"x": pd.Series([], dtype=float)})

    result = data.winsorize()

    assert result.empty


@pytest.mark.functions
@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"lower": -0.1}, ValueError, "between 0 and 1"),
        ({"upper": 1.5}, ValueError, "between 0 and 1"),
        ({"lower": 0.9, "upper": 0.1}, ValueError, "must not be greater"),
        ({"lower": "0.1"}, TypeError, "lower"),
        ({"upper": True}, TypeError, "upper"),
        ({"suffix": 1}, TypeError, "suffix"),
        ({"columns": "id"}, TypeError, "must be numeric"),
        ({"columns": "missing"}, ValueError, "missing"),
    ],
)
def test_winsorize_invalid_arguments(df, kwargs, error, match):
    """Invalid quantiles, suffixes and column choices raise clear errors."""
    with pytest.raises(error, match=match):
        df.winsorize(**kwargs)


@pytest.mark.functions
def test_winsorize_suffix_name_collision(df):
    """A suffix that produces an existing column name raises an error."""
    data = df.assign(amount_capped=0)

    with pytest.raises(ValueError):
        data.winsorize(columns="amount", suffix="_capped")


@pytest.mark.functions
def test_winsorize_boolean_column_rejected():
    """Boolean columns are not treated as numeric data to cap."""
    data = pd.DataFrame({"flag": [True, False, True]})

    with pytest.raises(TypeError, match="must be numeric"):
        data.winsorize(columns="flag")
    assert_frame_equal(data.winsorize(), data)
