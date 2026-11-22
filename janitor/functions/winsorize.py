"""Implementation of the `winsorize` function."""

from __future__ import annotations

from typing import Hashable, Iterable

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from janitor.utils import check, check_column


@pf.register_dataframe_method
def winsorize(
    df: pd.DataFrame,
    columns: Hashable | Iterable[Hashable] | None = None,
    lower: float = 0.05,
    upper: float = 0.95,
    suffix: str | None = None,
) -> pd.DataFrame:
    """Cap extreme values in numeric columns at chosen quantiles.

    Winsorizing replaces every value below the `lower` quantile with the
    value of that quantile, and every value above the `upper` quantile with
    the value of that quantile. Unlike dropping outliers, it keeps every row,
    so it is a common cleaning step before computing means or fitting models
    on data with a few extreme values (for example, order amounts or
    sensor readings).

    Quantiles are computed per column and ignore missing values.
    Missing values stay missing.

    This method does not mutate the original DataFrame.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {"amount": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000], "id": list("abcdefghij")}
        ... )
        >>> df.winsorize(lower=0.1, upper=0.9)
           amount id
        0     1.9  a
        1     2.0  b
        2     3.0  c
        3     4.0  d
        4     5.0  e
        5     6.0  f
        6     7.0  g
        7     8.0  h
        8     9.0  i
        9   108.1  j

        Keep the original column and write the capped values to a new one.

        >>> df.winsorize(columns="amount", lower=0, upper=0.9, suffix="_capped")
           amount id  amount_capped
        0       1  a            1.0
        1       2  b            2.0
        2       3  c            3.0
        3       4  d            4.0
        4       5  e            5.0
        5       6  f            6.0
        6       7  g            7.0
        7       8  h            8.0
        8       9  i            9.0
        9    1000  j          108.1

    Args:
        df: A pandas DataFrame.
        columns: A column label or an iterable of column labels to
            winsorize. Defaults to all numeric (non-boolean) columns.
        lower: Lower quantile, between 0 and 1. Values below this quantile
            are raised to it. Use 0 to leave the lower tail unchanged.
        upper: Upper quantile, between 0 and 1. Values above this quantile
            are lowered to it. Use 1 to leave the upper tail unchanged.
        suffix: Optional suffix for new columns holding the winsorized
            values. If omitted, the original columns are overwritten.

    Raises:
        TypeError: If `lower` or `upper` is not a number, `suffix` is not a
            string, or a selected column is not numeric.
        ValueError: If `lower` or `upper` is outside [0, 1], if `lower` is
            greater than `upper`, or if a new column name already exists.

    Returns:
        A pandas DataFrame with extreme values capped.
    """
    check("df", df, [pd.DataFrame])
    for name, value in (("lower", lower), ("upper", upper)):
        if isinstance(value, bool):
            raise TypeError(f"`{name}` must be a number, not bool.")
        check(name, value, [int, float])
        if not 0 <= value <= 1:
            raise ValueError(f"`{name}` must be between 0 and 1, got {value}.")
    if lower > upper:
        raise ValueError(
            f"`lower` ({lower}) must not be greater than `upper` ({upper})."
        )
    if suffix is not None:
        check("suffix", suffix, [str])

    if columns is None:
        selected = list(df.select_dtypes(include="number").columns)
    elif isinstance(columns, (str, tuple)) or not isinstance(columns, Iterable):
        selected = [columns]
    else:
        selected = list(dict.fromkeys(columns))

    if selected:
        check_column(df, selected)

    for col in selected:
        if not is_numeric_dtype(df[col]) or is_bool_dtype(df[col]):
            raise TypeError(
                f"Column '{col}' must be numeric to winsorize, "
                f"got dtype {df[col].dtype}."
            )

    result = df.copy()
    for col in selected:
        series = result[col]
        low, high = series.quantile([lower, upper])
        target = col if suffix is None else f"{col}{suffix}"
        if suffix is not None:
            check_column(result, target, present=False)
        result[target] = series.clip(lower=low, upper=high)

    return result
