"""Preprocessing utilities for Statiz feature tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import pandas as pd


@dataclass
class ValidationResult:
    """Summary of validation checks performed on a dataframe."""

    missing_columns: List[str]
    missing_values: List[str]
    duplicate_rows: int

    @property
    def is_valid(self) -> bool:
        """Return ``True`` when all checks pass."""

        return not self.missing_columns and not self.missing_values and self.duplicate_rows == 0


class ValidationError(RuntimeError):
    """Raised when validation on a dataframe fails."""


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    key_columns: Optional[Sequence[str]] = None,
    raise_on_error: bool = True,
) -> ValidationResult:
    """Validate the dataframe for required columns, nulls, and duplicate keys.

    Parameters
    ----------
    df
        DataFrame to validate.
    required_columns
        Columns that must exist in the dataframe and have no null values.
    key_columns
        Optional list of columns used to check for duplicate rows.
    raise_on_error
        Whether to raise :class:`ValidationError` when checks fail.

    Returns
    -------
    ValidationResult
        Object summarising the validation results.
    """

    missing_columns = [col for col in required_columns if col not in df.columns]

    missing_values = [
        col for col in required_columns if col in df.columns and df[col].isna().any()
    ]

    duplicate_rows = 0
    if key_columns:
        duplicate_rows = int(df.duplicated(subset=list(key_columns)).sum())

    result = ValidationResult(missing_columns, missing_values, duplicate_rows)

    if raise_on_error and not result.is_valid:
        messages: List[str] = []
        if result.missing_columns:
            messages.append(
                "Missing columns: " + ", ".join(sorted(result.missing_columns))
            )
        if result.missing_values:
            messages.append(
                "Columns contain null values: " + ", ".join(sorted(result.missing_values))
            )
        if result.duplicate_rows:
            messages.append(f"Found {result.duplicate_rows} duplicate rows for keys {key_columns}")
        raise ValidationError("; ".join(messages))

    return result


def normalize_numeric_columns(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    method: str = "zscore",
) -> pd.DataFrame:
    """Normalize numeric columns using the selected method.

    Parameters
    ----------
    df
        DataFrame containing numeric columns to normalise.
    columns
        Iterable of column names to normalise. When ``None`` all numeric columns except
        object/categorical columns are normalised.
    method
        Normalisation method. Supported values are ``"zscore"`` and ``"minmax"``.
    """

    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns

    normalized_df = df.copy()

    for column in columns:
        if column not in normalized_df.columns:
            continue
        series = normalized_df[column]
        if method == "zscore":
            std = series.std(ddof=0)
            normalized_df[column] = 0.0 if std == 0 else (series - series.mean()) / std
        elif method == "minmax":
            min_value = series.min()
            max_value = series.max()
            range_value = max_value - min_value
            normalized_df[column] = 0.0 if range_value == 0 else (series - min_value) / range_value
        else:
            raise ValueError(f"Unsupported normalisation method: {method}")

    return normalized_df


__all__ = [
    "ValidationError",
    "ValidationResult",
    "normalize_numeric_columns",
    "validate_dataframe",
]
