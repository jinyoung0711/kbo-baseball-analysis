"""Utilities for loading park factor data used in pitching metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

# Default location of the park factor csv produced from Statiz data exports.
_DEFAULT_PARK_FACTOR_PATH = Path(__file__).resolve().parents[2] / "baseball_data" / "WHIP_park.csv"


def load_park_factors(path: Union[str, Path] = _DEFAULT_PARK_FACTOR_PATH) -> pd.DataFrame:
    """Load the park factor table from the given ``path``.

    Parameters
    ----------
    path
        Path to the CSV file. Defaults to ``baseball_data/WHIP_park.csv`` in the
        repository root.

    Returns
    -------
    pandas.DataFrame
        Park factor table with validated column names and appropriate dtypes.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at ``path``.
    ValueError
        If required columns are missing from the CSV file.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Park factor CSV not found: {csv_path}")

    park_df = pd.read_csv(csv_path)

    required_columns = {"팀", "시즌"}
    missing_columns = required_columns.difference(park_df.columns)
    if missing_columns:
        raise ValueError(
            "Park factor CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Normalize data types for consistency in downstream merges
    park_df["시즌"] = park_df["시즌"].astype(int)
    park_df["팀"] = park_df["팀"].astype(str)

    return park_df


__all__ = ["load_park_factors"]
