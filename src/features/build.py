"""Utilities to build a consolidated pitcher feature table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .park_factors import load_park_factors
from .preprocess import normalize_numeric_columns, validate_dataframe
from .statiz import fetch_defense_metrics


@dataclass
class BuildConfig:
    """Configuration for building the pitcher feature table."""

    season: int
    basic_path: Path
    plus_path: Path
    park_factor_path: Path
    output_dir: Path
    normalization_method: str = "zscore"
    include_defense: bool = True


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def _merge_feature_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None
    for frame in frames:
        if frame is None or frame.empty:
            continue
        key_columns = [col for col in ("이름", "팀", "시즌") if col in frame.columns]
        if merged is None:
            merged = frame.copy()
        else:
            merged = merged.merge(frame, on=key_columns, how="left")
    if merged is None:
        raise ValueError("No data frames were provided for merging")
    return merged


def build_feature_table(
    season: int,
    *,
    basic_path: Optional[Path] = None,
    plus_path: Optional[Path] = None,
    park_factor_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    normalization_method: str = "zscore",
    include_defense: bool = True,
) -> Path:
    """Build the consolidated pitcher feature table and store it as parquet.

    Parameters
    ----------
    season
        Season to build features for.
    basic_path, plus_path, park_factor_path
        Optional overrides for the CSV locations. By default these point to the
        ``baseball_data`` folder in the repository.
    output_dir
        Directory to store the generated parquet files. Defaults to
        ``data/processed`` within the repository.
    normalization_method
        Method passed to :func:`normalize_numeric_columns`.
    include_defense
        Whether to download and merge Statiz defensive metrics.

    Returns
    -------
    Path
        Location of the generated parquet file.
    """

    repo_root = Path(__file__).resolve().parents[2]
    basic_csv = basic_path or (repo_root / "baseball_data" / "basic.csv")
    plus_csv = plus_path or (repo_root / "baseball_data" / "plus.csv")
    park_csv = park_factor_path or (repo_root / "baseball_data" / "WHIP_park.csv")
    output_directory = output_dir or (repo_root / "data" / "processed")
    output_directory.mkdir(parents=True, exist_ok=True)

    basic_df = _load_csv(basic_csv)
    plus_df = _load_csv(plus_csv)
    park_df = load_park_factors(park_csv)

    frames = [basic_df, plus_df, park_df]
    defense_df = None
    if include_defense:
        try:
            defense_df = fetch_defense_metrics(season, level="player")
        except Exception as exc:  # pragma: no cover - network error path
            raise RuntimeError("Failed to download Statiz defensive metrics") from exc
        else:
            frames.append(defense_df)

    merged_df = _merge_feature_frames(frames)

    required_columns = ["이름", "팀", "시즌"]
    validate_dataframe(merged_df, required_columns, key_columns=required_columns)

    numeric_columns = [
        column
        for column in merged_df.select_dtypes(include=["number"]).columns
        if column not in required_columns
    ]
    processed_df = normalize_numeric_columns(
        merged_df,
        columns=numeric_columns,
        method=normalization_method,
    )

    version_date = datetime.utcnow().date().isoformat()
    processed_df["데이터_버전"] = version_date
    processed_df["시즌"] = season

    output_path = output_directory / f"pitchers_{season}_{version_date}.parquet"
    processed_df.to_parquet(output_path, index=False)

    return output_path


__all__ = ["BuildConfig", "build_feature_table"]
