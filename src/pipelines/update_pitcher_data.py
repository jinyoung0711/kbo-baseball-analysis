"""Pipelines for downloading and refreshing pitcher datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from src.data_sources.statiz import (
    build_pitching_features,
    fetch_pitcher_basic,
    fetch_pitcher_plus,
)


def _collect_pitcher_tables(seasons: Iterable[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download Statiz pitcher tables for multiple seasons."""
    basic_frames = []
    plus_frames = []

    for season in seasons:
        basic_df = fetch_pitcher_basic(season)
        plus_df = fetch_pitcher_plus(season)
        basic_frames.append(basic_df)
        plus_frames.append(plus_df)

    basic_all = pd.concat(basic_frames, ignore_index=True)
    plus_all = pd.concat(plus_frames, ignore_index=True)
    return basic_all, plus_all


def update_pitcher_data(start_year: int, end_year: int, output_dir: Path | str) -> None:
    """Download pitcher datasets for the requested seasons and persist them as CSV files."""
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seasons = range(start_year, end_year + 1)
    basic_df, plus_df = _collect_pitcher_tables(seasons)
    pitching_features = build_pitching_features(basic_df, plus_df)

    basic_df.to_csv(output_path / "basic.csv", index=False, encoding="utf-8-sig")
    plus_df.to_csv(output_path / "plus.csv", index=False, encoding="utf-8-sig")
    pitching_features.to_csv(output_path / "pitching.csv", index=False, encoding="utf-8-sig")
