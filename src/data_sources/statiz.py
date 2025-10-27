"""Utilities for downloading pitching statistics from Statiz."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "http://www.statiz.co.kr/stat.php"


@dataclass(frozen=True)
class StatizTableConfig:
    """Configuration for a table that can be fetched from Statiz."""

    columns: Sequence[str]
    row_length: int
    footer_rows: int
    sort_field: str
    secondary_sort_field: str
    data_code: int


def _request_table(params: dict) -> BeautifulSoup:
    """Retrieve an HTML page from Statiz and return the parsed soup."""
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def _extract_table_rows(soup: BeautifulSoup, row_length: int) -> List[List[str]]:
    """Extract table rows from the Statiz HTML response."""
    rows: List[List[str]] = []
    for tr in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) == row_length:
            rows.append(cells)
    return rows


def _build_params(season: int, config: StatizTableConfig) -> dict:
    """Build query parameters for the Statiz endpoint."""
    return {
        "opt": 0,
        "sopt": 0,
        "re": 1,
        "ys": season,
        "ye": season,
        "se": 0,
        "te": "",
        "tm": "",
        "ty": 0,
        "qu": "p50",
        "po": 0,
        "as": "",
        "ae": "",
        "hi": "",
        "un": "",
        "pl": "",
        "da": config.data_code,
        "o1": config.sort_field,
        "o2": config.secondary_sort_field,
        "de": 0,
        "lr": 0,
        "tr": "",
        "cv": "",
        "ml": 1,
        "sn": 100,
        "si": "",
        "cn": "",
    }


def _fetch_table(season: int, config: StatizTableConfig) -> pd.DataFrame:
    params = _build_params(season, config)
    soup = _request_table(params)
    rows = _extract_table_rows(soup, config.row_length)
    df = pd.DataFrame(rows, columns=config.columns)
    if config.footer_rows:
        df = df.iloc[:-config.footer_rows]
    # Remove ranking column and duplicated metrics columns.
    df = df.drop(df.columns[0], axis=1)
    df = df.loc[:, ~df.T.duplicated(keep="first")]
    df.insert(0, "시즌", season)
    return df.reset_index(drop=True)


BASIC_CONFIG = StatizTableConfig(
    columns=[
        "순",
        "이름",
        "팀",
        "ERA",
        "출장",
        "완투",
        "완봉",
        "선발",
        "승",
        "패",
        "세",
        "홀드",
        "이닝",
        "실점",
        "자책",
        "타자",
        "안타",
        "2타",
        "3타",
        "홈런",
        "볼넷",
        "고4",
        "사구",
        "삼진",
        "보크",
        "폭투",
        "ERA",
        "FIP",
        "WHIP",
        "ERA+",
        "FIP+",
        "WAR",
        "WPA",
    ],
    row_length=33,
    footer_rows=7,
    sort_field="ERAP",
    secondary_sort_field="OutCount",
    data_code=1,
)

PLUS_CONFIG = StatizTableConfig(
    columns=[
        "순",
        "이름",
        "팀",
        "FIP",
        "출장",
        "이닝",
        "ERA",
        "FIP",
        "K/9",
        "BB/9",
        "K/BB",
        "HR/9",
        "K%",
        "BB%",
        "K-BB%",
        "PFR",
        "BABIP",
        "LOB%",
        "타율(상대)",
        "출루율(상대)",
        "장타율(상대)",
        "OPS(상대)",
        "WHIP",
        "WHIP+",
        "투구",
        "IP/G",
        "P/G",
        "P/IP",
        "P/PA",
        "CYP+",
    ],
    row_length=30,
    footer_rows=8,
    sort_field="FIP",
    secondary_sort_field="WAR",
    data_code=2,
)


def fetch_pitcher_basic(season: int) -> pd.DataFrame:
    """Fetch basic pitcher statistics for the given KBO season."""
    return _fetch_table(season, BASIC_CONFIG)


def fetch_pitcher_plus(season: int) -> pd.DataFrame:
    """Fetch advanced pitcher statistics for the given KBO season."""
    return _fetch_table(season, PLUS_CONFIG)


def build_pitching_features(
    basic_df: pd.DataFrame, plus_df: pd.DataFrame
) -> pd.DataFrame:
    """Combine basic and plus pitcher tables into a modelling dataset."""
    merged = basic_df[
        [
            "시즌",
            "이름",
            "WHIP",
            "자책",
            "삼진",
            "WPA",
            "폭투",
            "사구",
        ]
    ].copy()

    plus_features = plus_df[
        [
            "시즌",
            "이름",
            "K%",
            "BABIP",
            "LOB%",
            "타율(상대)",
            "OPS(상대)",
        ]
    ].copy()

    result = pd.merge(merged, plus_features, on=["시즌", "이름"], how="inner")

    # Ensure consistent dtypes for downstream use.
    for column in result.columns:
        if column in {"시즌", "이름"}:
            continue
        cleaned = result[column].astype(str).str.replace("%", "", regex=False)
        result[column] = pd.to_numeric(cleaned, errors="coerce")
    return result
