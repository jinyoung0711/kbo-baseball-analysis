"""Client utilities to download Statiz data tables."""

from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional, Sequence
from urllib.parse import urljoin

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://www.statiz.co.kr/"


def fetch_statiz_table(
    endpoint: str,
    params: Optional[Mapping[str, str]] = None,
    *,
    base_url: str = DEFAULT_BASE_URL,
    table_index: int = 0,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch an HTML table from Statiz and return it as a dataframe.

    Parameters
    ----------
    endpoint
        Relative path (e.g. ``"stat.php"``) within the Statiz site.
    params
        Query parameters appended to the request.
    base_url
        Base Statiz URL. This is overridable to simplify testing.
    table_index
        Index of the table to parse from the HTML response.
    session
        Optional :class:`requests.Session` to reuse connections.
    """

    if not endpoint:
        raise ValueError("endpoint must be a non-empty string")

    request_session = session or requests.Session()
    url = urljoin(base_url, endpoint)
    LOGGER.debug("Fetching Statiz table from %s with params %s", url, params)

    response = request_session.get(url, params=params, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(response.text)
    if not tables:
        raise ValueError(f"No tables found at {url}")

    if table_index >= len(tables):
        raise IndexError(
            f"Requested table_index {table_index} but only {len(tables)} tables were returned"
        )

    dataframe = tables[table_index]
    LOGGER.debug("Retrieved Statiz table with shape %s", dataframe.shape)
    return dataframe


def fetch_defense_metrics(
    season: int,
    team: Optional[str] = None,
    level: str = "team",
    *,
    endpoint: str = "stat.php",
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch defensive metrics table from Statiz.

    Parameters
    ----------
    season
        Season to download.
    team
        Optional team code/name used by Statiz to filter results.
    level
        ``"team"`` for team aggregates or ``"player"`` for individual players.
    endpoint
        Statiz endpoint path.
    session
        Optional requests session.
    """

    if level not in {"team", "player"}:
        raise ValueError("level must be either 'team' or 'player'")

    params: Dict[str, str] = {
        "opt": "1",
        "sopt": "0" if level == "team" else "1",
        "re": "0",
        "ys": str(season),
        "ye": str(season),
        "position": "2",  # 2 corresponds to defence tables in Statiz
    }
    if team:
        params["te"] = team

    df = fetch_statiz_table(endpoint, params=params, session=session)
    df["시즌"] = season
    if team:
        df["팀"] = team
    return df


def fetch_park_factor_table(
    season: int,
    *,
    endpoint: str = "park.php",
    params: Optional[Mapping[str, str]] = None,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch park factor data for the provided season directly from Statiz."""

    query: Dict[str, str] = {"year": str(season)}
    if params:
        query.update({key: str(value) for key, value in params.items()})

    df = fetch_statiz_table(endpoint, params=query, session=session)
    df["시즌"] = season
    return df


__all__ = [
    "DEFAULT_BASE_URL",
    "fetch_defense_metrics",
    "fetch_park_factor_table",
    "fetch_statiz_table",
]
