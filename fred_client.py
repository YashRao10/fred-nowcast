"""Minimal FRED API client.
generalized to pull a full history from a start date rather than the last N
observations (a nowcasting backtest needs decades of quarters, not ~14 readings).
"""
from __future__ import annotations

import pandas as pd
import requests

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_series(series_id: str, api_key: str, start: str) -> pd.Series:
    """Fetch every observation for a series from `start` onward, oldest first.

    Returns a pd.Series indexed by date (dtype float), missing values
    ("." in FRED's format) dropped.
    """
    resp = requests.get(
        FRED_BASE_URL,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
            "sort_order": "asc",
            "limit": 100000,
        },
        timeout=20,
    )
    resp.raise_for_status()
    observations = resp.json()["observations"]
    dates, values = [], []
    for o in observations:
        if o["value"] == ".":
            continue
        dates.append(o["date"])
        values.append(float(o["value"]))
    idx = pd.to_datetime(dates)
    return pd.Series(values, index=idx, name=series_id)
