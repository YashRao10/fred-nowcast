"""Pulls the target + feature series from FRED and builds a quarterly,
aligned feature matrix.

Every feature (whatever its native frequency — monthly, weekly) is reduced to
one number per quarter: the quarter's average level, then converted to a
quarter-over-quarter % change — the same units as the GDP growth target, so
regression coefficients are comparable growth-rate-on-growth-rate elasticities
rather than mixing raw index points against a percentage.
"""
from __future__ import annotations

import pandas as pd

from fred_client import fetch_series


def _quarterly_growth(raw: pd.Series) -> pd.Series:
    """Resample any frequency to quarterly mean level, then QoQ % change."""
    quarterly_level = raw.resample("QE").mean()
    growth = quarterly_level.pct_change() * 100
    growth.index = quarterly_level.index.to_period("Q")
    return growth.dropna()


def build_feature_matrix(feature_ids: list[str], api_key: str, start: str) -> pd.DataFrame:
    """Quarterly growth for every feature, independent of whether the target
    has been released yet for the most recent quarter — this is what makes a
    genuine nowcast possible (features arrive faster than GDP does)."""
    columns = {fid: _quarterly_growth(fetch_series(fid, api_key, start)) for fid in feature_ids}
    return pd.DataFrame(columns).sort_index()


def build_target(target_id: str, api_key: str, start: str) -> pd.Series:
    target = fetch_series(target_id, api_key, start)
    target.index = target.index.to_period("Q")
    target.name = target_id
    return target.sort_index()


def build_history(target: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
    """Inner-joins target + features to every quarter BOTH have data for —
    the training/backtest set. A partial row would otherwise silently let a
    regression train on a quarter where one indicator was actually missing."""
    df = features.copy()
    df[target.name] = target
    return df.dropna(how="any").sort_index()
