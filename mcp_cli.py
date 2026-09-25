"""JSON-emitting entrypoint for the Financial Analysis MCP server — same
pipeline as run.py, but returns one structured JSON object on stdout instead
of writing an HTML report, so a subprocess caller (the MCP server) gets
clean data back.

Usage:
    python mcp_cli.py
"""
from __future__ import annotations

import json
import os
import sys

from data import build_feature_matrix, build_target, build_history
from model import fit_ols, predict, walk_forward_backtest, backtest_summary
from series_config import TARGET, FEATURES, HISTORY_START

MIN_TRAIN_QUARTERS = 40
COVID_SHOCK_QUARTERS = ["2020Q1", "2020Q2", "2020Q3", "2020Q4"]


def _load_api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    raise SystemExit("Set FRED_API_KEY (a free key from the St. Louis Fed).")


def main() -> None:
    try:
        api_key = _load_api_key()
    except SystemExit as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)

    feature_ids = list(FEATURES.keys())
    target = build_target(TARGET["series_id"], api_key, HISTORY_START)
    features = build_feature_matrix(feature_ids, api_key, HISTORY_START)
    history = build_history(target, features)

    backtest_df = walk_forward_backtest(history, TARGET["series_id"], feature_ids, MIN_TRAIN_QUARTERS)
    summary = backtest_summary(backtest_df)
    summary_ex_shock = backtest_summary(backtest_df, exclude_quarters=COVID_SHOCK_QUARTERS)

    current_fit = fit_ols(history[feature_ids].values, history[TARGET["series_id"]].values)

    # Same "ragged edge" walk-back as run.py: find the latest quarter where
    # every feature is actually populated, since indicators publish on
    # different lags.
    nowcast_quarter = None
    for q in reversed(features.index):
        if not features.loc[q, feature_ids].isna().any():
            nowcast_quarter = q
            break
    if nowcast_quarter is None:
        print(json.dumps({"error": "No quarter has complete feature data. Check the FRED pull."}))
        sys.exit(1)

    has_actual = nowcast_quarter in target.index
    x_row = features.loc[nowcast_quarter, feature_ids].values
    nowcast_value = predict(current_fit, x_row)

    result = {
        "target": TARGET["label"],
        "nowcast_quarter": str(nowcast_quarter),
        "nowcast_value_pct": round(float(nowcast_value), 2),
        "already_released": has_actual,
        "actual_value_pct": round(float(target.loc[nowcast_quarter]), 2) if has_actual else None,
        "backtest_all_quarters": {k: round(v, 3) if isinstance(v, float) else v for k, v in summary.items()},
        "backtest_excl_covid_quarters": {
            k: round(v, 3) if isinstance(v, float) else v for k, v in summary_ex_shock.items()
        },
        "features_used": FEATURES,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
