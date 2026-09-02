"""Entrypoint: pull FRED data, backtest, and render the current nowcast.

Usage:
    cd fred-nowcast
    pip install -r requirements.txt
    python run.py
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from data import build_feature_matrix, build_target, build_history
from model import fit_ols, predict, walk_forward_backtest, backtest_summary
from series_config import TARGET, FEATURES, HISTORY_START
import report

MIN_TRAIN_QUARTERS = 40
COVID_SHOCK_QUARTERS = ["2020Q1", "2020Q2", "2020Q3", "2020Q4"]


def _load_api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    raise SystemExit("Set FRED_API_KEY (a free key from the St. Louis Fed).")


def main():
    api_key = _load_api_key()
    feature_ids = list(FEATURES.keys())

    print(f"Fetching {TARGET['series_id']} + {', '.join(feature_ids)} from {HISTORY_START}...")
    target = build_target(TARGET["series_id"], api_key, HISTORY_START)
    features = build_feature_matrix(feature_ids, api_key, HISTORY_START)
    history = build_history(target, features)

    print(f"{len(history)} aligned historical quarters, {len(features)} quarters of feature data.")

    backtest_df = walk_forward_backtest(history, TARGET["series_id"], feature_ids, MIN_TRAIN_QUARTERS)
    summary = backtest_summary(backtest_df)
    summary_ex_shock = backtest_summary(backtest_df, exclude_quarters=COVID_SHOCK_QUARTERS)
    print(f"Backtest (all quarters): model RMSE={summary['model_rmse']:.2f} vs naive RMSE={summary['naive_rmse']:.2f} "
          f"({summary['improvement_pct']:+.0f}% improvement over {summary['n_quarters']} quarters)")
    print(f"Backtest (excl. 2020 COVID quarters): model RMSE={summary_ex_shock['model_rmse']:.2f} vs "
          f"naive RMSE={summary_ex_shock['naive_rmse']:.2f} "
          f"({summary_ex_shock['improvement_pct']:+.0f}% improvement over {summary_ex_shock['n_quarters']} quarters)")

    # fit on ALL available history for the live nowcast
    current_fit = fit_ols(history[feature_ids].values, history[TARGET["series_id"]].values)

    # "Ragged edge" of real-world data: indicators publish on different lags
    # (weekly claims arrive fast, monthly payrolls/retail-sales/housing lag
    # 2-6 weeks), so the most recent quarter often has SOME but not all
    # features yet. Walk back to the latest quarter where every feature is
    # actually populated, rather than nowcasting off a partially-NaN row.
    nowcast_quarter = None
    for q in reversed(features.index):
        if not features.loc[q, feature_ids].isna().any():
            nowcast_quarter = q
            break
    if nowcast_quarter is None:
        raise SystemExit("No quarter has complete feature data - check the FRED pull.")

    has_actual = nowcast_quarter in target.index
    x_row = features.loc[nowcast_quarter, feature_ids].values
    nowcast_value = predict(current_fit, x_row)
    print(f"{nowcast_quarter} nowcast: {nowcast_value:+.2f}% "
          f"({'already released: ' + f'{target.loc[nowcast_quarter]:+.2f}%' if has_actual else 'not yet released'})")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = report.render(
        TARGET["label"], FEATURES, feature_ids, current_fit, backtest_df, summary, summary_ex_shock,
        nowcast_value, str(nowcast_quarter), has_actual, generated_at,
    )

    os.makedirs("reports", exist_ok=True)
    out_path = "reports/nowcast.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
