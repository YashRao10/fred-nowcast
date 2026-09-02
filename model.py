"""A small, transparent bridge-equation nowcast: plain OLS regressing GDP
growth on a handful of higher-frequency leading indicators' own growth rates.

No dynamic factor model, no vintage/real-time data (see README's "what this
tool won't do for you") — deliberately the simplest honest version of the
idea, in the same spirit as the sibling research tools.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_ols(X: np.ndarray, y: np.ndarray) -> dict:
    """Plain OLS with intercept. Returns coefficients, R^2, and residual std
    (used for a rough prediction-interval width on the live nowcast)."""
    X_design = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    fitted = X_design @ coef
    resid = y - fitted
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "intercept": float(coef[0]),
        "coefficients": coef[1:],
        "r_squared": r_squared,
        "resid_std": float(np.std(resid, ddof=X_design.shape[1])),
        "n_obs": len(y),
    }


def predict(fit: dict, x_row: np.ndarray) -> float:
    return float(fit["intercept"] + np.dot(fit["coefficients"], x_row))


def walk_forward_backtest(df: pd.DataFrame, target_col: str, feature_cols: list[str], min_train: int = 20) -> pd.DataFrame:
    """Expanding-window out-of-sample backtest: for each quarter after
    `min_train` quarters of history exist, fit ONLY on strictly prior
    quarters and predict that quarter — no lookahead. Compared against a
    naive baseline (this quarter's growth = last quarter's actual growth),
    the standard bar a nowcast has to clear to be worth anything.
    """
    rows = []
    for i in range(min_train, len(df)):
        train = df.iloc[:i]
        test = df.iloc[i]
        fit = fit_ols(train[feature_cols].values, train[target_col].values)
        model_pred = predict(fit, test[feature_cols].values)
        naive_pred = df.iloc[i - 1][target_col]
        rows.append({
            "quarter": df.index[i],
            "actual": test[target_col],
            "model_pred": model_pred,
            "naive_pred": naive_pred,
        })
    return pd.DataFrame(rows).set_index("quarter")


def backtest_summary(backtest_df: pd.DataFrame, exclude_quarters: list[str] | None = None) -> dict:
    """RMSE of model vs. naive over the backtest window.

    `exclude_quarters` (e.g. the 2020 COVID quarters) lets the caller also
    report a version with historic macro-shock outliers removed — a naive
    "repeat last quarter" baseline is catastrophically wrong across a crash
    /rebound like 2020Q1-Q4, which can make almost any model relying on
    contemporaneous indicators look artificially good by comparison. Both
    numbers are worth showing rather than letting the inflated one stand
    alone (see fred-nowcast's README).
    """
    df = backtest_df
    if exclude_quarters:
        df = df[~df.index.astype(str).isin(exclude_quarters)]
    model_rmse = float(np.sqrt(np.mean((df["actual"] - df["model_pred"]) ** 2)))
    naive_rmse = float(np.sqrt(np.mean((df["actual"] - df["naive_pred"]) ** 2)))
    return {
        "model_rmse": model_rmse,
        "naive_rmse": naive_rmse,
        "improvement_pct": (naive_rmse - model_rmse) / naive_rmse * 100 if naive_rmse else 0.0,
        "n_quarters": len(df),
    }
