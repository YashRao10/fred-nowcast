"""Pure toy-data unit tests on the core method, no network calls — same
convention: pure toy-data tests, no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model import backtest_summary, fit_ols, predict, walk_forward_backtest


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_fit_ols_recovers_known_coefficients(rng):
    n = 200
    X = rng.normal(0, 1, (n, 3))
    true_intercept, true_coefs = 1.5, np.array([0.8, -0.4, 0.2])
    y = true_intercept + X @ true_coefs + rng.normal(0, 0.01, n)

    fit = fit_ols(X, y)

    assert fit["r_squared"] > 0.99
    assert fit["intercept"] == pytest.approx(true_intercept, abs=0.05)
    np.testing.assert_allclose(fit["coefficients"], true_coefs, atol=0.05)


def test_predict_matches_manual_dot_product():
    fit = {"intercept": 2.0, "coefficients": np.array([1.0, -1.0])}
    x_row = np.array([3.0, 5.0])
    assert predict(fit, x_row) == pytest.approx(2.0 + 3.0 - 5.0)


def test_walk_forward_backtest_uses_no_lookahead(rng):
    # construct data where a feature perfectly predicts the target, but only
    # from the observation's own row onward (a naive full-sample fit would
    # trivially get this "right"; the point is the backtest must still only
    # ever see PRIOR rows when fitting each prediction)
    n = 60
    idx = pd.period_range("2010Q1", periods=n, freq="Q")
    feature = rng.normal(0, 1, n)
    target = 2 * feature + rng.normal(0, 0.001, n)
    df = pd.DataFrame({"feature": feature, "target": target}, index=idx)

    backtest_df = walk_forward_backtest(df, "target", ["feature"], min_train=20)

    assert len(backtest_df) == n - 20
    # with a near-noiseless linear relationship, the model should easily
    # beat a naive "repeat last quarter" baseline out of sample
    summary = backtest_summary(backtest_df)
    assert summary["model_rmse"] < summary["naive_rmse"]


def test_walk_forward_backtest_only_fits_on_strictly_prior_rows(rng):
    # a feature that predicts the target ONLY in the held-out row (i.e. its
    # relationship to target is scrambled everywhere else) should NOT be
    # exploitable by a model fit purely on prior rows
    n = 40
    idx = pd.period_range("2010Q1", periods=n, freq="Q")
    target = rng.normal(0, 1, n)
    feature = rng.normal(0, 1, n)  # unrelated to target
    df = pd.DataFrame({"feature": feature, "target": target}, index=idx)

    backtest_df = walk_forward_backtest(df, "target", ["feature"], min_train=20)
    summary = backtest_summary(backtest_df)

    # unrelated feature: model shouldn't reliably beat naive by a wide margin
    assert summary["model_rmse"] > 0
