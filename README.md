# FRED Nowcast — a small, transparent GDP bridge-equation model

A small GDP-nowcasting model built to pair machine-learning practice with an economics
minor, using the standard FRED leading-indicator series.

Nowcasts real GDP growth (the classic nowcasting use case — GDP is only released quarterly,
with a real lag) from five higher-frequency leading/coincident indicators, via a plain OLS
bridge equation. Same honesty conventions used throughout this work:
small and transparent over sophisticated and opaque, and an explicit check against a naive
baseline before claiming the model is worth anything.

## Quickstart

```bash
cd fred-nowcast
pip install -r requirements.txt
python run.py
```

Uses `FRED_API_KEY` from the environment (a free key from the St. Louis Fed). Writes `reports/nowcast.html`: the current
quarter's live nowcast, a backtest chart (actual vs. model vs. naive baseline), the fitted
coefficients, and a written read.

## The method

1. **Target:** `A191RL1Q225SBEA` — real GDP, % change, annualized, quarterly (the same series
   the BEA itself reports and GDPNow-style models target).
2. **Features** (`series_config.py`, hand-picked, not algorithmically searched — same
   "state the judgment call" convention as the donor pools and sector maps elsewhere in this
   repo): Industrial Production, Retail & Food Services Sales, Initial Unemployment Claims,
   Total Nonfarm Payrolls, Housing Starts. Classic bridge-equation indicators — higher
   frequency than GDP, so they're available before GDP is.
3. Every feature series (whatever its native frequency — monthly, weekly) is reduced to one
   number per quarter: the quarter's average level, then its quarter-over-quarter % change —
   same units as the GDP growth target, so the fitted coefficients are growth-on-growth
   elasticities, not a mix of raw index points and a percentage.
4. Plain OLS (numpy `lstsq`, no external ML library) regresses GDP growth on the five
   features' growth rates.

## The backtest is the whole point — a model that doesn't beat "no change" is not useful

Every quarter after the first `MIN_TRAIN_QUARTERS` (`run.py`, default 40) is predicted using
**only strictly prior quarters** in an expanding window — no lookahead. That's compared
against the simplest possible baseline: assume this quarter's growth equals last quarter's.
If the model doesn't beat that baseline, the report says so plainly rather than presenting a
confident-looking nowcast anyway (mirrors the fit-quality gates in the other two tools here).

## What this tool won't do for you

- **Not real-time vintage data.** This uses FRED's final, revised series — the numbers as they
  look *today*, not as they looked on the date each quarter's nowcast would actually have been
  made. Real nowcasting products (e.g. the Atlanta Fed's GDPNow) use vintage/real-time data
  specifically because early releases get revised, sometimes substantially. This tool is a
  methodology exercise in the bridge-equation *relationship*, not a live real-time competitor
  to GDPNow — stated plainly rather than implied.
- **Five hand-picked indicators, not a searched or PCA-reduced set.** Same donor-pool
  philosophy as the causal toolkit: the judgment call is visible in `series_config.py`, not
  hidden inside an automated feature-selection step.
- **Plain OLS, not a dynamic factor model.** No Kalman filter, no mixed-frequency state-space
  model — deliberately the simplest honest version of "do these indicators explain GDP growth."

## Tests

```bash
pip install -r requirements-dev.txt
pytest test_model.py
```

Pure toy-data unit tests (OLS coefficient recovery, and — the important one — that the
walk-forward backtest genuinely never lets a quarter's prediction see that quarter's own data)
— no network calls.
