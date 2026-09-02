"""Which FRED series to nowcast and which leading/coincident indicators to
nowcast it from. Picked by hand (same "state the judgment call" convention as
hand-picked feature sets elsewhere in this work) —
these are the classic bridge-equation indicators (industrial production,
retail sales, initial claims, payrolls), not an algorithmically searched set.
"""
from __future__ import annotations

TARGET = {
    "series_id": "A191RL1Q225SBEA",
    "label": "Real GDP, % change (annualized, quarterly)",
}

FEATURES = {
    "INDPRO": "Industrial Production Index",
    "RSAFS": "Retail & Food Services Sales",
    "ICSA": "Initial Unemployment Claims",
    "PAYEMS": "Total Nonfarm Payrolls",
    "HOUST": "Housing Starts",
}

HISTORY_START = "1990-01-01"
