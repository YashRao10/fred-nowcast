"""Renders the nowcast report: current-quarter estimate, a backtest line
chart (actual vs. model vs. naive baseline), coefficients, and a written
read. No external dependencies — same convention as the other tools here.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

CHART_W, CHART_H = 860, 360
MARGIN = {"top": 24, "right": 90, "bottom": 40, "left": 56}


def _build_chart(backtest_df: pd.DataFrame) -> str:
    plot_w = CHART_W - MARGIN["left"] - MARGIN["right"]
    plot_h = CHART_H - MARGIN["top"] - MARGIN["bottom"]

    quarters = [str(q) for q in backtest_df.index]
    n = len(quarters)
    x_step = plot_w / max(n - 1, 1)

    all_vals = pd.concat([backtest_df["actual"], backtest_df["model_pred"], backtest_df["naive_pred"]])
    y_min, y_max = all_vals.min(), all_vals.max()
    pad = (y_max - y_min) * 0.1 or 1.0
    y_min, y_max = y_min - pad, y_max + pad

    def xscale(i):
        return MARGIN["left"] + i * x_step

    def yscale(v):
        return MARGIN["top"] + (1 - (v - y_min) / (y_max - y_min)) * plot_h

    def path(values):
        pts = [f"{xscale(i):.2f},{yscale(v):.2f}" for i, v in enumerate(values)]
        return "M" + " L".join(pts)

    gridlines = []
    for i in range(5):
        v = y_min + (y_max - y_min) * i / 4
        y = yscale(v)
        gridlines.append(
            f'<line class="gridline" x1="{MARGIN["left"]}" y1="{y:.1f}" '
            f'x2="{CHART_W - MARGIN["right"]}" y2="{y:.1f}"/>'
            f'<text class="tick" x="{MARGIN["left"] - 8}" y="{y:.1f}" '
            f'text-anchor="end" dominant-baseline="middle">{v:.1f}</text>'
        )
    zero_y = yscale(0)
    gridlines.append(f'<line class="zero-line" x1="{MARGIN["left"]}" y1="{zero_y:.1f}" '
                      f'x2="{CHART_W - MARGIN["right"]}" y2="{zero_y:.1f}"/>')

    hover_data = [
        {
            "x": round(xscale(i), 2),
            "quarter": quarters[i],
            "actual": round(float(backtest_df["actual"].iloc[i]), 2),
            "model": round(float(backtest_df["model_pred"].iloc[i]), 2),
            "naive": round(float(backtest_df["naive_pred"].iloc[i]), 2),
        }
        for i in range(n)
    ]

    svg = f"""
<div class="viz-root">
  <svg viewBox="0 0 {CHART_W} {CHART_H}" width="100%" height="{CHART_H}" id="chart-svg">
    {"".join(gridlines)}
    <line class="axis" x1="{MARGIN["left"]}" y1="{MARGIN["top"] + plot_h}"
          x2="{CHART_W - MARGIN["right"]}" y2="{MARGIN["top"] + plot_h}"/>
    <path class="line-naive" d="{path(backtest_df['naive_pred'])}" fill="none"/>
    <path class="line-model" d="{path(backtest_df['model_pred'])}" fill="none"/>
    <path class="line-actual" d="{path(backtest_df['actual'])}" fill="none"/>
    <rect class="hover-capture" x="{MARGIN["left"]}" y="{MARGIN["top"]}"
          width="{plot_w}" height="{plot_h}" fill="transparent"
          onmousemove="onChartHover(event)" onmouseleave="onChartLeave()"/>
    <line class="crosshair" id="crosshair" x1="0" y1="{MARGIN["top"]}" x2="0"
          y2="{MARGIN["top"] + plot_h}" style="display:none"/>
  </svg>
  <div class="legend">
    <span class="legend-item"><span class="swatch swatch-actual"></span>Actual GDP growth</span>
    <span class="legend-item"><span class="swatch swatch-model"></span>Model nowcast</span>
    <span class="legend-item"><span class="swatch swatch-naive"></span>Naive (last quarter)</span>
  </div>
  <div class="tooltip" id="chart-tooltip" style="display:none"></div>
</div>
<script>
  const HOVER_DATA = {json.dumps(hover_data)};
  const X_STEP = {x_step:.4f};
</script>
"""
    return svg


def _verdict(summary: dict, summary_ex_shock: dict, current_fit: dict, nowcast_value: float, nowcast_quarter: str, has_actual: bool) -> str:
    parts = []
    r2 = current_fit["r_squared"]

    if summary["improvement_pct"] <= 0:
        parts.append(
            f"<p><strong>This model does not beat the naive baseline.</strong> Out-of-sample "
            f"RMSE is {summary['model_rmse']:.2f} vs. {summary['naive_rmse']:.2f} for simply "
            f"assuming GDP growth repeats last quarter's value: the five indicators here don't "
            f"add real predictive signal over that trivial baseline across "
            f"{summary['n_quarters']} backtested quarters. Read the nowcast below as a rough "
            f"directional estimate at best, not a validated forecast.</p>"
        )
    else:
        parts.append(
            f"<p>Out-of-sample, this model beats the naive baseline by "
            f"<strong>{summary['improvement_pct']:.0f}%</strong> (RMSE {summary['model_rmse']:.2f} "
            f"vs. {summary['naive_rmse']:.2f} for naive) across {summary['n_quarters']} "
            f"expanding-window backtested quarters: no lookahead, each quarter predicted using "
            f"only prior data.</p>"
        )
    parts.append(
        f"<p><strong>Caveat on that number:</strong> the 2020 COVID quarters (a -28% crash then "
        f"a +34.9% rebound) make the naive baseline look catastrophically bad almost by "
        f"definition: \"assume no change\" is guaranteed to fail across a V-shaped shock, "
        f"regardless of model quality. Excluding those four quarters, the improvement is a more "
        f"honest <strong>{summary_ex_shock['improvement_pct']:.0f}%</strong> (RMSE "
        f"{summary_ex_shock['model_rmse']:.2f} vs. {summary_ex_shock['naive_rmse']:.2f} over "
        f"{summary_ex_shock['n_quarters']} quarters): still a real improvement, just not the "
        f"inflated headline number.</p>"
    )

    band = 1.0 * current_fit["resid_std"]
    parts.append(
        f"<p><strong>{nowcast_quarter} nowcast: {nowcast_value:+.2f}%</strong> "
        f"(&plusmn;{band:.2f} at 1 residual std dev, from a fit with R&sup2;={r2:.2f}). "
        f"{'This quarter has not been officially released yet. This is a live estimate.' if not has_actual else 'The official figure for this quarter has already been released; shown for reference against the model.'}</p>"
    )
    return "\n".join(parts)


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{
    margin:0; font-family: system-ui,-apple-system,"Segoe UI",sans-serif;
    background:#0d0d0d; color:#ffffff;
  }}
  :root[data-theme="light"] body {{ background:#f9f9f7; color:#0b0b0b; }}

  .wrap {{ max-width: 920px; margin: 0 auto; padding: 32px 20px 60px; }}
  h1 {{ font-size: 22px; margin: 0 0 6px; }}
  .subtitle {{ color:#c3c2b7; font-size: 14px; margin-bottom: 24px; }}
  :root[data-theme="light"] .subtitle {{ color:#52514e; }}

  .card {{
    background:#1a1a19; border:1px solid rgba(255,255,255,0.10); border-radius:10px;
    padding:20px 24px; margin-bottom:20px;
  }}
  :root[data-theme="light"] .card {{ background:#fcfcfb; border-color: rgba(11,11,11,0.10); }}

  .card h2 {{ font-size: 15px; text-transform: uppercase; letter-spacing: 0.5px;
    color:#c3c2b7; margin: 0 0 14px; }}
  :root[data-theme="light"] .card h2 {{ color:#52514e; }}

  .hero {{ font-size: 40px; font-weight: 700; margin: 0 0 4px; }}
  .hero-label {{ color:#898781; font-size: 13px; margin-bottom: 18px; }}

  .viz-root {{
    --surface-1:#1a1a19; --text-primary:#ffffff; --text-secondary:#c3c2b7;
    --text-muted:#898781; --grid:#2c2c2a; --baseline:#383835;
    --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  }}
  :root[data-theme="light"] .viz-root {{
    --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
    --text-muted:#898781; --grid:#e1e0d9; --baseline:#c3c2b7;
    --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  }}

  .gridline {{ stroke: var(--grid); stroke-width: 1; }}
  .zero-line {{ stroke: var(--baseline); stroke-width: 1; stroke-dasharray: 2 2; }}
  .axis {{ stroke: var(--baseline); stroke-width: 1; }}
  .tick {{ fill: var(--text-muted); font-size: 11px; }}
  .line-actual {{ stroke: var(--series-1); stroke-width: 2; }}
  .line-model {{ stroke: var(--series-2); stroke-width: 2; }}
  .line-naive {{ stroke: var(--series-3); stroke-width: 1.5; stroke-dasharray: 4 3; opacity: 0.75; }}

  .legend {{ display:flex; gap:18px; padding: 6px 0 0 56px; font-size: 13px;
    color: var(--text-secondary); flex-wrap: wrap; }}
  .legend-item {{ display:flex; align-items:center; gap:6px; }}
  .swatch {{ width:10px; height:10px; border-radius:2px; display:inline-block; }}
  .swatch-actual {{ background: var(--series-1); }}
  .swatch-model {{ background: var(--series-2); }}
  .swatch-naive {{ background: var(--series-3); }}

  .tooltip {{
    position:absolute; background: var(--surface-1); border:1px solid rgba(255,255,255,0.10);
    border-radius:6px; padding:8px 10px; font-size:12px; pointer-events:none;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4); color: var(--text-primary);
  }}
  :root[data-theme="light"] .tooltip {{
    border-color: rgba(11,11,11,0.10); box-shadow: 0 2px 8px rgba(0,0,0,0.12);
  }}
  .tooltip .t-q {{ color: var(--text-secondary); margin-bottom:4px; }}
  .tooltip .t-row {{ display:flex; justify-content:space-between; gap:16px; }}

  table.coef-table {{ width:100%; border-collapse: collapse; font-size: 13px; }}
  table.coef-table th, table.coef-table td {{
    padding: 7px 10px; text-align:left; border-bottom: 1px solid #2c2c2a;
    font-variant-numeric: tabular-nums;
  }}
  :root[data-theme="light"] table.coef-table th,
  :root[data-theme="light"] table.coef-table td {{ border-bottom-color: #e1e0d9; }}
  table.coef-table td.num {{ text-align: right; }}
  table.coef-table th {{ color:#898781; font-size:11px; text-transform:uppercase; letter-spacing:0.5px; }}

  .verdict {{ font-size: 14px; line-height: 1.6; }}
  .footer {{ color:#898781; font-size: 11px; margin-top: 32px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{title}</h1>
  <div class="subtitle">{description}</div>

  <div class="card">
    <div class="hero">{nowcast_value:+.2f}%</div>
    <div class="hero-label">{nowcast_quarter} real GDP growth nowcast (annualized) &plusmn;{band:.2f}pp</div>
    <h2>Backtest: actual vs. model vs. naive baseline</h2>
    {chart}
  </div>

  <div class="card">
    <h2>Model coefficients (this quarter's fit, all history)</h2>
    <table class="coef-table">
      <thead><tr><th>Indicator</th><th class="num">Coefficient</th></tr></thead>
      <tbody>{coef_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Read</h2>
    <div class="verdict">{verdict}</div>
  </div>

  <div class="footer">Generated {generated_at} &middot; fred-nowcast &middot; plain OLS bridge
    equation on final-revised FRED data (not real-time vintage) &middot; backtested with an
    expanding window, no lookahead &middot; a transparent estimate, not a GDPNow-class model.</div>
</div>
<script>
function onChartHover(evt) {{
  const svg = document.getElementById('chart-svg');
  const pt = svg.createSVGPoint();
  pt.x = evt.clientX; pt.y = evt.clientY;
  const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
  let nearest = HOVER_DATA[0], minDist = Infinity;
  for (const d of HOVER_DATA) {{
    const dist = Math.abs(d.x - loc.x);
    if (dist < minDist) {{ minDist = dist; nearest = d; }}
  }}
  const crosshair = document.getElementById('crosshair');
  crosshair.setAttribute('x1', nearest.x); crosshair.setAttribute('x2', nearest.x);
  crosshair.style.display = 'block';
  const tip = document.getElementById('chart-tooltip');
  tip.style.display = 'block';
  tip.style.left = (evt.pageX + 14) + 'px';
  tip.style.top = (evt.pageY - 50) + 'px';
  tip.innerHTML = `<div class="t-q">${{nearest.quarter}}</div>` +
    `<div class="t-row"><span>Actual</span><strong>${{nearest.actual.toFixed(2)}}</strong></div>` +
    `<div class="t-row"><span>Model</span><strong>${{nearest.model.toFixed(2)}}</strong></div>` +
    `<div class="t-row"><span>Naive</span><strong>${{nearest.naive.toFixed(2)}}</strong></div>`;
}}
function onChartLeave() {{
  document.getElementById('crosshair').style.display = 'none';
  document.getElementById('chart-tooltip').style.display = 'none';
}}
</script>
</body>
</html>
"""


def render(
    target_label: str,
    feature_labels: dict,
    feature_cols: list[str],
    current_fit: dict,
    backtest_df: pd.DataFrame,
    summary: dict,
    summary_ex_shock: dict,
    nowcast_value: float,
    nowcast_quarter: str,
    has_actual: bool,
    generated_at: str,
) -> str:
    coef_rows = "".join(
        f'<tr><td>{feature_labels[fid]}</td><td class="num">{coef:+.3f}</td></tr>'
        for fid, coef in zip(feature_cols, current_fit["coefficients"])
    )
    return PAGE_TEMPLATE.format(
        title=f"FRED Nowcast: {target_label}",
        description=(
            f"A bridge-equation OLS nowcast of {target_label}, built from "
            f"{len(feature_cols)} higher-frequency leading indicators."
        ),
        nowcast_value=nowcast_value,
        nowcast_quarter=nowcast_quarter,
        band=1.0 * current_fit["resid_std"],
        chart=_build_chart(backtest_df),
        coef_rows=coef_rows,
        verdict=_verdict(summary, summary_ex_shock, current_fit, nowcast_value, nowcast_quarter, has_actual),
        generated_at=generated_at,
    )
