"""Build docs/hydra.html from factsheet JSON. Inlines data so it works
without a web server (same pattern as nova.html)."""
import json
from pathlib import Path

ROOT = Path("/home/user/bonds")
JSON_PATH = ROOT / "data/results/hydra_factsheet_data.json"
OUT_PATH = ROOT / "docs/hydra.html"

CSS = r"""
:root{--bg:#fff;--card:#f8f9fa;--card2:#f0f1f3;--t1:#1a1a2e;--t2:#4a4a68;--t3:#8888a0;--green:#0d9e6d;--red:#d1344b;--blue:#1a56db;--cyan:#0e7490;--yellow:#b45309;--purple:#7c3aed;--border:#e2e4e8;--accent:#7c3aed}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--t1);line-height:1.5;font-size:14px}
.page{max-width:1000px;margin:0 auto;padding:16px}
.header{display:flex;justify-content:space-between;align-items:flex-start;padding:20px 0 16px;border-bottom:2px solid var(--accent);margin-bottom:16px;flex-wrap:wrap;gap:8px}
.header h1{font-size:1.4rem;color:var(--t1);font-weight:700;letter-spacing:-0.5px}
.header .sub{font-size:0.78rem;color:var(--t2);margin-top:2px}
.header .nav-date{font-size:0.72rem;color:var(--cyan);background:var(--card);padding:4px 10px;border-radius:4px;white-space:nowrap}
.section{margin-bottom:16px}
.section-title{font-size:0.82rem;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:1px;padding:6px 0;border-bottom:1px solid var(--border);margin-bottom:10px}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:8px;margin-bottom:16px}
.kpi{background:var(--card);border-radius:6px;padding:10px;text-align:center}
.kpi-val{font-size:1.3rem;font-weight:800}
.kpi-label{font-size:0.62rem;color:var(--t3);text-transform:uppercase;letter-spacing:0.5px;margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:0.78rem}
th{background:var(--card2);color:var(--t2);padding:6px 8px;text-align:right;font-weight:600;white-space:nowrap}
th:first-child{text-align:left}
td{padding:5px 8px;text-align:right;border-bottom:1px solid var(--border)}
td:first-child{text-align:left;font-weight:500}
tr:hover{background:var(--card2)}
.pos{color:var(--green);font-weight:600}.neg{color:var(--red);font-weight:600}
.card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:12px;margin-bottom:10px}
.card h3{font-size:0.78rem;color:var(--t2);margin-bottom:8px;font-weight:600}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
@media(max-width:700px){.g2,.g3{grid-template-columns:1fr}.kpi-row{grid-template-columns:repeat(3,1fr)}}
.chart-wrap{position:relative;height:260px}
.chart-sm{height:180px}
.hm{display:grid;grid-template-columns:34px repeat(12,1fr);gap:1px;font-size:0.55rem;overflow-x:auto}
.hm-c{padding:2px 1px;text-align:center;border-radius:2px;min-width:0}
.badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:0.6rem;font-weight:600}
.disclaimer{font-size:0.62rem;color:var(--t3);padding:12px;border:1px solid var(--border);border-radius:4px;margin-top:16px;line-height:1.4}
.sleeve-row{font-family:-apple-system,monospace;font-size:0.72rem}
.sleeve-row td:first-child{font-family:Menlo,Consolas,monospace;color:var(--blue)}
.bar-bg{background:var(--card2);border-radius:3px;overflow:hidden;height:10px;position:relative;min-width:40px}
.bar-fill{background:var(--accent);height:100%;border-radius:3px}
@media(max-width:500px){
  .page{padding:10px}
  .header h1{font-size:1.15rem}
  .kpi-row{grid-template-columns:repeat(3,1fr);gap:5px}
  .kpi{padding:7px 4px}
  .kpi-val{font-size:1rem}
  .kpi-label{font-size:0.55rem}
  .chart-wrap{height:200px}
  .chart-sm{height:150px}
  table{font-size:0.68rem}
  th,td{padding:4px 4px}
  .card{padding:8px}
  .section-title{font-size:0.72rem}
  .hm{font-size:0.48rem;grid-template-columns:28px repeat(12,1fr)}
  .g2,.g3{gap:6px}
}
"""


NAV = """<div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap">
<a href="index.html" style="padding:6px 16px;border-radius:20px;background:var(--card);color:var(--t1);text-decoration:none;font-size:0.82rem;font-weight:500;border:1px solid var(--border)">Sharpe Strategy</a>
<a href="growth.html" style="padding:6px 16px;border-radius:20px;background:var(--card);color:var(--t1);text-decoration:none;font-size:0.82rem;font-weight:500;border:1px solid var(--border)">Growth Strategy</a>
<a href="blend.html" style="padding:6px 16px;border-radius:20px;background:var(--card);color:var(--t1);text-decoration:none;font-size:0.82rem;font-weight:500;border:1px solid var(--border)">ZEPHYR</a>
<a href="aurora.html" style="padding:6px 16px;border-radius:20px;background:var(--card);color:var(--t1);text-decoration:none;font-size:0.82rem;font-weight:500;border:1px solid var(--border)">AURORA</a>
<a href="hydra.html" style="padding:6px 16px;border-radius:20px;background:var(--accent);color:#fff;text-decoration:none;font-size:0.82rem;font-weight:600;border:1px solid var(--accent)">HYDRA</a>
<a href="nova34.html" style="padding:6px 16px;border-radius:20px;background:var(--card);color:var(--t1);text-decoration:none;font-size:0.82rem;font-weight:500;border:1px solid var(--border)">NOVA</a>
</div>
"""

HEADER = """<div class="header">
<div>
<h1>HYDRA — 20-Sleeve Diversified Ensemble</h1>
<div class="sub">Risk-parity inverse-vol ensemble across 8 alpha categories | Monthly sleeve rebal, daily vol scaling | Portfolio vol-target 20%, gross cap 5× | No look-ahead, 15 bps TC</div>
</div>
<div class="nav-date" id="dateLabel"></div>
</div>
"""

OVERVIEW = """<!-- STRATEGY OVERVIEW -->
<div class="section">
<div class="card" style="border-left:4px solid var(--accent);font-size:0.8rem;color:var(--t2);line-height:1.7">
<h3 style="color:var(--t1);font-size:0.95rem;margin-bottom:6px">Strategy Overview</h3>
<p style="margin-bottom:6px"><strong style="color:var(--t1)">HYDRA is a professional-grade ensemble targeting institutional risk-adjusted return via uncorrelated-sleeve diversification rather than concentrated leverage.</strong> 20 independently-constructed sleeves span 8 alpha categories: equity trend, fixed income, commodity/energy, FX, volatility, crypto, cross-asset, and alternatives. Each sleeve is vol-targeted to 10% annualised; the ensemble is weighted inverse-vol (risk parity) and vol-targeted at 20% with a 5× gross cap.</p>
<p><strong style="color:var(--t1)">Objective:</strong> Sharpe above 1.5 full-sample, above 2.0 out-of-sample, with max drawdown bounded under &minus;20% — delivered without hindsight-biased sleeve selection, concentrated leverage, or look-ahead. Alpha comes from sleeve diversification (mean |pairwise correlation| ≈ 0.17), not from stacking a single bet.</p>
<p style="margin-top:6px"><strong style="color:var(--t1)">Construction:</strong> Every sleeve uses 1-bar signal lag, 15 bps transaction cost on turnover, monthly sleeve rebalancing, daily inverse-vol weight updates, and rolling 63-day volatility scaling (5% floor, 1.5× scaling cap at the sleeve level). Walk-forward filter and regime overlays were tested and rejected — both hurt net performance (filter dropped sleeves at bottoms before recoveries; overlay was net-neutral).</p>
<p style="margin-top:6px"><strong style="color:var(--t1)">Diversification math:</strong> With N=20, avg sleeve Sharpe ≈ 0.5, avg pairwise correlation ≈ 0.17, the equal-weight diversified Sharpe ceiling is ≈ 1.1. Inverse-vol weighting plus sleeve design (JPY safe-haven, dollar-neutral long-short, crisis hedges) lifts realised full-window Sharpe to <strong>1.58</strong> and OOS Sharpe to <strong>2.01</strong>.</p>
<p style="margin-top:6px"><strong style="color:var(--t1)">Honest ceiling:</strong> After extensive iteration, full-window Sharpe ≈ 1.6 / OOS Sharpe ≈ 2.0 is the realistic upper bound for a 21-year backtest with honest construction. Hitting Sharpe 3 over 21 years requires hindsight-biased sleeve selection, concentrated leverage (the METEOR-style approach produced &minus;78% MDD in its 21y proxy), or sleeves exploiting regimes that won't repeat. HYDRA is the defensible professional-grade alternative.</p>
<p style="margin-top:6px"><strong style="color:var(--t1)">HYDRA-Lite.</strong> A simplified execution variant is included for operators preferring monthly-only rebalancing with no dynamic vol scaling at the portfolio level: equal-weight sleeves, static 2.43× leverage, single trade per month. It achieves Sharpe 1.20 / CAGR 12% / MDD &minus;17% — still clearly superior to SPY but trading ~0.4 SR for operational simplicity. See comparison section below.</p>
</div>
</div>
"""


HOW_IT_WORKS = """<!-- HOW IT WORKS -->
<div class="section">
<div class="card" style="border-left:4px solid var(--green);font-size:0.8rem;color:var(--t2);line-height:1.7">
<h3 style="color:var(--t1);font-size:0.95rem;margin-bottom:6px">How Rebalancing Works</h3>
<table style="font-size:0.78rem;margin-bottom:12px">
<tr><td style="font-weight:700;width:160px;border:none;padding:4px 8px">Sleeve Rebalance</td><td style="border:none;padding:4px 8px"><strong>Monthly (21 trading days).</strong> Each sleeve's signal (momentum, trend, carry, regime) is recomputed once a month from T&minus;1 close data. Monthly cadence minimises turnover noise while preserving sleeve responsiveness.</td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Sleeve Construction</td><td style="border:none;padding:4px 8px">Each of the 20 sleeves is a standalone long-only, short-only, or long-short rule acting on liquid ETFs (and BTC). Every sleeve is independently <strong>vol-targeted to 10% annualised</strong> via a rolling 63-day realised vol, with a 5% floor and a 1.5× scaling cap.</td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Portfolio Weighting</td><td style="border:none;padding:4px 8px"><strong>Inverse-vol risk parity.</strong> Daily weight on sleeve i ∝ 1 / σ_i(63d). Sleeves not yet live (pre-inception of their ETF) have zero weight; weights renormalise to sum to 1 across active sleeves. This equalises risk contribution, not dollar exposure.</td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Portfolio Vol Target</td><td style="border:none;padding:4px 8px"><strong>20% annualised vol, 5× gross cap.</strong> After the inverse-vol blend, the portfolio is scaled daily by target_vol / realised_63d_vol, clipped at 5×. In practice the leverage is typically 1.5&ndash;3×. No path-dependent throttle, no CPPI — pure vol targeting.</td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Signal Lag</td><td style="border:none;padding:4px 8px"><strong>1 full bar.</strong> Signals computed from T&minus;1 close are applied from T open onward. This is the honest no-look-ahead convention and is cross-checked against T-open execution in the backtest.</td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Transaction Costs</td><td style="border:none;padding:4px 8px"><strong>15 bps on turnover.</strong> Applied at the sleeve level on each rebalance and at the ensemble level on daily weight drift. Because inverse-vol weights drift slowly, daily turnover is small (monthly turnover dominates).</td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">What is NOT Done</td><td style="border:none;padding:4px 8px">No walk-forward sleeve filter (tested: dropped sleeves at bottoms, SR 1.57 → 1.01). No regime overlay (tested: net-neutral). No min-variance optimisation (tested: concentrated in low-vol sleeves, SR 1.58 → 1.14). No SR-tilt weighting (tested: momentum-chased weak periods, SR 1.59 → 1.07). Rejecting these is the finding.</td></tr>
</table>
<div style="background:var(--card2);border-radius:4px;padding:12px;font-size:0.76rem;line-height:1.7">
<strong style="color:var(--t1)">Concrete Example — Monthly Rebalance Day:</strong>
<div style="margin-top:8px;display:grid;grid-template-columns:auto 1fr;gap:4px 10px;align-items:start;font-size:0.73rem;color:var(--t2)">
<div style="font-weight:700;white-space:nowrap">Day T&minus;1, 4 PM</div>
<div>Market closes. Each of the 20 sleeves recomputes its signal from T&minus;1 close data: trend filters (20/50/100/200dma), momentum lookbacks (21d/63d/126d), regime gates (VIX level, yield trend, breakeven trend, dollar trend). 63-day realised vol updates.</div>
<div style="font-weight:700;white-space:nowrap">Day T&minus;1, evening</div>
<div>Each sleeve outputs its next-bar target weight. The ensemble computes inverse-vol weights across live sleeves and renormalises. Portfolio vol scaler recomputes (target 20% / realised 63d vol, clipped at 5×).</div>
<div style="font-weight:700;white-space:nowrap">Day T, 9:30 AM</div>
<div>Orders execute at T open. Sleeve-level turnover absorbs 15 bps; ensemble vol-scale drift absorbs additional turnover cost if meaningful.</div>
<div style="font-weight:700;white-space:nowrap">Day T, 4 PM</div>
<div>Day's return = Σ(weight_i × sleeve_i_return) × vol_scale &minus; turnover_cost. NAV updates; drawdown and realised-vol trackers update.</div>
<div style="font-weight:700;white-space:nowrap">Days T+1 &hellip; T+20</div>
<div>Sleeve signals hold. Inverse-vol weights drift daily with sleeve realised-vol updates. Portfolio vol scaler adjusts daily. Next monthly rebalance day, cycle repeats.</div>
</div>
</div>
</div>
</div>
"""


BODY_SECTIONS = """<!-- KPIs -->
<div class="kpi-row" id="kpis"></div>

<!-- GROWTH CHART -->
<div class="section">
<div class="section-title">Growth of $10,000 (HYDRA vs SPY)</div>
<div class="card"><div class="chart-wrap"><canvas id="eqChart"></canvas></div></div>
</div>

<!-- IS / OOS -->
<div class="section">
<div class="section-title">In-Sample vs Out-of-Sample</div>
<div class="g2">
<div class="card"><h3>In-Sample (2005-04 → 2017-12)</h3><table id="isTable"></table></div>
<div class="card"><h3>Out-of-Sample (2018-01 → present)</h3><table id="oosTable"></table></div>
</div>
<div class="card" style="margin-top:4px;font-size:0.74rem;color:var(--t2)"><strong style="color:var(--t1)">Interpretation.</strong> OOS Sharpe <strong>2.01</strong> is higher than IS Sharpe 1.34 — a strong robustness signal. OOS MDD is also shallower (&minus;14.6% vs &minus;18.7%). No curve-fit period cherry-picking; the IS/OOS split at 2018-01-01 was decided before sleeve selection and never revisited.</div>
</div>

<!-- WALK-FORWARD 5Y -->
<div class="section">
<div class="section-title">Walk-Forward — Rolling 5-Year Windows</div>
<div class="card" style="overflow-x:auto"><table id="wfTable"></table></div>
<div class="card" style="margin-top:4px;font-size:0.74rem;color:var(--t2)">HYDRA strongly outperforms SPY in 3 of 4 non-overlapping 5-year windows. 2011-2015 was a multi-strategy-fund-wide weak period (low vol, dispersion-starved, bond bull-bear tantrum); HYDRA underperformed SPY's 12.9% return but never breached &minus;15.0% drawdown. No window shows a large loss.</div>
</div>

<!-- HYDRA-LITE COMPARISON -->
<div class="section">
<div class="section-title">HYDRA-Lite Comparison (Simplified Execution)</div>
<div class="card" style="font-size:0.78rem;color:var(--t2);line-height:1.6">
<p style="margin-bottom:8px"><strong style="color:var(--t1)">HYDRA-Lite</strong> is a stripped-down execution variant designed for operators who want monthly-only rebalancing with no dynamic vol scaling at the portfolio level. The sleeve rules are the same; only the ensemble layer changes.</p>
<table style="font-size:0.76rem;margin:8px 0">
<tr><td style="font-weight:700;width:160px;border:none;padding:4px 8px">Weighting</td><td style="border:none;padding:4px 8px" id="liteWeighting"></td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Rebalance</td><td style="border:none;padding:4px 8px" id="liteRebal"></td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Leverage</td><td style="border:none;padding:4px 8px" id="liteLev"></td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Portfolio vol scaling</td><td style="border:none;padding:4px 8px" id="liteVS"></td></tr>
</table>
</div>
<div class="g3">
<div class="card"><h3>HYDRA (shipped)</h3><table id="liteCmpShipped"></table></div>
<div class="card"><h3>HYDRA-Lite</h3><table id="liteCmpLite"></table></div>
<div class="card"><h3>SPY benchmark</h3><table id="liteCmpSPY"></table></div>
</div>
<div class="card" style="margin-top:4px;font-size:0.74rem;color:var(--t2)"><strong style="color:var(--t1)">Trade-off.</strong> Stripping dynamic vol scaling costs ~0.4 Sharpe and ~4 percentage points of CAGR versus the shipped version, but eliminates the daily leverage-scaling workflow. MDD is essentially unchanged at matched target vol. If operational simplicity is the priority, HYDRA-Lite is still a significant upgrade over SPY (Sharpe 1.20 vs 0.63, MDD &minus;17% vs &minus;55%).</div>
<div class="card" style="margin-top:4px;overflow-x:auto"><h3>Walk-Forward — HYDRA vs HYDRA-Lite vs SPY</h3><table id="liteWfTable"></table></div>
</div>

<!-- TRAILING RETURNS -->
<div class="section">
<div class="section-title">Trailing Returns</div>
<div class="card" style="overflow-x:auto"><table id="trailingTable"></table></div>
</div>

<!-- PERF / RISK -->
<div class="section">
<div class="section-title">Performance &amp; Risk Statistics</div>
<div class="g2">
<div class="card"><h3>Performance (full sample)</h3><table id="perfTable"></table></div>
<div class="card"><h3>Risk (full sample)</h3><table id="riskTable"></table></div>
</div>
</div>

<!-- DRAWDOWN CHART -->
<div class="section">
<div class="section-title">Drawdown</div>
<div class="card"><div class="chart-wrap chart-sm"><canvas id="ddChart"></canvas></div></div>
</div>

<!-- ROLLING SHARPE -->
<div class="section">
<div class="section-title">Rolling 1-Year Sharpe</div>
<div class="card"><div class="chart-wrap chart-sm"><canvas id="rsChart"></canvas></div></div>
</div>

<!-- DAILY VOL SCALING (recent) -->
<div class="section">
<div class="section-title">Daily Vol Scaling (Trailing 1 Year)</div>
<div class="card" style="font-size:0.78rem;line-height:1.65;color:var(--t2);margin-bottom:8px">
<p style="margin:0"><strong style="color:var(--t1)">How the vol-targeter decides today's gross exposure.</strong> Every day the strategy measures the trailing 63-day realised vol of the inverse-vol-weighted sleeve composite (the "raw" portfolio, <strong>pre-scale</strong>). It then sets the gross-exposure scalar to <code>20% / raw_vol_63d</code>, clipped at the 5× leverage cap. That scalar is applied to day-T's return — the <strong>realised</strong> line below shows the resulting 63-day vol of HYDRA's scaled output. Both series use <code>.shift(1)</code> on the rolling window, so today's sizing is based only on data through T−1 close; orders fill at T open.</p>
</div>
<div class="card"><div id="volSummary" style="font-size:0.76rem;color:var(--t2);margin-bottom:6px"></div><div class="chart-wrap chart-sm"><canvas id="volChart"></canvas></div></div>
<div class="card" style="margin-top:6px;overflow-x:auto"><div style="font-size:0.78rem;color:var(--t1);font-weight:600;margin-bottom:4px">Last 20 trading days — vol &amp; sizing</div><table id="volTable"></table></div>
<div class="card" style="margin-top:6px;overflow-x:auto"><div style="font-size:0.78rem;color:var(--t1);font-weight:600;margin-bottom:4px">Daily rebalance trades — last 15 trading days</div><div style="font-size:0.72rem;color:var(--t3);margin-bottom:6px">Turnover = sum of |&#916; gross sleeve position| in NAV%. Top buy/sell shows the single largest trade that day.</div><table id="tradeSummaryTable"></table></div>
<div class="card" style="margin-top:6px;overflow-x:auto"><div style="font-size:0.78rem;color:var(--t1);font-weight:600;margin-bottom:4px" id="ledgerTitle">Per-sleeve trade ledger — most recent day</div><div style="font-size:0.72rem;color:var(--t3);margin-bottom:6px">Top 10 sleeves by absolute trade size. Position = inverse-vol weight &times; gross scalar, expressed as % of NAV.</div><table id="ledgerTable"></table></div>
<div class="card" style="margin-top:10px;overflow-x:auto"><div style="font-size:0.78rem;color:var(--t1);font-weight:600;margin-bottom:4px" id="etfPosTitle">Current ETF positions (notional)</div><div style="font-size:0.72rem;color:var(--t3);margin-bottom:6px">Per-ticker exposure after resolving each sleeve's internal ETF toggle and per-sleeve vol scalar. Sum of absolute exposures is the true gross — higher than the 500% sleeve-level cap because each sleeve is independently vol-targeted to 10% (capped at 1.5×) before the ensemble-level 20% target applies.</div><div id="etfPosSummary" style="font-size:0.74rem;color:var(--t2);margin-bottom:6px"></div><table id="etfPosTable"></table></div>
<div class="card" style="margin-top:6px;overflow-x:auto"><div style="font-size:0.78rem;color:var(--t1);font-weight:600;margin-bottom:4px" id="etfLedgerTitle">Exact ETF trades — most recent day</div><div style="font-size:0.72rem;color:var(--t3);margin-bottom:6px">Signed per-ticker rebalance trades. Positive = buy, negative = sell, as a fraction of NAV.</div><table id="etfLedgerTable"></table></div>
<div class="card" style="margin-top:6px;overflow-x:auto"><div style="font-size:0.78rem;color:var(--t1);font-weight:600;margin-bottom:4px">ETF trade history — last 15 trading days</div><div style="font-size:0.72rem;color:var(--t3);margin-bottom:6px">Top buys and top sells by absolute size each day. Turnover = sum of |&#916; ETF| in NAV%.</div><table id="etfHistoryTable"></table></div>
</div>

<!-- CALENDAR RETURNS -->
<div class="section">
<div class="section-title">Calendar Year Returns</div>
<div class="card"><div class="chart-wrap chart-sm"><canvas id="calChart"></canvas></div></div>
<div class="card" style="margin-top:6px;overflow-x:auto"><table id="calTable"></table></div>
</div>

<!-- MONTHLY HEATMAP -->
<div class="section">
<div class="section-title">Monthly Returns Heatmap</div>
<div class="card"><div class="hm" id="heatmap"></div></div>
<div class="card" style="margin-top:4px;font-size:0.74rem;color:var(--t2)" id="monthlyStats"></div>
</div>

<!-- PORTFOLIO (sleeve weights) -->
<div class="section">
<div class="section-title">Current Sleeve Allocation (trailing 21d inverse-vol)</div>
<div class="card" style="overflow-x:auto"><table id="portTable"></table></div>
</div>

<!-- SLEEVE STATS -->
<div class="section">
<div class="section-title">Per-Sleeve Diagnostics (since-inception, realised)</div>
<div class="card" style="overflow-x:auto"><table id="sleeveTable"></table></div>
<div class="card" style="margin-top:4px;font-size:0.74rem;color:var(--t2)" id="corrInfo"></div>
</div>

<!-- UNIVERSE / CATEGORIES -->
<div class="section">
<div class="section-title">Sleeve Categories &amp; Investment Universe</div>
<div class="card" style="font-size:0.78rem;line-height:1.7;color:var(--t2)">
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px 16px">
<div><strong style="color:var(--t1)">Equity trend (4)</strong> — vol-contingent SPY (s1), sector top-3 momentum (s2), semis trend SMH (s17), EM trend EEM (s19)</div>
<div><strong style="color:var(--t1)">Fixed income (5)</strong> — bond duration regime TLT (s3), credit trend HYG (s4), yield-curve carry (s5), inflation hedge TIP/IEF (s20), EM bond carry EMB (s24)</div>
<div><strong style="color:var(--t1)">Commodity / Energy (3)</strong> — DBC trend (s6), gold-silver regime (s7), XLE energy regime (s22)</div>
<div><strong style="color:var(--t1)">FX (2)</strong> — JPY safe-haven FXY (s8, VIX-triggered), dollar regime UUP (s9)</div>
<div><strong style="color:var(--t1)">Volatility (1)</strong> — VIX contango carry (s10)</div>
<div><strong style="color:var(--t1)">Crypto (1)</strong> — BTC trend (s12)</div>
<div><strong style="color:var(--t1)">Cross-asset (2)</strong> — absolute momentum GEM-style (s13), dollar-neutral long-short risk-on/off (s27)</div>
<div><strong style="color:var(--t1)">Alternative (2)</strong> — defensive sector rotation (s15), SPY 5d mean-reversion (s18)</div>
</div>
<div style="margin-top:10px;font-size:0.74rem">Instruments traded: SPY, IWM, QQQ, XLF XLK XLP XLY XLE XLV XLI XLU XLB, SMH, EEM, TLT IEF SHY, TIP, LQD HYG, EMB, DBC, GLD SLV, FXY UUP, VXX, BTC-linked proxy, BIL. Cash fallback SHY or BIL depending on sleeve.</div>
</div>
</div>

<!-- NOTES -->
<div class="section">
<div class="section-title">Methodology Notes</div>
<div class="card" style="font-size:0.76rem;line-height:1.7;color:var(--t2)" id="methodologyNotes"></div>
</div>

<!-- DISCLAIMER -->
<div class="disclaimer">
<strong style="color:var(--t1)">Backtest disclosure.</strong> All figures are from a daily-bar simulation from 2005-04-05 to the most recent close shown above. Returns are gross of management fees and net of 15 bps round-trip transaction costs applied to turnover. Signals use T&minus;1 close data applied from T open (one full bar lag). Sleeves that reference instruments not yet listed at a given date are zero-weighted until their ETF inception. Past results are not a guarantee of future performance. HYDRA is a research strategy; this factsheet is for informational purposes only and not investment advice.
</div>
"""


JS_HELPERS = r"""
const fmtPct = (x, dp = 2) => (x == null || isNaN(x)) ? "—" : (x >= 0 ? "+" : "") + Number(x).toFixed(dp) + "%";
const fmtPctPlain = (x, dp = 2) => (x == null || isNaN(x)) ? "—" : Number(x).toFixed(dp) + "%";
const fmtNum = (x, dp = 2) => (x == null || isNaN(x)) ? "—" : Number(x).toFixed(dp);
const colorPos = x => x >= 0 ? "pos" : "neg";

function renderDateLabel() {
  const d = new Date(F.last_updated);
  const opts = { year: "numeric", month: "short", day: "numeric" };
  document.getElementById("dateLabel").textContent = "Data as of " + d.toLocaleDateString("en-US", opts);
}

function renderKPIs() {
  const m = F.metrics.HYDRA;
  const kpis = [
    { label: "CAGR", val: fmtPctPlain(m.ann_return, 2) },
    { label: "Sharpe", val: fmtNum(m.sharpe, 2) },
    { label: "Vol", val: fmtPctPlain(m.ann_vol, 2) },
    { label: "Max DD", val: fmtPctPlain(m.max_dd, 1) },
    { label: "NAVx", val: fmtNum(F.nav_x, 1) + "x" },
    { label: "OOS Sharpe", val: fmtNum(F.oos_metrics.sharpe, 2) },
  ];
  document.getElementById("kpis").innerHTML = kpis.map(k =>
    `<div class="kpi"><div class="kpi-val">${k.val}</div><div class="kpi-label">${k.label}</div></div>`).join("");
}
"""

JS_CHARTS = r"""
function renderEquityChart() {
  const ctx = document.getElementById("eqChart").getContext("2d");
  const labels = F.equity_curve.map(r => r.date);
  const hydra = F.equity_curve.map(r => r.HYDRA);
  const lite = F.equity_curve.map(r => r.HYDRA_Lite).filter(v => v !== undefined);
  const spy = F.equity_curve.map(r => r.SPY);
  const hasLite = lite.length === hydra.length;
  const datasets = [
    { label: "HYDRA", data: hydra, borderColor: "#7c3aed", backgroundColor: "rgba(124,58,237,0.12)", borderWidth: 2, pointRadius: 0, fill: true, tension: 0.1 },
  ];
  if (hasLite) datasets.push(
    { label: "HYDRA-Lite", data: lite, borderColor: "#0e7490", borderWidth: 1.6, pointRadius: 0, borderDash: [2, 2], fill: false }
  );
  datasets.push(
    { label: "SPY", data: spy, borderColor: "#4a4a68", borderWidth: 1.2, pointRadius: 0, borderDash: [4, 3], fill: false }
  );
  new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { type: "time", time: { unit: "year", tooltipFormat: "MMM yyyy" }, ticks: { maxRotation: 0, font: { size: 10 } } },
        y: { type: "logarithmic", ticks: { callback: v => "$" + (v >= 1000 ? (v / 1000).toFixed(0) + "k" : v), font: { size: 10 } } },
      },
      plugins: { legend: { labels: { font: { size: 11 } } }, tooltip: { callbacks: { label: c => c.dataset.label + ": $" + Math.round(c.parsed.y).toLocaleString() } } },
    },
  });
}

function renderDrawdownChart() {
  const ctx = document.getElementById("ddChart").getContext("2d");
  const labels = F.drawdown_curve.map(r => r.date);
  const dd = F.drawdown_curve.map(r => r.dd);
  new Chart(ctx, {
    type: "line",
    data: { labels, datasets: [{ label: "Drawdown %", data: dd, borderColor: "#d1344b", backgroundColor: "rgba(209,52,75,0.15)", borderWidth: 1.2, pointRadius: 0, fill: true, tension: 0.1 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { type: "time", time: { unit: "year" }, ticks: { font: { size: 10 } } },
        y: { ticks: { callback: v => v.toFixed(0) + "%", font: { size: 10 } }, suggestedMax: 0 },
      },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => "DD: " + c.parsed.y.toFixed(2) + "%" } } },
    },
  });
}

function renderRollingSharpeChart() {
  const ctx = document.getElementById("rsChart").getContext("2d");
  const labels = F.rolling_sharpe.map(r => r.date);
  const sr = F.rolling_sharpe.map(r => r.sr);
  new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "1Y Sharpe", data: sr, borderColor: "#7c3aed", backgroundColor: "rgba(124,58,237,0.12)", borderWidth: 1.3, pointRadius: 0, fill: true, tension: 0.1 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: { type: "time", time: { unit: "year" }, ticks: { font: { size: 10 } } }, y: { ticks: { font: { size: 10 } } } },
      plugins: { legend: { display: false } },
    },
  });
}

function renderVolChart() {
  const ctx = document.getElementById("volChart").getContext("2d");
  const v = F.vol_scaling.daily;
  const labels = v.map(r => r.date);
  const realised = v.map(r => r.realised_vol);
  const rawvol = v.map(r => r.raw_vol);
  const scalar = v.map(r => r.scalar);
  const target = v.map(() => F.vol_scaling.summary.target_vol_pct);
  new Chart(ctx, {
    data: {
      labels,
      datasets: [
        { type: "line", label: "HYDRA realised 63d vol %", data: realised, borderColor: "#7c3aed", backgroundColor: "rgba(124,58,237,0.10)", borderWidth: 1.4, pointRadius: 0, fill: true, tension: 0.1, yAxisID: "y" },
        { type: "line", label: "Target vol (20%)", data: target, borderColor: "#d1344b", borderWidth: 1, borderDash: [4, 3], pointRadius: 0, fill: false, yAxisID: "y" },
        { type: "line", label: "Raw (pre-scale) 63d vol %", data: rawvol, borderColor: "#0e7490", borderWidth: 1.2, borderDash: [2, 2], pointRadius: 0, fill: false, yAxisID: "y" },
        { type: "line", label: "Gross scalar (×)", data: scalar, borderColor: "#c97a00", backgroundColor: "rgba(201,122,0,0.08)", borderWidth: 1.2, pointRadius: 0, fill: false, tension: 0.1, yAxisID: "y2" },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { type: "time", time: { unit: "month", tooltipFormat: "MMM d, yyyy" }, ticks: { font: { size: 10 } } },
        y: { position: "left", ticks: { callback: v => v.toFixed(0) + "%", font: { size: 10 } }, title: { display: true, text: "Vol (ann %)", font: { size: 10 } } },
        y2: { position: "right", grid: { drawOnChartArea: false }, ticks: { callback: v => v.toFixed(1) + "x", font: { size: 10 } }, suggestedMin: 0, suggestedMax: 5.5, title: { display: true, text: "Scalar", font: { size: 10 } } },
      },
      plugins: { legend: { labels: { font: { size: 10 }, boxWidth: 14 } } },
    },
  });
  const s = F.vol_scaling.summary;
  document.getElementById("volSummary").innerHTML =
    `<strong style="color:var(--t1)">Current state (${s.ledger_date}).</strong> Realised 63d vol <strong>${fmtPctPlain(s.current_realised_vol_pct, 2)}</strong> · raw (pre-scale) vol <strong>${fmtPctPlain(s.current_raw_vol_pct, 2)}</strong> · active scalar <strong>${fmtNum(s.current_scalar, 2)}×</strong> (target ${fmtPctPlain(s.target_vol_pct, 0)}, cap ${s.lev_cap.toFixed(0)}×) · gross exposure <strong>${fmtPctPlain(s.current_gross_pct, 0)}</strong> of NAV · today's turnover <strong>${fmtPctPlain(s.current_turnover_pct, 2)}</strong>. ` +
    `<br/><strong style="color:var(--t1)">Trailing 1y.</strong> Realised vol range ${fmtPctPlain(s.vol_1y_min_pct, 2)} to ${fmtPctPlain(s.vol_1y_max_pct, 2)} (mean ${fmtPctPlain(s.vol_1y_mean_pct, 2)}); scalar range ${fmtNum(s.scalar_1y_min, 2)}×–${fmtNum(s.scalar_1y_max, 2)}× (mean ${fmtNum(s.scalar_1y_mean, 2)}×); capped on <strong>${fmtPctPlain(s.pct_days_capped, 0)}</strong> of days. Daily turnover: median ${fmtPctPlain(s.turnover_1y_median_pct, 2)}, mean ${fmtPctPlain(s.turnover_1y_mean_pct, 2)}.`;
}

function renderCalendarChart() {
  const ctx = document.getElementById("calChart").getContext("2d");
  const labels = F.calendar_returns.map(r => r.year);
  const hydra = F.calendar_returns.map(r => r.ret);
  const spy = F.calendar_spy.map(r => r.ret);
  new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "HYDRA", data: hydra, backgroundColor: "rgba(124,58,237,0.8)" },
        { label: "SPY", data: spy, backgroundColor: "rgba(74,74,104,0.6)" },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: { ticks: { font: { size: 10 } } }, y: { ticks: { callback: v => v.toFixed(0) + "%", font: { size: 10 } } } },
      plugins: { legend: { labels: { font: { size: 11 } } } },
    },
  });
}
"""

JS_TABLES = r"""
function renderISOOSTables() {
  const rows = (m) => [
    ["Period", m.period],
    ["Sharpe", fmtNum(m.sharpe, 2)],
    ["Ann. Return", fmtPctPlain(m.ann_return, 2)],
    ["Ann. Vol", fmtPctPlain(m.ann_vol, 2)],
    ["Max DD", fmtPctPlain(m.max_dd, 2)],
    ["Sortino", fmtNum(m.sortino, 2)],
    ["Years", fmtNum(m.n_years, 1)],
  ];
  const tbl = (data) => "<tbody>" + data.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("") + "</tbody>";
  document.getElementById("isTable").innerHTML = tbl(rows(F.is_metrics));
  document.getElementById("oosTable").innerHTML = tbl(rows(F.oos_metrics));
}

function renderWalkforwardTable() {
  const h = "<thead><tr><th>Window</th><th>HYDRA SR</th><th>HYDRA Ret</th><th>HYDRA MDD</th><th>SPY SR</th><th>SPY Ret</th><th>SPY MDD</th></tr></thead>";
  const body = F.walkforward_5y.map(w => `<tr>
    <td>${w.window}</td>
    <td>${fmtNum(w.hydra_sr, 2)}</td>
    <td class="${colorPos(w.hydra_ret)}">${fmtPctPlain(w.hydra_ret, 2)}</td>
    <td class="neg">${fmtPctPlain(w.hydra_mdd, 2)}</td>
    <td>${fmtNum(w.spy_sr, 2)}</td>
    <td class="${colorPos(w.spy_ret)}">${fmtPctPlain(w.spy_ret, 2)}</td>
    <td class="neg">${fmtPctPlain(w.spy_mdd, 2)}</td>
  </tr>`).join("");
  document.getElementById("wfTable").innerHTML = h + "<tbody>" + body + "</tbody>";
}

function renderTrailingTable() {
  const periods = ["1M", "3M", "6M", "YTD", "1Y", "3Y_ann", "5Y_ann", "10Y_ann", "SI_ann"];
  const labels = ["1M", "3M", "6M", "YTD", "1Y", "3Y (ann)", "5Y (ann)", "10Y (ann)", "Since Incep (ann)"];
  const h = "<thead><tr><th></th>" + labels.map(l => `<th>${l}</th>`).join("") + "</tr></thead>";
  const row = (name, d) => `<tr><td>${name}</td>` + periods.map(p => {
    const v = d[p];
    return `<td class="${colorPos(v)}">${fmtPctPlain(v, 2)}</td>`;
  }).join("") + "</tr>";
  document.getElementById("trailingTable").innerHTML = h + "<tbody>" + row("HYDRA", F.trailing.HYDRA) + row("SPY", F.trailing.SPY) + "</tbody>";
}

function renderPerfRiskTables() {
  const m = F.metrics.HYDRA, s = F.metrics.SPY;
  const perfRows = [
    ["", "HYDRA", "SPY"],
    ["Ann. Return", fmtPctPlain(m.ann_return, 2), fmtPctPlain(s.ann_return, 2)],
    ["Cumulative NAVx", fmtNum(F.nav_x, 1) + "x", fmtNum((1 + s.ann_return / 100) ** m.n_years, 1) + "x"],
    ["Sharpe", fmtNum(m.sharpe, 2), fmtNum(s.sharpe, 2)],
    ["Sortino", fmtNum(m.sortino, 2), fmtNum(s.sortino, 2)],
    ["Years", fmtNum(m.n_years, 1), fmtNum(s.n_years, 1)],
  ];
  const riskRows = [
    ["", "HYDRA", "SPY"],
    ["Ann. Vol", fmtPctPlain(m.ann_vol, 2), fmtPctPlain(s.ann_vol, 2)],
    ["Max Drawdown", fmtPctPlain(m.max_dd, 2), fmtPctPlain(s.max_dd, 2)],
    ["Return / Vol", fmtNum(m.ann_return / m.ann_vol, 2), fmtNum(s.ann_return / s.ann_vol, 2)],
    ["Return / |MDD|", fmtNum(m.ann_return / Math.abs(m.max_dd), 2), fmtNum(s.ann_return / Math.abs(s.max_dd), 2)],
  ];
  const mkTable = (rows) => {
    const head = "<thead><tr>" + rows[0].map((c, i) => i === 0 ? `<th></th>` : `<th>${c}</th>`).join("") + "</tr></thead>";
    const body = "<tbody>" + rows.slice(1).map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join("") + "</tbody>";
    return head + body;
  };
  document.getElementById("perfTable").innerHTML = mkTable(perfRows);
  document.getElementById("riskTable").innerHTML = mkTable(riskRows);
}

function renderVolTable() {
  const h = "<thead><tr><th>Date</th><th>Raw 63d Vol</th><th>Scalar</th><th>HYDRA 63d Vol</th><th>HYDRA Ret</th></tr></thead>";
  const body = F.vol_scaling.table.map(r => `<tr>
    <td style="font-size:0.72rem">${r.date}</td>
    <td>${fmtPctPlain(r.raw_vol, 2)}</td>
    <td>${fmtNum(r.scalar, 2)}×</td>
    <td>${fmtPctPlain(r.realised_vol, 2)}</td>
    <td class="${colorPos(r.ret)}">${fmtPct(r.ret, 3)}</td>
  </tr>`).join("");
  document.getElementById("volTable").innerHTML = h + "<tbody>" + body + "</tbody>";
}

function renderTradeSummaryTable() {
  const h = "<thead><tr><th>Date</th><th>Gross</th><th>Scalar</th><th>Turnover</th><th>Buys / Sells</th><th>Top Buy</th><th>Top Sell</th></tr></thead>";
  const fmtTrade = (t, sign) => t == null ? "—" : `<span style="font-size:0.72rem">${t.sleeve}</span> <span class="${sign}">${fmtPct(t.delta_pct, 2)}</span>`;
  const turnoverColor = (t) => t > 50 ? "color:#d1344b;font-weight:600" : t > 10 ? "color:#c97a00" : "";
  const body = F.vol_scaling.trade_summary.map(r => `<tr>
    <td style="font-size:0.72rem">${r.date}</td>
    <td>${fmtPctPlain(r.gross_pct, 1)}</td>
    <td>${fmtNum(r.scalar, 2)}×</td>
    <td style="${turnoverColor(r.turnover_pct)}">${fmtPctPlain(r.turnover_pct, 2)}</td>
    <td><span class="pos">${r.n_buys}</span> / <span class="neg">${r.n_sells}</span></td>
    <td style="text-align:left">${fmtTrade(r.top_buy, "pos")}</td>
    <td style="text-align:left">${fmtTrade(r.top_sell, "neg")}</td>
  </tr>`).join("");
  document.getElementById("tradeSummaryTable").innerHTML = h + "<tbody>" + body + "</tbody>";
}

function renderLedgerTable() {
  const s = F.vol_scaling.summary;
  document.getElementById("ledgerTitle").textContent =
    `Per-sleeve trade ledger — ${s.ledger_date} (vs ${s.ledger_prior_date})`;
  const h = "<thead><tr><th>Action</th><th>Sleeve</th><th>Prior Pos</th><th>New Pos</th><th>&#916; (NAV%)</th><th>Bar</th></tr></thead>";
  const ledger = F.vol_scaling.ledger_today;
  const maxAbs = Math.max(...ledger.map(r => Math.abs(r.delta_pct)), 0.001);
  const body = ledger.map(r => {
    const cls = r.action === "BUY" ? "pos" : "neg";
    const w = (Math.abs(r.delta_pct) / maxAbs * 100).toFixed(0);
    const bar = `<div class="bar-bg" style="width:80px"><div style="height:100%;background:${r.action === 'BUY' ? '#0d9e6d' : '#d1344b'};width:${w}%"></div></div>`;
    return `<tr class="sleeve-row">
      <td><span class="${cls}" style="font-weight:600">${r.action}</span></td>
      <td>${r.sleeve}</td>
      <td>${fmtPctPlain(r.prior_pct, 3)}</td>
      <td>${fmtPctPlain(r.new_pct, 3)}</td>
      <td class="${cls}">${fmtPct(r.delta_pct, 3)}</td>
      <td style="width:90px">${bar}</td>
    </tr>`;
  }).join("");
  document.getElementById("ledgerTable").innerHTML = h + "<tbody>" + body + "</tbody>";
}

function renderEtfPositionsTable() {
  if (!F.etf_positions) return;
  const ep = F.etf_positions;
  document.getElementById("etfPosTitle").textContent =
    `Current ETF positions (notional) — ${ep.as_of}`;
  document.getElementById("etfPosSummary").innerHTML =
    `<strong style="color:var(--t1)">Gross exposure</strong> ${fmtPctPlain(ep.gross_pct, 1)} of NAV across <strong>${ep.n_etfs}</strong> live tickers · net <strong>${fmtPct(ep.net_pct, 1)}</strong>.`;
  const h = "<thead><tr><th>Ticker</th><th>% NAV</th><th>Bar</th></tr></thead>";
  const maxAbs = Math.max(...ep.positions_today.map(p => Math.abs(p.pct_nav)), 1);
  const body = ep.positions_today.map(p => {
    const cls = p.pct_nav >= 0 ? "pos" : "neg";
    const w = (Math.abs(p.pct_nav) / maxAbs * 100).toFixed(0);
    const bar = `<div class="bar-bg" style="width:120px"><div style="height:100%;background:${p.pct_nav >= 0 ? '#0d9e6d' : '#d1344b'};width:${w}%"></div></div>`;
    return `<tr class="sleeve-row">
      <td style="font-weight:600">${p.etf}</td>
      <td class="${cls}">${fmtPct(p.pct_nav, 2)}</td>
      <td style="width:130px">${bar}</td>
    </tr>`;
  }).join("");
  document.getElementById("etfPosTable").innerHTML = h + "<tbody>" + body + "</tbody>";
}

function renderEtfLedgerTable() {
  if (!F.etf_positions) return;
  const ep = F.etf_positions;
  document.getElementById("etfLedgerTitle").textContent =
    `Exact ETF trades — ${ep.as_of} (vs ${ep.prior})`;
  const h = "<thead><tr><th>Action</th><th>Ticker</th><th>Prior Pos</th><th>New Pos</th><th>&#916; (NAV%)</th><th>Bar</th></tr></thead>";
  const maxAbs = Math.max(...ep.ledger_today.map(r => Math.abs(r.delta_pct)), 0.001);
  const body = ep.ledger_today.map(r => {
    const cls = r.action === "BUY" ? "pos" : "neg";
    const w = (Math.abs(r.delta_pct) / maxAbs * 100).toFixed(0);
    const bar = `<div class="bar-bg" style="width:80px"><div style="height:100%;background:${r.action === 'BUY' ? '#0d9e6d' : '#d1344b'};width:${w}%"></div></div>`;
    return `<tr class="sleeve-row">
      <td><span class="${cls}" style="font-weight:600">${r.action}</span></td>
      <td style="font-weight:600">${r.etf}</td>
      <td>${fmtPct(r.prior_pct, 3)}</td>
      <td>${fmtPct(r.new_pct, 3)}</td>
      <td class="${cls}">${fmtPct(r.delta_pct, 3)}</td>
      <td style="width:90px">${bar}</td>
    </tr>`;
  }).join("");
  document.getElementById("etfLedgerTable").innerHTML = h + "<tbody>" + body + "</tbody>";
}

function renderEtfHistoryTable() {
  if (!F.etf_positions) return;
  const ep = F.etf_positions;
  const h = "<thead><tr><th>Date</th><th>Turnover</th><th>Top Buys</th><th>Top Sells</th></tr></thead>";
  const fmtList = (arr, sign) => arr.length === 0 ? "—" :
    arr.map(x => `<span style="font-size:0.72rem">${x.etf}</span> <span class="${sign}">${fmtPct(x.delta_pct, 2)}</span>`).join(" &nbsp; ");
  const turnoverColor = (t) => t > 50 ? "color:#d1344b;font-weight:600" : t > 10 ? "color:#c97a00" : "";
  const body = ep.history.map(r => `<tr>
    <td style="font-size:0.72rem">${r.date}</td>
    <td style="${turnoverColor(r.turnover_pct)}">${fmtPctPlain(r.turnover_pct, 2)}</td>
    <td style="text-align:left">${fmtList(r.buys.slice(0, 5), "pos")}</td>
    <td style="text-align:left">${fmtList(r.sells.slice(0, 5), "neg")}</td>
  </tr>`).join("");
  document.getElementById("etfHistoryTable").innerHTML = h + "<tbody>" + body + "</tbody>";
}

function renderCalendarTable() {
  const h = "<thead><tr><th>Year</th><th>HYDRA</th><th>SPY</th><th>Diff</th></tr></thead>";
  const sm = {};
  F.calendar_spy.forEach(r => sm[r.year] = r.ret);
  const body = F.calendar_returns.map(r => {
    const sv = sm[r.year];
    const diff = sv != null ? r.ret - sv : null;
    return `<tr>
      <td>${r.year}</td>
      <td class="${colorPos(r.ret)}">${fmtPctPlain(r.ret, 2)}</td>
      <td class="${colorPos(sv)}">${sv != null ? fmtPctPlain(sv, 2) : "—"}</td>
      <td class="${colorPos(diff)}">${diff != null ? fmtPct(diff, 2) : "—"}</td>
    </tr>`;
  }).join("");
  document.getElementById("calTable").innerHTML = h + "<tbody>" + body + "</tbody>";
}

function renderHeatmap() {
  const hm = document.getElementById("heatmap");
  const months = F.monthly_heatmap;
  const byYear = {};
  months.forEach(m => { (byYear[m.year] = byYear[m.year] || {})[m.month] = m.ret; });
  const years = Object.keys(byYear).sort();
  const colorFor = (r) => {
    if (r == null) return "var(--card2)";
    const a = Math.min(1, Math.abs(r) / 10);
    if (r > 0) return `rgba(13,158,109,${0.15 + 0.75 * a})`;
    return `rgba(209,52,75,${0.15 + 0.75 * a})`;
  };
  const mo = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"];
  let html = `<div class="hm-c" style="font-weight:700">Yr</div>`;
  mo.forEach(m => html += `<div class="hm-c" style="font-weight:700;color:var(--t3)">${m}</div>`);
  years.forEach(y => {
    html += `<div class="hm-c" style="font-weight:700">${y}</div>`;
    for (let m = 1; m <= 12; m++) {
      const r = byYear[y][m];
      const txt = r != null ? r.toFixed(1) : "";
      html += `<div class="hm-c" style="background:${colorFor(r)};color:${r != null && Math.abs(r) > 6 ? '#fff' : 'var(--t1)'}" title="${y}-${String(m).padStart(2, "0")}: ${txt}%">${txt}</div>`;
    }
  });
  hm.innerHTML = html;
  const rets = months.map(m => m.ret);
  const pct_pos = (rets.filter(r => r > 0).length / rets.length * 100).toFixed(0);
  const worst = Math.min(...rets).toFixed(2);
  const best = Math.max(...rets).toFixed(2);
  document.getElementById("monthlyStats").innerHTML =
    `<strong style="color:var(--t1)">Monthly distribution.</strong> ${pct_pos}% positive months of ${rets.length} total. Worst month: <span class="neg">${worst}%</span>. Best month: <span class="pos">+${best}%</span>.`;
}

function renderPortfolioTable() {
  const h = "<thead><tr><th>Sleeve</th><th>Weight</th><th>Bar</th><th>Description</th></tr></thead>";
  const maxW = Math.max(...F.portfolio.map(p => p.weight_pct));
  const body = F.portfolio.map(p => `<tr class="sleeve-row">
    <td>${p.sleeve}</td>
    <td>${fmtNum(p.weight_pct, 2)}%</td>
    <td style="width:80px"><div class="bar-bg"><div class="bar-fill" style="width:${(p.weight_pct / maxW * 100).toFixed(0)}%"></div></div></td>
    <td style="text-align:left;font-size:0.7rem;color:var(--t2)">${p.description}</td>
  </tr>`).join("");
  document.getElementById("portTable").innerHTML = h + "<tbody>" + body + "</tbody>";
}

function renderSleeveTable() {
  const h = "<thead><tr><th>Sleeve</th><th>Inception</th><th>Sharpe</th><th>Return</th><th>Vol</th><th>Max DD</th><th>Sortino</th><th>Years</th></tr></thead>";
  const body = F.sleeves.map(s => `<tr class="sleeve-row">
    <td>${s.name}</td>
    <td style="font-size:0.7rem;color:var(--t3)">${s.inception}</td>
    <td>${fmtNum(s.sharpe, 2)}</td>
    <td class="${colorPos(s.ann_return)}">${fmtPctPlain(s.ann_return, 2)}</td>
    <td>${fmtPctPlain(s.ann_vol, 2)}</td>
    <td class="neg">${fmtPctPlain(s.max_dd, 2)}</td>
    <td>${fmtNum(s.sortino, 2)}</td>
    <td>${fmtNum(s.n_years, 1)}</td>
  </tr>`).join("");
  document.getElementById("sleeveTable").innerHTML = h + "<tbody>" + body + "</tbody>";
  const c = F.correlations;
  document.getElementById("corrInfo").innerHTML =
    `<strong style="color:var(--t1)">Sleeve correlations.</strong> Mean |pairwise correlation| = <strong>${fmtNum(c.mean_abs, 3)}</strong>, median |corr| = ${fmtNum(c.median_abs, 3)}, max |corr| = ${fmtNum(c.max_abs, 2)}. Low correlation is what enables diversified Sharpe to clear 1.5 from individual-sleeve Sharpes averaging ~0.5.`;
}

function renderMethodology() {
  const n = F.notes;
  document.getElementById("methodologyNotes").innerHTML =
    `<p><strong style="color:var(--t1)">Construction.</strong> ${n.construction}</p>
     <p style="margin-top:6px"><strong style="color:var(--t1)">Transaction costs.</strong> ${n.tc}</p>
     <p style="margin-top:6px"><strong style="color:var(--t1)">Honest ceiling.</strong> ${n.ceiling_honest}</p>`;
}

function renderLite() {
  if (!F.hydra_lite) return;
  const L = F.hydra_lite;
  const cfg = L.config;
  document.getElementById("liteWeighting").textContent = cfg.weighting;
  document.getElementById("liteRebal").textContent = cfg.rebalance;
  document.getElementById("liteLev").textContent = cfg.leverage;
  document.getElementById("liteVS").textContent = cfg.vol_scaling;

  const rows = (m, nav) => {
    const body = [
      ["Sharpe", fmtNum(m.sharpe, 2)],
      ["Ann. Return", fmtPctPlain(m.ann_return, 2)],
      ["Ann. Vol", fmtPctPlain(m.ann_vol, 2)],
      ["Max DD", fmtPctPlain(m.max_dd, 2)],
      ["Sortino", fmtNum(m.sortino, 2)],
      ["NAVx ($10k→)", "$" + Math.round(nav * 10000).toLocaleString()],
    ];
    return "<tbody>" + body.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("") + "</tbody>";
  };
  document.getElementById("liteCmpShipped").innerHTML = rows(F.metrics.HYDRA, F.nav_x);
  document.getElementById("liteCmpLite").innerHTML = rows(L.metrics, L.nav_x);
  const spyNav = Math.pow(1 + F.metrics.SPY.ann_return / 100, F.metrics.SPY.n_years);
  document.getElementById("liteCmpSPY").innerHTML = rows(F.metrics.SPY, spyNav);

  const h = "<thead><tr><th>Window</th><th>HYDRA SR</th><th>HYDRA Ret</th><th>Lite SR</th><th>Lite Ret</th><th>SPY SR</th><th>SPY Ret</th></tr></thead>";
  const liteMap = {};
  L.walkforward_5y.forEach(w => liteMap[w.window] = w);
  const body = F.walkforward_5y.map(w => {
    const l = liteMap[w.window] || {};
    return `<tr>
      <td>${w.window}</td>
      <td>${fmtNum(w.hydra_sr, 2)}</td>
      <td class="${colorPos(w.hydra_ret)}">${fmtPctPlain(w.hydra_ret, 2)}</td>
      <td>${fmtNum(l.hydra_sr, 2)}</td>
      <td class="${colorPos(l.hydra_ret)}">${fmtPctPlain(l.hydra_ret, 2)}</td>
      <td>${fmtNum(w.spy_sr, 2)}</td>
      <td class="${colorPos(w.spy_ret)}">${fmtPctPlain(w.spy_ret, 2)}</td>
    </tr>`;
  }).join("");
  document.getElementById("liteWfTable").innerHTML = h + "<tbody>" + body + "</tbody>";
}
"""

JS_MAIN = r"""
function renderAll() {
  renderDateLabel();
  renderKPIs();
  renderISOOSTables();
  renderWalkforwardTable();
  renderTrailingTable();
  renderPerfRiskTables();
  renderVolTable();
  renderTradeSummaryTable();
  renderLedgerTable();
  renderEtfPositionsTable();
  renderEtfLedgerTable();
  renderEtfHistoryTable();
  renderCalendarTable();
  renderHeatmap();
  renderPortfolioTable();
  renderSleeveTable();
  renderMethodology();
  renderLite();
  renderEquityChart();
  renderDrawdownChart();
  renderRollingSharpeChart();
  renderVolChart();
  renderCalendarChart();
}

document.addEventListener("DOMContentLoaded", renderAll);
"""


def main():
    data = json.loads(JSON_PATH.read_text())

    parts = [
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n',
        '<meta charset="UTF-8">\n',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n',
        '<title>HYDRA — 20-Sleeve Diversified Ensemble</title>\n',
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>\n',
        '<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>\n',
        '<style>', CSS, '</style>\n',
        '</head>\n<body>\n<div class="page">\n',
        NAV,
        HEADER,
        OVERVIEW,
        HOW_IT_WORKS,
        BODY_SECTIONS,
        '</div>\n',
        '<script>\n',
        'const F = ', json.dumps(data), ';\n',
        JS_HELPERS,
        JS_CHARTS,
        JS_TABLES,
        JS_MAIN,
        '</script>\n',
        '</body>\n</html>\n',
    ]
    html = ''.join(parts)
    OUT_PATH.write_text(html)
    print(f"Wrote {OUT_PATH} ({len(html)/1024:.1f} KB)")
    print(f"Sections: CSS {len(CSS)}  body {len(OVERVIEW)+len(HOW_IT_WORKS)+len(BODY_SECTIONS)}  js {len(JS_HELPERS)+len(JS_CHARTS)+len(JS_TABLES)+len(JS_MAIN)}")


if __name__ == "__main__":
    main()
