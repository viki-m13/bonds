"""Build docs/atlas.html from atlas_factsheet_data.json.

ATLAS page mirrors nova34.html with accent swapped to teal-green and the
comparison columns adjusted for ATLAS vs base/60-40/SPY (instead of NOVA's
4x-margin comparison).
"""
import json
from pathlib import Path

ROOT = Path("/home/user/bonds")
JSON_PATH = ROOT / "data/results/atlas_factsheet_data.json"
OUT_PATH = ROOT / "docs/atlas.html"

ACCENT = "#0d9e6d"  # teal-green

CSS = r"""
:root{--bg:#fff;--card:#f8f9fa;--card2:#f0f1f3;--t1:#1a1a2e;--t2:#4a4a68;--t3:#8888a0;--green:#0d9e6d;--red:#d1344b;--blue:#1a56db;--cyan:#0e7490;--yellow:#b45309;--purple:#7c3aed;--border:#e2e4e8;--accent:#0d9e6d}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--t1);line-height:1.5;font-size:14px}
.page{max-width:1000px;margin:0 auto;padding:16px}
.header{display:flex;justify-content:space-between;align-items:flex-start;padding:20px 0 16px;border-bottom:2px solid var(--accent);margin-bottom:16px;flex-wrap:wrap;gap:8px}
.header h1{font-size:1.4rem;color:var(--t1);font-weight:700;letter-spacing:-0.5px}
.header .sub{font-size:0.78rem;color:var(--t2);margin-top:2px}
.header .nav-date{font-size:0.72rem;color:var(--accent);background:var(--card);padding:4px 10px;border-radius:4px;white-space:nowrap}
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
.disclaimer{font-size:0.62rem;color:var(--t3);padding:12px;border:1px solid var(--border);border-radius:4px;margin-top:16px;line-height:1.4}
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
<a href="hydra.html" style="padding:6px 16px;border-radius:20px;background:var(--card);color:var(--t1);text-decoration:none;font-size:0.82rem;font-weight:500;border:1px solid var(--border)">HYDRA</a>
<a href="nova34.html" style="padding:6px 16px;border-radius:20px;background:var(--card);color:var(--t1);text-decoration:none;font-size:0.82rem;font-weight:500;border:1px solid var(--border)">NOVA</a>
<a href="atlas.html" style="padding:6px 16px;border-radius:20px;background:var(--accent);color:#fff;text-decoration:none;font-size:0.82rem;font-weight:600;border:1px solid var(--accent)">ATLAS</a>
</div>
"""

HEADER = """<div class="header">
<div>
<h1>ATLAS &mdash; Drawdown-Hardened TSMOM LETF</h1>
<div class="sub">Time-series momentum on UPRO/TQQQ/TMF/UGL | Drawdown-throttle overlay | Monthly rebal | Pre-registered 2023-2026 holdout passed</div>
</div>
<div class="nav-date" id="dateLabel"></div>
</div>
"""

OVERVIEW = """<!-- STRATEGY OVERVIEW -->
<div class="section">
<div class="card" style="border-left:4px solid var(--accent);font-size:0.8rem;color:var(--t2);line-height:1.7">
<h3 style="color:var(--t1);font-size:0.95rem;margin-bottom:6px">Strategy Overview</h3>
<p style="margin-bottom:6px"><strong style="color:var(--t1)">ATLAS is a LETF strategy designed for client deployability: LETF-scale CAGR with a 60/40-compatible drawdown envelope.</strong> Backtest full-sample CAGR is <strong>15.4%</strong> at Sharpe 0.95 over 15 years with max drawdown <strong>&minus;26.5%</strong>. 0% of 2-year rolling windows end with a drawdown worse than &minus;30% &mdash; matching 60/40 on path discipline while delivering median 2-year CAGR of 15.6% vs 60/40&rsquo;s 10.4%.</p>
<p><strong style="color:var(--t1)">Two layers.</strong> (1) A time-series momentum core, Moskowitz-Ooi-Pedersen style, taking the sign of the trailing 3-month return on SPY/QQQ/TLT/GLD and expressing the bet through the 3&times; LETFs UPRO, TQQQ, TMF, UGL, vol-targeted to 15% annualised with monthly rebalance. (2) A drawdown-throttle overlay that scales exposure based on how far the live strategy sits below its own 252-day peak, floor at 25%. Residual capacity sits in BIL.</p>
<p style="margin-top:6px"><strong style="color:var(--t1)">Why the overlay is the invention.</strong> Every LETF strategy we tested ran a &minus;40 to &minus;80% drawdown at some point &mdash; that is the user-experience problem, not Sharpe. The overlay cuts the full-sample max drawdown from the base strategy&rsquo;s &minus;44% to &minus;26.5% without harming Sharpe. A 500-permutation null test (21-day block shuffle of the multiplier) confirms the MDD reduction is real (p&lt;0.001).</p>
<p style="margin-top:6px"><strong style="color:var(--t1)">Pre-registered holdout.</strong> Parameters were fixed on 2011-01-01 &rarr; 2023-01-01 discovery data. On the 2023-01-01 &rarr; 2026-04-02 holdout, with zero re-tuning, ATLAS delivered <strong>Sharpe 1.07, CAGR 18.1%, MDD &minus;20.0%</strong> &mdash; better than discovery on all three metrics, consistent with a real signal rather than an overfit.</p>
<p style="margin-top:6px"><strong style="color:var(--t1)">What it is not.</strong> Not a Sharpe booster &mdash; the overlay&rsquo;s Sharpe lift is within sampling noise (p=0.15). Its value shows up in worst-case path, not long-run CAGR/vol. Not a tail hedge &mdash; it reacts AFTER drawdown has started, it does not pre-empt. Not tested over 100 years &mdash; 15 years contains only two serious equity bears (2020, 2022) and a multi-decade bond-bull tail.</p>
</div>
</div>
"""

HOW_IT_WORKS = """<!-- HOW IT WORKS -->
<div class="section">
<div class="card" style="border-left:4px solid var(--blue);font-size:0.8rem;color:var(--t2);line-height:1.7">
<h3 style="color:var(--t1);font-size:0.95rem;margin-bottom:6px">How Rebalancing Works</h3>
<table style="font-size:0.78rem;margin-bottom:12px">
<tr><td style="font-weight:700;width:160px;border:none;padding:4px 8px">Signal day (T&minus;1)</td><td style="border:none;padding:4px 8px">At close T&minus;1, compute 63-day log return on each underlying (SPY, QQQ, TLT, GLD). If return &gt; 0, target the corresponding LETF (UPRO/TQQQ/TMF/UGL); if &le; 0, target BIL for that slot.</td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Vol target</td><td style="border:none;padding:4px 8px">Compute 63-day realised vol of the raw target portfolio. Scale weights so ex-ante annualised vol equals 15%, capped at 3&times;. This is the TSMOM base.</td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">DD-throttle</td><td style="border:none;padding:4px 8px">Track the 252-day running max of the TSMOM NAV. Current DD = NAV / peak &minus; 1. Apply a leverage multiplier: 0 to &minus;5% &rarr; 100%; &minus;5% to &minus;10% &rarr; linear 100&rarr;50%; &minus;10% to &minus;20% &rarr; linear 50&rarr;25%; below &minus;20% &rarr; floor 25%. 5-day smooth.</td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Execution</td><td style="border:none;padding:4px 8px">Weights and multiplier set at close T&minus;1 &mdash; execute at next-day open T. Monthly rebalance for the core (21 trading days); overlay adjusts daily. 15 bps transaction cost on |&Delta;w|.</td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Residual</td><td style="border:none;padding:4px 8px">Any capacity not deployed by TSMOM or &times; overlay multiplier is parked in BIL (1-3 month T-bills). Gross exposure never exceeds the multiplier&rsquo;s current cap.</td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Signal lag</td><td style="border:none;padding:4px 8px">Two-day lag throughout: signal computed at close T&minus;1 using data through T&minus;2. No look-ahead. Execution assumed at open T.</td></tr>
</table>
<div style="background:var(--card2);border-radius:4px;padding:12px;font-size:0.76rem;line-height:1.7">
<strong style="color:var(--t1)">Concrete Example &mdash; Entering a drawdown in 2020:</strong>
<div style="margin-top:8px;display:grid;grid-template-columns:auto 1fr;gap:4px 10px;align-items:start;font-size:0.73rem;color:var(--t2)">
<div style="font-weight:700;white-space:nowrap">Jan 2020</div><div>TSMOM long UPRO/TQQQ/TMF/UGL (all four trends positive). NAV at 252d peak &rarr; multiplier = 100%. Gross exposure ~110%.</div>
<div style="font-weight:700;white-space:nowrap">Feb 24 2020</div><div>S&amp;P drops 3.4%. TSMOM NAV now 4% below 252d peak. Multiplier stays 100% (DD &gt; &minus;5%).</div>
<div style="font-weight:700;white-space:nowrap">Mar 9 2020</div><div>NAV now &minus;12% from peak. Multiplier smooths to ~45% (linear zone &minus;10 to &minus;20%). Strategy cuts to 45% gross exposure; 55% sits in BIL.</div>
<div style="font-weight:700;white-space:nowrap">Mar 23 2020</div><div>NAV &minus;18% from peak. Multiplier floors at 25%. Strategy cannot lever further into the tail.</div>
<div style="font-weight:700;white-space:nowrap">Apr &mdash; Jul 2020</div><div>As SPY recovers and TSMOM NAV approaches its peak, multiplier climbs back through the linear zones. Strategy does UNDERPERFORM the TSMOM base on the recovery leg &mdash; that is the cost of the insurance.</div>
</div>
</div>
</div>
</div>
"""

BODY_SECTIONS = """<!-- KPIs -->
<div class="kpi-row" id="kpis"></div>

<!-- GROWTH CHART -->
<div class="section">
<div class="section-title">Growth of $10,000 (ATLAS vs base TSMOM vs 60/40 vs SPY)</div>
<div class="card"><div class="chart-wrap"><canvas id="eqChart"></canvas></div></div>
<div class="card" style="font-size:0.74rem;color:var(--t2);line-height:1.6">
<strong style="color:var(--t1)">Reading the chart.</strong> &ldquo;TSMOM base&rdquo; is the same strategy without the DD-throttle &mdash; the overlay gives up some absolute compounding (ATLAS 15.4% CAGR vs base 20.2%) in exchange for a much tighter drawdown profile. 60/40 SPY/TLT is the humility benchmark: similar drawdown ceiling (~30%) with less than half the CAGR.
</div>
</div>

<!-- IS / OOS -->
<div class="section">
<div class="section-title">In-Sample vs Out-of-Sample</div>
<div class="g2">
<div class="card"><h3>In-Sample (2011-01 &rarr; 2020-12)</h3><table id="isTable"></table></div>
<div class="card"><h3>Out-of-Sample (2021-01 &rarr; present)</h3><table id="oosTable"></table></div>
</div>
<div class="card" style="margin-top:4px;font-size:0.74rem;color:var(--t2)"><strong style="color:var(--t1)">Interpretation.</strong> OOS Sharpe is HIGHER than IS (1.05 vs 0.85), driven by the 2022 bond-crash survival (TSMOM went short TLT via 0-position) and the 2023-2025 equity rally. The pre-registered 2023-2026 holdout (a strict sub-window of OOS) posted SR 1.07, CAGR 18.1%, MDD &minus;20.0% &mdash; all better than IS.</div>
</div>

<!-- COMPARE WITH OTHER SITE STRATEGIES -->
<div class="section">
<div class="section-title">Comparison with Other Strategies on This Site</div>
<div class="card" style="overflow-x:auto"><table id="siteCompareTable"></table></div>
<div class="card" style="margin-top:4px;font-size:0.74rem;color:var(--t2)"><strong style="color:var(--t1)">How to read this.</strong> ATLAS trades Sharpe for drawdown discipline. HYDRA and ZEPHYR are higher-Sharpe research products &mdash; ATLAS&rsquo;s design goal is a LETF that a client can actually hold through a bear market without panic-selling. 60/40-like drawdown, LETF-like CAGR.</div>
</div>

<!-- WALK-FORWARD 3Y -->
<div class="section">
<div class="section-title">Walk-Forward &mdash; Rolling 3-Year Windows</div>
<div class="card" style="overflow-x:auto"><table id="wfTable"></table></div>
<div class="card" style="margin-top:4px;font-size:0.74rem;color:var(--t2)">Five non-overlapping 3-year windows spanning 2011-2026. ATLAS is net-positive in all of them.</div>
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
<div class="section-title">Rolling 6-Month Sharpe</div>
<div class="card"><div class="chart-wrap chart-sm"><canvas id="rsChart"></canvas></div></div>
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

<!-- NOTES -->
<div class="section">
<div class="section-title">Methodology Notes &amp; Caveats</div>
<div class="card" style="font-size:0.76rem;line-height:1.7;color:var(--t2)" id="methodologyNotes"></div>
</div>

<!-- DISCLAIMER -->
<div class="disclaimer">
<strong style="color:var(--t1)">Backtest disclosure.</strong> All figures are from a daily-bar simulation from 2011-01-03 to the most recent close shown above. Signals use T&minus;1 data applied from open T onward. Transaction costs applied at 15 bps on turnover. LETF expense ratios (0.84&ndash;1.1% per year) and swap costs are ON TOP of modelled TC and are not netted into the headline figures. The strategy uses leveraged ETFs &mdash; these carry compounding risk on multi-day holds. The drawdown-throttle cuts but does not eliminate tail risk: a sudden &minus;30% single-day LETF gap remains possible and is not hedged. Past performance is not a guarantee of future returns. This factsheet is for informational purposes only and is not investment advice.
</div>
"""

JS_HELPERS = r"""
const fmtPct = (x, dp = 2) => (x == null || isNaN(x)) ? "—" : (x >= 0 ? "+" : "") + Number(x).toFixed(dp) + "%";
const fmtPctPlain = (x, dp = 2) => (x == null || isNaN(x)) ? "—" : Number(x).toFixed(dp) + "%";
const fmtNum = (x, dp = 2) => (x == null || isNaN(x)) ? "—" : Number(x).toFixed(dp);
const colorPos = x => x == null ? "" : (x >= 0 ? "pos" : "neg");

function renderDateLabel() {
  const d = new Date(F.last_updated);
  const opts = { year: "numeric", month: "short", day: "numeric" };
  document.getElementById("dateLabel").textContent = "Data as of " + d.toLocaleDateString("en-US", opts);
}

function renderKPIs() {
  const m = F.metrics.ATLAS;
  const kpis = [
    { label: "CAGR", val: fmtPctPlain(m.ann_return, 1) },
    { label: "Sharpe", val: fmtNum(m.sharpe, 2) },
    { label: "Vol", val: fmtPctPlain(m.ann_vol, 1) },
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
  const atlas = F.equity_curve.map(r => r.ATLAS);
  const base = F.equity_curve.map(r => r.ATLAS_BASE_1x);
  const s6040 = F.equity_curve.map(r => r.SIXTY40);
  const spy = F.equity_curve.map(r => r.SPY);
  new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "ATLAS", data: atlas, borderColor: "#0d9e6d", backgroundColor: "rgba(13,158,109,0.12)", borderWidth: 2, pointRadius: 0, fill: true, tension: 0.1 },
        { label: "TSMOM base (no overlay)", data: base, borderColor: "#b45309", borderWidth: 1.3, pointRadius: 0, fill: false, borderDash: [4, 3] },
        { label: "60/40 SPY/TLT", data: s6040, borderColor: "#7c3aed", borderWidth: 1.2, pointRadius: 0, fill: false, borderDash: [2, 2] },
        { label: "SPY", data: spy, borderColor: "#1a1a2e", borderWidth: 1.1, pointRadius: 0, fill: false },
      ],
    },
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
        { label: "6M Sharpe", data: sr, borderColor: "#0d9e6d", backgroundColor: "rgba(13,158,109,0.12)", borderWidth: 1.3, pointRadius: 0, fill: true, tension: 0.1 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: { type: "time", time: { unit: "year" }, ticks: { font: { size: 10 } } }, y: { ticks: { font: { size: 10 } } } },
      plugins: { legend: { display: false } },
    },
  });
}

function renderCalendarChart() {
  const ctx = document.getElementById("calChart").getContext("2d");
  const labels = F.calendar_returns.map(r => r.year);
  const atlas = F.calendar_returns.map(r => r.ret);
  const sm = {}; F.calendar_spy.forEach(r => sm[r.year] = r.ret);
  const spy = labels.map(y => sm[y] != null ? sm[y] : 0);
  new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "ATLAS", data: atlas, backgroundColor: "rgba(13,158,109,0.85)" },
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

function renderSiteCompareTable() {
  const rows = [
    ["Strategy", "Years", "CAGR", "Sharpe", "OOS Sharpe", "Max DD"],
    ["ATLAS (this page)",
      F.metrics.ATLAS.n_years.toFixed(1),
      fmtPctPlain(F.metrics.ATLAS.ann_return, 1),
      fmtNum(F.metrics.ATLAS.sharpe, 2),
      fmtNum(F.oos_metrics.sharpe, 2),
      fmtPctPlain(F.metrics.ATLAS.max_dd, 1)],
    ["TSMOM base (no overlay)",
      F.metrics.ATLAS_BASE_1x.n_years.toFixed(1),
      fmtPctPlain(F.metrics.ATLAS_BASE_1x.ann_return, 1),
      fmtNum(F.metrics.ATLAS_BASE_1x.sharpe, 2),
      "—",
      fmtPctPlain(F.metrics.ATLAS_BASE_1x.max_dd, 1)],
    ["NOVA",  "8.9",  "40.38%", "1.69", "1.61", "-29.18%"],
    ["HYDRA", "21.0", "11.47%", "1.58", "2.01", "-18.74%"],
    ["ZEPHYR (BLEND)", "20.2", "16.85%", "2.83", "3.70", "-9.41%"],
    ["AURORA", "20.0", "13.90%", "1.55", "1.80", "-14.80%"],
    ["60/40 SPY/TLT",
      F.metrics.SIXTY40.n_years.toFixed(1),
      fmtPctPlain(F.metrics.SIXTY40.ann_return, 1),
      fmtNum(F.metrics.SIXTY40.sharpe, 2),
      "—",
      fmtPctPlain(F.metrics.SIXTY40.max_dd, 1)],
    ["SPY", F.metrics.SPY.n_years.toFixed(1),
      fmtPctPlain(F.metrics.SPY.ann_return, 1),
      fmtNum(F.metrics.SPY.sharpe, 2),
      "—",
      fmtPctPlain(F.metrics.SPY.max_dd, 1)],
  ];
  const head = "<thead><tr>" + rows[0].map(c => `<th>${c}</th>`).join("") + "</tr></thead>";
  const body = "<tbody>" + rows.slice(1).map(r => {
    return `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td><td>${r[4]}</td><td class="neg">${r[5]}</td></tr>`;
  }).join("") + "</tbody>";
  document.getElementById("siteCompareTable").innerHTML = head + body;
}

function renderWalkforwardTable() {
  const h = "<thead><tr><th>Window</th><th>ATLAS SR</th><th>ATLAS Ret</th><th>ATLAS MDD</th><th>SPY SR</th><th>SPY Ret</th><th>SPY MDD</th></tr></thead>";
  const body = F.walkforward_3y.map(w => `<tr>
    <td>${w.window}</td>
    <td>${fmtNum(w.atlas_sr, 2)}</td>
    <td class="${colorPos(w.atlas_ret)}">${fmtPctPlain(w.atlas_ret, 2)}</td>
    <td class="neg">${fmtPctPlain(w.atlas_mdd, 2)}</td>
    <td>${fmtNum(w.spy_sr, 2)}</td>
    <td class="${colorPos(w.spy_ret)}">${fmtPctPlain(w.spy_ret, 2)}</td>
    <td class="neg">${fmtPctPlain(w.spy_mdd, 2)}</td>
  </tr>`).join("");
  document.getElementById("wfTable").innerHTML = h + "<tbody>" + body + "</tbody>";
}

function renderTrailingTable() {
  const periods = ["1M", "3M", "6M", "YTD", "1Y", "3Y_ann", "5Y_ann", "SI_ann"];
  const labels = ["1M", "3M", "6M", "YTD", "1Y", "3Y (ann)", "5Y (ann)", "Since Incep (ann)"];
  const h = "<thead><tr><th></th>" + labels.map(l => `<th>${l}</th>`).join("") + "</tr></thead>";
  const row = (name, d) => `<tr><td>${name}</td>` + periods.map(p => {
    const v = d[p];
    return `<td class="${colorPos(v)}">${fmtPctPlain(v, 2)}</td>`;
  }).join("") + "</tr>";
  document.getElementById("trailingTable").innerHTML = h + "<tbody>"
    + row("ATLAS", F.trailing.ATLAS)
    + row("60/40", F.trailing.SIXTY40)
    + row("SPY", F.trailing.SPY)
    + "</tbody>";
}

function renderPerfRiskTables() {
  const m = F.metrics.ATLAS, s = F.metrics.SPY, b = F.metrics.SIXTY40;
  const perfRows = [
    ["", "ATLAS", "SPY", "60/40"],
    ["Ann. Return", fmtPctPlain(m.ann_return, 2), fmtPctPlain(s.ann_return, 2), fmtPctPlain(b.ann_return, 2)],
    ["Cumulative NAVx", fmtNum(F.nav_x, 1) + "x",
      fmtNum(Math.pow(1 + s.ann_return / 100, m.n_years), 1) + "x",
      fmtNum(Math.pow(1 + b.ann_return / 100, m.n_years), 1) + "x"],
    ["Sharpe", fmtNum(m.sharpe, 2), fmtNum(s.sharpe, 2), fmtNum(b.sharpe, 2)],
    ["Sortino", fmtNum(m.sortino, 2), fmtNum(s.sortino, 2), fmtNum(b.sortino, 2)],
    ["Years", fmtNum(m.n_years, 1), fmtNum(s.n_years, 1), fmtNum(b.n_years, 1)],
  ];
  const riskRows = [
    ["", "ATLAS", "SPY", "60/40"],
    ["Ann. Vol", fmtPctPlain(m.ann_vol, 2), fmtPctPlain(s.ann_vol, 2), fmtPctPlain(b.ann_vol, 2)],
    ["Max Drawdown", fmtPctPlain(m.max_dd, 2), fmtPctPlain(s.max_dd, 2), fmtPctPlain(b.max_dd, 2)],
    ["Return / Vol", fmtNum(m.ann_return / m.ann_vol, 2), fmtNum(s.ann_return / s.ann_vol, 2), fmtNum(b.ann_return / b.ann_vol, 2)],
    ["Return / |MDD|", fmtNum(m.ann_return / Math.abs(m.max_dd), 2), fmtNum(s.ann_return / Math.abs(s.max_dd), 2), fmtNum(b.ann_return / Math.abs(b.max_dd), 2)],
  ];
  const mkTable = (rows) => {
    const head = "<thead><tr>" + rows[0].map((c, i) => i === 0 ? `<th></th>` : `<th>${c}</th>`).join("") + "</tr></thead>";
    const body = "<tbody>" + rows.slice(1).map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`).join("") + "</tbody>";
    return head + body;
  };
  document.getElementById("perfTable").innerHTML = mkTable(perfRows);
  document.getElementById("riskTable").innerHTML = mkTable(riskRows);
}

function renderCalendarTable() {
  const h = "<thead><tr><th>Year</th><th>ATLAS</th><th>Base TSMOM</th><th>60/40</th><th>SPY</th><th>ATLAS-SPY</th></tr></thead>";
  const sm = {}; F.calendar_spy.forEach(r => sm[r.year] = r.ret);
  const bm = {}; F.calendar_base_1x.forEach(r => bm[r.year] = r.ret);
  const mm = {}; F.calendar_sixty40.forEach(r => mm[r.year] = r.ret);
  const body = F.calendar_returns.map(r => {
    const sv = sm[r.year]; const bv = bm[r.year]; const mv = mm[r.year];
    const diff = sv != null ? r.ret - sv : null;
    return `<tr>
      <td>${r.year}</td>
      <td class="${colorPos(r.ret)}">${fmtPctPlain(r.ret, 2)}</td>
      <td class="${colorPos(bv)}">${bv != null ? fmtPctPlain(bv, 2) : "—"}</td>
      <td class="${colorPos(mv)}">${mv != null ? fmtPctPlain(mv, 2) : "—"}</td>
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
    const a = Math.min(1, Math.abs(r) / 15);
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
      html += `<div class="hm-c" style="background:${colorFor(r)};color:${r != null && Math.abs(r) > 8 ? '#fff' : 'var(--t1)'}" title="${y}-${String(m).padStart(2, "0")}: ${txt}%">${txt}</div>`;
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

function renderMethodology() {
  const n = F.notes;
  document.getElementById("methodologyNotes").innerHTML =
    `<p><strong style="color:var(--t1)">Key idea.</strong> ${n.key_idea}</p>
     <p style="margin-top:6px"><strong style="color:var(--t1)">DD-throttle overlay.</strong> ${n.dd_throttle}</p>
     <p style="margin-top:6px"><strong style="color:var(--t1)">Validation.</strong> ${n.validation}</p>
     <p style="margin-top:6px"><strong style="color:var(--t1)">Caveats.</strong> ${n.caveats}</p>`;
}
"""

JS_MAIN = r"""
function renderAll() {
  renderDateLabel();
  renderKPIs();
  renderISOOSTables();
  renderSiteCompareTable();
  renderWalkforwardTable();
  renderTrailingTable();
  renderPerfRiskTables();
  renderCalendarTable();
  renderHeatmap();
  renderMethodology();
  renderEquityChart();
  renderDrawdownChart();
  renderRollingSharpeChart();
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
        '<title>ATLAS — Drawdown-Hardened TSMOM LETF</title>\n',
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


if __name__ == "__main__":
    main()
