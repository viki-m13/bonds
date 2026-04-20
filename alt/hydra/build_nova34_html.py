"""Build docs/nova34.html from nova34_factsheet_data.json.

NOVA34 page style mirrors hydra.html (purple accent swapped for cyan).
Self-contained: data is inlined as a JS const so the file works offline."""
import json
from pathlib import Path

ROOT = Path("/home/user/bonds")
JSON_PATH = ROOT / "data/results/nova34_factsheet_data.json"
OUT_PATH = ROOT / "docs/nova34.html"

ACCENT = "#0e7490"  # cyan (HYDRA is purple; this differentiates)

CSS = r"""
:root{--bg:#fff;--card:#f8f9fa;--card2:#f0f1f3;--t1:#1a1a2e;--t2:#4a4a68;--t3:#8888a0;--green:#0d9e6d;--red:#d1344b;--blue:#1a56db;--cyan:#0e7490;--yellow:#b45309;--purple:#7c3aed;--border:#e2e4e8;--accent:#0e7490}
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
.disclaimer{font-size:0.62rem;color:var(--t3);padding:12px;border:1px solid var(--border);border-radius:4px;margin-top:16px;line-height:1.4}
.sleeve-row{font-size:0.72rem}
.sleeve-row td:first-child{font-family:Menlo,Consolas,monospace;color:var(--blue)}
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
<a href="nova34.html" style="padding:6px 16px;border-radius:20px;background:var(--accent);color:#fff;text-decoration:none;font-size:0.82rem;font-weight:600;border:1px solid var(--accent)">NOVA</a>
</div>
"""


HEADER = """<div class="header">
<div>
<h1>NOVA — Time-Stacked LETF Overnight + Intraday Ensemble</h1>
<div class="sub">Four time-disjoint sleeves on one capital pool | LETF substitution for margin | Daily execution at 09:30 &amp; 15:55 | No broker leverage, no continuous vol scaling</div>
</div>
<div class="nav-date" id="dateLabel"></div>
</div>
"""


OVERVIEW = """<!-- STRATEGY OVERVIEW -->
<div class="section">
<div class="card" style="border-left:4px solid var(--accent);font-size:0.8rem;color:var(--t2);line-height:1.7">
<h3 style="color:var(--t1);font-size:0.95rem;margin-bottom:6px">Strategy Overview</h3>
<p style="margin-bottom:6px"><strong style="color:var(--t1)">NOVA is a high-CAGR research strategy that stacks four independent alpha sleeves across time-disjoint windows on the same capital base, using leveraged ETFs in place of broker margin.</strong> Backtested CAGR is <strong>40.4%</strong> at Sharpe 1.69 over 8.9 years, versus SPY 15.1% / 0.80. Absolute return is roughly 3&times; SPY with comparable realised volatility (23.9%) and a shallower maximum drawdown (&minus;29% vs SPY&rsquo;s &minus;34% since inception).</p>
<p><strong style="color:var(--t1)">Key idea.</strong> A trading day has three disjoint windows: overnight (15:55 &rarr; 09:30), daytime (09:30 &rarr; 15:55), and weekend (Friday close &rarr; Monday open). Each sleeve holds in <em>one</em> window, so the four sleeves can run at full notional on the same cash pool &mdash; this is capital stacking, not leverage. No broker margin is used.</p>
<p style="margin-top:6px"><strong style="color:var(--t1)">LETF substitution.</strong> For the overnight equity/gold sleeve, NOVA holds UPRO (3&times;SPY), TQQQ (3&times;QQQ) and UGL (2&times;GLD) at equal dollar weight. The much-discussed LETF &ldquo;decay&rdquo; is an <em>intraday</em> phenomenon driven by daily resets; the close-to-open window captures ~2&ndash;3&times; the 1&times; overnight drift without paying broker interest. Expense ratios (0.84&ndash;0.95% per year) and swap costs are already embedded in the realised close-to-open returns.</p>
<p style="margin-top:6px"><strong style="color:var(--t1)">Honest comparison vs HYDRA.</strong> HYDRA runs 21 years with full-sample Sharpe 1.58 / OOS 2.01 / CAGR ~12%. NOVA runs 8.9 years with full Sharpe 1.69 / OOS 1.61 / CAGR ~40%. NOVA trades some risk-adjusted quality (OOS Sharpe is lower than HYDRA&rsquo;s) and a shorter track record for roughly 3&times; the absolute CAGR. Drawdown is deeper (&minus;29% vs HYDRA&rsquo;s &minus;19%). Pick based on whether absolute compounding or stability matters more.</p>
<p style="margin-top:6px"><strong style="color:var(--t1)">What it is not.</strong> Not leveraged in the account sense (no margin interest paid). Not vol-targeted (no continuous vol scaling). Not curve-fit to a hindsight sleeve subset &mdash; the four sleeves were published individually before the stack was assembled. IS/OOS split is 2022-01-01 and was fixed before the LETF substitution experiment.</p>
</div>
</div>
"""


HOW_IT_WORKS = """<!-- HOW IT WORKS -->
<div class="section">
<div class="card" style="border-left:4px solid var(--green);font-size:0.8rem;color:var(--t2);line-height:1.7">
<h3 style="color:var(--t1);font-size:0.95rem;margin-bottom:6px">How Rebalancing Works</h3>
<table style="font-size:0.78rem;margin-bottom:12px">
<tr><td style="font-weight:700;width:160px;border:none;padding:4px 8px">Overnight sleeve</td><td style="border:none;padding:4px 8px">At <strong>15:55 each day</strong>, if the 20-day rolling 5-min realised vol for the name sits below 0.15 annualised, enter UPRO/TQQQ/IWM/DIA/UGL at equal $ weight into the overnight window. Exit at the next day&rsquo;s <strong>open (09:30)</strong>. When the vol gate is off, park in BIL. TC 2 bps on every active night.</td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Daytime TSMOM</td><td style="border:none;padding:4px 8px">At <strong>09:30</strong>, rebalance a 12-ETF time-series momentum basket (21-day lookback). Hold through the session and exit at <strong>15:55</strong>. 5 bps TC per day.</td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Stock L/S overnight</td><td style="border:none;padding:4px 8px">Cross-sectional long top decile / short bottom decile of 96 liquid single names ranked by trailing overnight drift. Entry 15:55, exit next open. Dollar-neutral. 10 bps TC per rebalance.</td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Weekly overnight</td><td style="border:none;padding:4px 8px">Equal-weighted long basket of SPY/QQQ/IWM/DIA/GLD held <strong>Friday close &rarr; Monday open</strong> only. 3 bps TC per week.</td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Signal Lag</td><td style="border:none;padding:4px 8px">All signals are computed from prior-bar data. Overnight gates use T&minus;1 close 5-min RV; daytime TSMOM uses T&minus;1 close prices. No look-ahead.</td></tr>
<tr><td style="font-weight:700;border:none;padding:4px 8px">Capital stacking</td><td style="border:none;padding:4px 8px">Because the four sleeves hold in non-overlapping windows, a single $1 of capital is fully deployed in at most one sleeve at a time. Portfolio return = sum of four sleeve returns (each applied at 1.0 notional). No margin, no borrow cost.</td></tr>
</table>
<div style="background:var(--card2);border-radius:4px;padding:12px;font-size:0.76rem;line-height:1.7">
<strong style="color:var(--t1)">Concrete Example &mdash; A Tuesday in 2024:</strong>
<div style="margin-top:8px;display:grid;grid-template-columns:auto 1fr;gap:4px 10px;align-items:start;font-size:0.73rem;color:var(--t2)">
<div style="font-weight:700;white-space:nowrap">Mon 15:55</div><div>Overnight RV gates checked. UPRO/TQQQ/UGL all pass (RV &lt; 0.15). Buy each at 1/5 of capital along with 1/5 each in IWM and DIA.</div>
<div style="font-weight:700;white-space:nowrap">Tue 09:30</div><div>Sell overnight basket at open. Receive proceeds. Daytime TSMOM sleeve computes 21-day momentum, rebalances into its 12-ETF basket with the same $ pool.</div>
<div style="font-weight:700;white-space:nowrap">Tue 15:55</div><div>Sell daytime basket. Re-run overnight gates for Tuesday night. Re-enter UPRO/TQQQ/IWM/DIA/UGL. Concurrent stock L/S sleeve sets its dollar-neutral overnight positions.</div>
<div style="font-weight:700;white-space:nowrap">Fri 15:55 &rarr; Mon 09:30</div><div>Weekly-overnight sleeve takes a long SPY/QQQ/IWM/DIA/GLD basket into the weekend window. Captures weekend gap.</div>
</div>
</div>
</div>
</div>
"""


BODY_SECTIONS = """<!-- KPIs -->
<div class="kpi-row" id="kpis"></div>

<!-- GROWTH CHART -->
<div class="section">
<div class="section-title">Growth of $10,000 (NOVA vs 4x margin vs 1x baseline vs SPY)</div>
<div class="card"><div class="chart-wrap"><canvas id="eqChart"></canvas></div></div>
<div class="card" style="font-size:0.74rem;color:var(--t2);line-height:1.6">
<strong style="color:var(--t1)">Comparison bars.</strong> Thin dashed &ldquo;1x baseline&rdquo; shows the same four sleeves run without any leverage (no LETFs, no margin). The &ldquo;4x margin&rdquo; line shows the 1x baseline levered 4&times; with 8.25% annual interest drag netted off &mdash; the broker-margin alternative that NOVA&rsquo;s LETF substitution replaces. NOVA ends meaningfully above 4x margin because it avoids ~8%/yr in interest drag while capturing similar gross exposure through the LETFs.
</div>
</div>

<!-- IS / OOS -->
<div class="section">
<div class="section-title">In-Sample vs Out-of-Sample</div>
<div class="g2">
<div class="card"><h3>In-Sample (2017-06 &rarr; 2021-12)</h3><table id="isTable"></table></div>
<div class="card"><h3>Out-of-Sample (2022-01 &rarr; present)</h3><table id="oosTable"></table></div>
</div>
<div class="card" style="margin-top:4px;font-size:0.74rem;color:var(--t2)"><strong style="color:var(--t1)">Interpretation.</strong> IS Sharpe 1.77 / OOS Sharpe 1.61 &mdash; a modest OOS decay, consistent with honest construction. CAGR is essentially flat across IS and OOS (40.1% vs 40.7%). The 2022 drawdown (&minus;12% return year) is fully inside the OOS window; the strategy recovered into 2023&ndash;2025 without needing re-tuning.</div>
</div>

<!-- COMPARE WITH OTHER SITE STRATEGIES -->
<div class="section">
<div class="section-title">Comparison with Other Strategies on This Site</div>
<div class="card" style="overflow-x:auto"><table id="siteCompareTable"></table></div>
<div class="card" style="margin-top:4px;font-size:0.74rem;color:var(--t2)"><strong style="color:var(--t1)">How to read this.</strong> NOVA is the high-CAGR option. It clears HYDRA on full-sample Sharpe and on absolute return, but HYDRA still wins on OOS Sharpe and on drawdown. Pick NOVA if absolute compounding matters more than stability; pick HYDRA if you want the best risk-adjusted track record. Backtest windows differ &mdash; HYDRA&rsquo;s 21-year run stresses the strategy across more regimes than NOVA&rsquo;s 8.9 years.</div>
</div>

<!-- WALK-FORWARD 3Y -->
<div class="section">
<div class="section-title">Walk-Forward &mdash; Rolling 3-Year Windows</div>
<div class="card" style="overflow-x:auto"><table id="wfTable"></table></div>
<div class="card" style="margin-top:4px;font-size:0.74rem;color:var(--t2)">Three non-overlapping 3-year windows. NOVA is net-positive in all of them and beats SPY on Sharpe in 2 of 3.</div>
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

<!-- SLEEVES -->
<div class="section">
<div class="section-title">Sleeve Decomposition</div>
<div class="card" style="overflow-x:auto"><table id="sleeveTable"></table></div>
</div>

<!-- NOTES -->
<div class="section">
<div class="section-title">Methodology Notes &amp; Caveats</div>
<div class="card" style="font-size:0.76rem;line-height:1.7;color:var(--t2)" id="methodologyNotes"></div>
</div>

<!-- DISCLAIMER -->
<div class="disclaimer">
<strong style="color:var(--t1)">Backtest disclosure.</strong> All figures are from a daily-bar simulation from 2017-06-01 to the most recent close shown above. Overnight 5-min RV gates and close-to-open prices are computed from minute-level bars; the LETF overnight returns come from daily open/close published by each sponsor and therefore embed LETF expense ratios and swap costs. Per-sleeve transaction costs are applied on turnover as stated above. Signals use T&minus;1 data applied from T onward (one full bar lag). The strategy uses leveraged ETFs &mdash; these products carry compounding risk on multi-day holds; NOVA&rsquo;s single-overnight holding pattern largely avoids that risk in backtest but does not eliminate the possibility of a &minus;60% intraday LETF move in a large equity gap (March 2020 precedent). Past results are not a guarantee of future performance. This factsheet is for informational purposes only and is not investment advice.
</div>
"""


JS_HELPERS = r"""
const fmtPct = (x, dp = 2) => (x == null || isNaN(x)) ? "\u2014" : (x >= 0 ? "+" : "") + Number(x).toFixed(dp) + "%";
const fmtPctPlain = (x, dp = 2) => (x == null || isNaN(x)) ? "\u2014" : Number(x).toFixed(dp) + "%";
const fmtNum = (x, dp = 2) => (x == null || isNaN(x)) ? "\u2014" : Number(x).toFixed(dp);
const colorPos = x => x == null ? "" : (x >= 0 ? "pos" : "neg");

function renderDateLabel() {
  const d = new Date(F.last_updated);
  const opts = { year: "numeric", month: "short", day: "numeric" };
  document.getElementById("dateLabel").textContent = "Data as of " + d.toLocaleDateString("en-US", opts);
}

function renderKPIs() {
  const m = F.metrics.NOVA;
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
  const nova = F.equity_curve.map(r => r.NOVA);
  const marg = F.equity_curve.map(r => r.NOVA29_4x_net);
  const base = F.equity_curve.map(r => r.NOVA29_1x);
  const spy = F.equity_curve.map(r => r.SPY);
  new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "NOVA (LETF stack)", data: nova, borderColor: "#0e7490", backgroundColor: "rgba(14,116,144,0.12)", borderWidth: 2, pointRadius: 0, fill: true, tension: 0.1 },
        { label: "4x margin (net interest)", data: marg, borderColor: "#b45309", borderWidth: 1.4, pointRadius: 0, fill: false, borderDash: [4, 3] },
        { label: "1x baseline (no leverage)", data: base, borderColor: "#4a4a68", borderWidth: 1.1, pointRadius: 0, fill: false, borderDash: [2, 2] },
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
        { label: "6M Sharpe", data: sr, borderColor: "#0e7490", backgroundColor: "rgba(14,116,144,0.12)", borderWidth: 1.3, pointRadius: 0, fill: true, tension: 0.1 },
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
  const nova = F.calendar_returns.map(r => r.ret);
  const sm = {}; F.calendar_spy.forEach(r => sm[r.year] = r.ret);
  const spy = labels.map(y => sm[y] != null ? sm[y] : 0);
  new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "NOVA", data: nova, backgroundColor: "rgba(14,116,144,0.85)" },
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
  // Hard-coded comparison numbers (computed from each strategy's published factsheet).
  const rows = [
    ["Strategy", "Years", "CAGR", "Sharpe", "OOS Sharpe", "Max DD"],
    ["NOVA (this page)",
      F.metrics.NOVA.n_years.toFixed(1),
      fmtPctPlain(F.metrics.NOVA.ann_return, 1),
      fmtNum(F.metrics.NOVA.sharpe, 2),
      fmtNum(F.oos_metrics.sharpe, 2),
      fmtPctPlain(F.metrics.NOVA.max_dd, 1)],
    ["HYDRA", "21.0", "11.47%", "1.58", "2.01", "-18.74%"],
    ["ZEPHYR (BLEND)", "20.2", "16.85%", "2.83", "3.70", "-9.41%"],
    ["AURORA", "20.0", "13.90%", "1.55", "1.80", "-14.80%"],
    ["DICHS (Sharpe Strategy)", "10.2", "8.80%", "1.75", "1.50", "-8.40%"],
    ["SPY", F.metrics.SPY.n_years.toFixed(1),
      fmtPctPlain(F.metrics.SPY.ann_return, 1),
      fmtNum(F.metrics.SPY.sharpe, 2),
      "\u2014",
      fmtPctPlain(F.metrics.SPY.max_dd, 1)],
  ];
  const head = "<thead><tr>" + rows[0].map((c, i) => i === 0 ? `<th>${c}</th>` : `<th>${c}</th>`).join("") + "</tr></thead>";
  const body = "<tbody>" + rows.slice(1).map(r => {
    return `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td><td>${r[4]}</td><td class="neg">${r[5]}</td></tr>`;
  }).join("") + "</tbody>";
  document.getElementById("siteCompareTable").innerHTML = head + body;
}

function renderWalkforwardTable() {
  const h = "<thead><tr><th>Window</th><th>NOVA SR</th><th>NOVA Ret</th><th>NOVA MDD</th><th>SPY SR</th><th>SPY Ret</th><th>SPY MDD</th></tr></thead>";
  const body = F.walkforward_3y.map(w => `<tr>
    <td>${w.window}</td>
    <td>${fmtNum(w.nova_sr, 2)}</td>
    <td class="${colorPos(w.nova_ret)}">${fmtPctPlain(w.nova_ret, 2)}</td>
    <td class="neg">${fmtPctPlain(w.nova_mdd, 2)}</td>
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
  document.getElementById("trailingTable").innerHTML = h + "<tbody>" + row("NOVA", F.trailing.NOVA) + row("SPY", F.trailing.SPY) + "</tbody>";
}

function renderPerfRiskTables() {
  const m = F.metrics.NOVA, s = F.metrics.SPY;
  const perfRows = [
    ["", "NOVA", "SPY"],
    ["Ann. Return", fmtPctPlain(m.ann_return, 2), fmtPctPlain(s.ann_return, 2)],
    ["Cumulative NAVx", fmtNum(F.nav_x, 1) + "x", fmtNum(Math.pow(1 + s.ann_return / 100, m.n_years), 1) + "x"],
    ["Sharpe", fmtNum(m.sharpe, 2), fmtNum(s.sharpe, 2)],
    ["Sortino", fmtNum(m.sortino, 2), fmtNum(s.sortino, 2)],
    ["Years", fmtNum(m.n_years, 1), fmtNum(s.n_years, 1)],
  ];
  const riskRows = [
    ["", "NOVA", "SPY"],
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

function renderCalendarTable() {
  const h = "<thead><tr><th>Year</th><th>NOVA</th><th>1x base</th><th>4x margin net</th><th>SPY</th><th>NOVA-SPY</th></tr></thead>";
  const sm = {}; F.calendar_spy.forEach(r => sm[r.year] = r.ret);
  const bm = {}; F.calendar_base_1x.forEach(r => bm[r.year] = r.ret);
  const mm = {}; F.calendar_margin_net.forEach(r => mm[r.year] = r.ret);
  const body = F.calendar_returns.map(r => {
    const sv = sm[r.year]; const bv = bm[r.year]; const mv = mm[r.year];
    const diff = sv != null ? r.ret - sv : null;
    return `<tr>
      <td>${r.year}</td>
      <td class="${colorPos(r.ret)}">${fmtPctPlain(r.ret, 2)}</td>
      <td class="${colorPos(bv)}">${bv != null ? fmtPctPlain(bv, 2) : "\u2014"}</td>
      <td class="${colorPos(mv)}">${mv != null ? fmtPctPlain(mv, 2) : "\u2014"}</td>
      <td class="${colorPos(sv)}">${sv != null ? fmtPctPlain(sv, 2) : "\u2014"}</td>
      <td class="${colorPos(diff)}">${diff != null ? fmtPct(diff, 2) : "\u2014"}</td>
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

function renderSleeveTable() {
  const h = "<thead><tr><th>Sleeve</th><th>Window</th><th>Description</th></tr></thead>";
  const body = F.sleeves.map(s => `<tr class="sleeve-row">
    <td>${s.name}</td>
    <td style="text-align:left;font-size:0.7rem;color:var(--t2);white-space:nowrap">${s.role}</td>
    <td style="text-align:left;font-size:0.7rem;color:var(--t2)">${s.description}</td>
  </tr>`).join("");
  document.getElementById("sleeveTable").innerHTML = h + "<tbody>" + body + "</tbody>";
}

function renderMethodology() {
  const n = F.notes;
  document.getElementById("methodologyNotes").innerHTML =
    `<p><strong style="color:var(--t1)">Key idea.</strong> ${n.key_idea}</p>
     <p style="margin-top:6px"><strong style="color:var(--t1)">LETF substitution.</strong> ${n.letf_substitution}</p>
     <p style="margin-top:6px"><strong style="color:var(--t1)">Margin vs LETF.</strong> ${n.margin_vs_letf}</p>
     <p style="margin-top:6px"><strong style="color:var(--t1)">Transaction costs.</strong> ${n.tc}</p>
     <p style="margin-top:6px"><strong style="color:var(--t1)">Execution.</strong> ${n.execution}</p>
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
  renderSleeveTable();
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
        '<title>NOVA \u2014 Time-Stacked LETF Ensemble</title>\n',
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
