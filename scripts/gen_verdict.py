"""Generate docs/verdict.html — "The QQQ-DCA Verdict".

A plain-English, evidence-first page arguing the repo's most-replicated finding:
across 250+ tested configurations, nothing honestly beats biweekly/monthly
DCA-into-QQQ on this data except taking more risk (leverage). Every number on
the page is sourced from committed research records; the equity curves are
recomputed here from the committed ETF panel (no network).

Run:  python3 scripts/gen_verdict.py
"""
import os, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- curves (real, from committed panel)
P = pd.read_pickle(f"{ROOT}/leverage_etf_dca/scripts/_etf_panel.pkl")
close = P["close"].sort_index(); retd = close.pct_change()
mgrid = pd.DatetimeIndex(sorted(close.groupby(close.index.to_period("M")).apply(lambda x: x.index[-1]).values))
mret = close.reindex(mgrid).pct_change()

def volt_weight():
    def av(w): return (retd["TQQQ"].rolling(w, min_periods=int(w*.7)).std()*np.sqrt(252)).reindex(mgrid, method="ffill")
    vs = av(63)*((av(20)/av(63)).clip(lower=1.0)**2.0)
    w = (0.30/vs).clip(0, 1)
    q = close["QQQ"]
    sec = (q > q.rolling(200, min_periods=120).mean()).reindex(mgrid, method="ffill")
    ma50 = q.rolling(50, min_periods=25).mean().reindex(mgrid, method="ffill")
    px = q.reindex(mgrid, method="ffill")
    boost = (1.0 + 6.0*(ma50/px - 1.0).clip(lower=0.0).where(sec.fillna(False), 0.0)).clip(1.0, 2.5)
    return (w*boost).clip(0, 1)

def volt_ret(start="2006-01", end="2026-12"):
    w = volt_weight(); rr = []; prev = 0.0
    for dt in mgrid[(mgrid >= pd.Timestamp(start)) & (mgrid <= pd.Timestamp(end))][1:]:
        i = mgrid.get_loc(dt); wt = w.iloc[i-1]
        if not np.isfinite(wt): continue
        rt = mret["TQQQ"].loc[dt]; rd = np.nanmean([mret[d].loc[dt] for d in ("GLD", "TLT")])
        if not (np.isfinite(rt) and np.isfinite(rd)): continue
        rr.append((dt, wt*rt + (1-wt)*rd - abs(wt-prev)*0.002)); prev = wt
    return pd.Series(dict(rr))

def dca_curve(r, c=1000.0):
    v = 0.0; out = []
    for dt, x in r.items():
        v = (v + c)*(1 + x); out.append((dt, v))
    return pd.Series(dict(out))

rv = volt_ret(); idx = rv.index
CURVES = {"VOLT (leverage dial)": dca_curve(rv)}
for t, nm in [("QQQ", "QQQ-DCA"), ("SPY", "SPY-DCA"), ("TQQQ", "TQQQ-DCA (raw 3×)")]:
    CURVES[nm] = dca_curve(mret[t].loc[idx])
CURVES["Contributions"] = pd.Series(np.arange(1, len(idx)+1)*1000.0, index=idx)

# ---------------------------------------------------------------- SVG helpers
def svg_line_chart(series, W=700, H=300, logy=True, colors=None, dashes=None, ylab="$", tick_years=4):
    keys = list(series.keys()); n = len(series[keys[0]])
    xs = list(range(n)); pad_l, pad_r, pad_t, pad_b = 52, 8, 8, 22
    allv = [v for k in keys for v in series[k] if v and v > 0]
    lo, hi = min(allv), max(allv)
    f = (lambda v: math.log10(v)) if logy else (lambda v: v)
    flo, fhi = f(lo), f(hi)
    def X(i): return pad_l + (W-pad_l-pad_r)*i/(n-1)
    def Y(v): return pad_t + (H-pad_t-pad_b)*(1 - (f(v)-flo)/(fhi-flo))
    out = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" role="img">']
    # y gridlines at powers of 10 (log) between lo..hi
    if logy:
        d = int(math.floor(flo))
        while d <= math.ceil(fhi):
            v = 10**d
            if lo*0.8 <= v <= hi*1.3:
                y = Y(v)
                lab = f"${v/1e6:.0f}M" if v >= 1e6 else (f"${v/1e3:.0f}k" if v >= 1e3 else f"${v:.0f}")
                out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" stroke="#eee"/>' +
                           f'<text x="{pad_l-6}" y="{y+3:.1f}" font-size="10" fill="#9ca3af" text-anchor="end">{lab}</text>')
            d += 1
    dates = series["_dates"] if "_dates" in series else None
    return out, X, Y, pad_l, pad_r, pad_t, pad_b

def line_chart(dates, series, W=700, H=300, note=""):
    """series: list of (name, values, color, width, dash)"""
    pad_l, pad_r, pad_t, pad_b = 50, 8, 6, 20
    allv = [v for _, vals, *_ in series for v in vals if v and v > 0]
    lo, hi = min(allv), max(allv)
    flo, fhi = math.log10(lo), math.log10(hi)
    n = len(dates)
    def X(i): return pad_l + (W-pad_l-pad_r)*i/(n-1)
    def Y(v): return pad_t + (H-pad_t-pad_b)*(1 - (math.log10(v)-flo)/(fhi-flo))
    s = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" style="width:100%;min-width:600px;height:auto" role="img">']
    d = int(math.floor(flo))
    while d <= math.ceil(fhi)+1:
        for m in (1, 3):
            v = m*10**d
            if lo*0.9 <= v <= hi*1.15:
                y = Y(v); lab = f"${v/1e6:g}M" if v >= 1e6 else f"${v/1e3:g}k"
                s.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" stroke="#eeeeee"/>'
                         f'<text x="{pad_l-5}" y="{y+3:.1f}" font-size="10" fill="#9ca3af" text-anchor="end">{lab}</text>')
        d += 1
    last = ""
    for i, dt in enumerate(dates):
        yr = dt[:4]
        if yr != last and int(yr) % 4 == 2:
            x = X(i)
            s.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{H-pad_b}" stroke="#f3f4f6"/>'
                     f'<text x="{x:.1f}" y="{H-6}" font-size="10" fill="#9ca3af" text-anchor="middle">{yr}</text>')
        last = yr
    for name, vals, color, wd, dash in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals) if v and v > 0)
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{wd}"{dd}/>')
    s.append("</svg>")
    return "".join(s)

def scoreboard_svg(rows, W=620):
    """rows: (label, value, cat, note). log-x bars around 1.0."""
    rh = 26; H = len(rows)*rh + 34
    lo, hi = 0.09, 11.0
    def X(v): return 170 + (W-170-10)*(math.log10(max(v, lo))-math.log10(lo))/(math.log10(hi)-math.log10(lo))
    s = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" style="width:100%;min-width:560px;height:auto" role="img">']
    for v, lab in [(0.1, "0.1×"), (0.25, "0.25×"), (0.5, "0.5×"), (1, "1× = QQQ-DCA"), (2, "2×"), (5, "5×"), (10, "10×")]:
        x = X(v); em = (v == 1)
        s.append(f'<line x1="{x:.1f}" y1="14" x2="{x:.1f}" y2="{H-20}" stroke="{"#111418" if em else "#eeeeee"}" stroke-width="{1.5 if em else 1}"/>'
                 f'<text x="{x:.1f}" y="{H-6}" font-size="10" fill="{"#111418" if em else "#9ca3af"}" text-anchor="middle" font-weight="{700 if em else 400}">{lab}</text>')
    y = 16
    for lab, v, cat, note in rows:
        color = {"sel": "#b91c1c", "tim": "#b91c1c", "lev": "#6b7280", "dead": "#e3a1a1"}[cat]
        x0, x1 = X(1.0), X(v)
        s.append(f'<text x="164" y="{y+11}" font-size="10" fill="#111418" text-anchor="end">{lab}</text>')
        xa, xb = min(x0, x1), max(x0, x1)
        s.append(f'<rect x="{xa:.1f}" y="{y+2}" width="{max(xb-xa,1.5):.1f}" height="12" fill="{color}" rx="2"/>')
        if note:
            s.append(f'<text x="{xb+5:.1f}" y="{y+11}" font-size="9" fill="#6b7280">{note}</text>')
        y += rh
    s.append("</svg>")
    return "".join(s)

def traj_svg(cutoffs, series, W=700, H=240):
    pad_l, pad_r, pad_t, pad_b = 40, 110, 10, 22
    lo, hi = 0.5, 7.0
    def X(i): return pad_l + (W-pad_l-pad_r)*i/(len(cutoffs)-1)
    def Y(v): return pad_t + (H-pad_t-pad_b)*(1 - (math.log10(v)-math.log10(lo))/(math.log10(hi)-math.log10(lo)))
    s = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" style="width:100%;min-width:600px;height:auto" role="img">']
    for v, lab in [(0.5, "0.5×"), (1, "1×"), (2, "2×"), (4, "4×"), (7, "7×")]:
        y = Y(v); em = v == 1
        s.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" stroke="{"#111418" if em else "#eeeeee"}" stroke-width="{1.4 if em else 1}"/>'
                 f'<text x="{pad_l-5}" y="{y+3:.1f}" font-size="10" fill="#6b7280" text-anchor="end">{lab}</text>')
    for i, c in enumerate(cutoffs):
        s.append(f'<text x="{X(i):.1f}" y="{H-6}" font-size="9.5" fill="#9ca3af" text-anchor="middle">{c}</text>')
    for name, vals, color in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        s.append(f'<text x="{W-pad_r+6}" y="{Y(vals[-1])+3:.1f}" font-size="10" fill="{color}" font-weight="700">{name}</text>')
    s.append("</svg>")
    return "".join(s)

# ---------------------------------------------------------------- assemble data
dates = [d.strftime("%Y-%m") for d in idx]
hero = line_chart(dates, [
    ("Contributions", list(CURVES["Contributions"].values), "#9ca3af", 1.3, "4 3"),
    ("SPY-DCA", list(CURVES["SPY-DCA"].values), "#6b7280", 1.6, None),
    ("QQQ-DCA", list(CURVES["QQQ-DCA"].values), "#111418", 2.6, None),
    ("VOLT (leverage dial)", list(CURVES["VOLT (leverage dial)"].values), "#1d4ed8", 1.8, None),
    ("TQQQ raw 3×", list(CURVES["TQQQ-DCA (raw 3×)"].values), "#b91c1c", 1.4, "2 3"),
])
fin = {k: CURVES[k].iloc[-1] for k in CURVES}

SB_SEL = [
    ("Best ML stock-picker (honest)", 0.71, "sel", "0.47–0.71× across 8 configs"),
    ("Factor composite screen", 0.60, "sel", "quality+value+momentum"),
    ("Broad 12-mo momentum picker", 0.26, "sel", "−82% drawdown"),
    ("Best stock screen (“qualifier”)", 1.21, "dead", "= random-null max — died"),
    ("Mega-cap momentum k2", 6.63, "dead", "recency — died in audit"),
    ("NDX momentum k8", 2.05, "dead", "½ of lead in final 6 mo — died"),
    ("SUMMIT-DCA (rebuilt clean)", 1.08, "dead", "published 2.2× — retired"),
    ("Random stock pickers (5 seeds)", 0.58, "sel", "0.33–0.84× band"),
]
SB_TIM = [
    ("Dual-momentum ETF rotation", 0.20, "tim", ""),
    ("All-leveraged-ETF rotation (best K)", 0.90, "tim", ""),
    ("Buy-the-dip / oversold rotation (best)", 0.45, "tim", "wins busts, loses bulls"),
    ("200-day trend switch on lev-tech", 0.74, "dead", "0.74–3.31× by trade day — died"),
    ("Regime-gated switch (best)", 1.08, "dead", "0.44× on other trade days — died"),
    ("Leveraged risk parity", 0.49, "tim", ""),
]
SB_LEV = [
    ("VOLT: vol-managed 3× NASDAQ", 2.78, "lev", "Sharpe 0.95 vs QQQ 0.90"),
    ("Raw TQQQ (3×) buy-and-DCA", 8.92, "lev", "−84% DD; dot-com ≈ −100%"),
]
sb1 = scoreboard_svg(SB_SEL); sb2 = scoreboard_svg(SB_TIM); sb3 = scoreboard_svg(SB_LEV)

traj = traj_svg(["2017", "2019", "2021", "2023", "2025", "2026-06"], [
    ("Mega-cap k2", [0.94, 0.88, 2.06, 1.41, 3.88, 6.63], "#b91c1c"),
    ("NDX-mom k8", [0.90, 0.80, 1.26, 0.97, 1.31, 2.05], "#6b7280"),
    ("Qualifier", [0.98, 1.12, 0.89, 0.96, 1.26, 1.21], "#9ca3af"),
])

GH = "https://github.com/viki-m13/bonds/blob/main"

html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>The QQQ-DCA Verdict</title>
<style>
:root{{--txt:#111418;--mut:#6b7280;--line:#e5e7eb;--card:#fafafa;--good:#15803d;--bad:#b91c1c;--blue:#1d4ed8}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:var(--txt);background:#fff;line-height:1.55;font-size:15px}}
.wrap{{max-width:760px;margin:0 auto;padding:20px 16px 60px}}
header{{padding:28px 0 14px;border-bottom:2px solid var(--txt)}}
h1{{font-size:26px;letter-spacing:-.5px}}
.sub{{color:var(--mut);font-size:14px;margin-top:4px}}
h2{{font-size:15px;text-transform:uppercase;letter-spacing:1.2px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}}
p{{margin:10px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}}
.verdict{{border-left:4px solid var(--txt);font-size:16px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}}
.k{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;text-align:center}}
.k .v{{font-size:22px;font-weight:800}}
.k .l{{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;margin:10px 0}}
th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--mut);padding:6px 8px;border-bottom:1px solid var(--line)}}
td{{padding:6px 8px;border-bottom:1px solid #f0f1f3}}
td.r,th.r{{text-align:right;font-variant-numeric:tabular-nums}}
.good{{color:var(--good);font-weight:700}}.bad{{color:var(--bad);font-weight:700}}
.note{{font-size:12.5px;color:var(--mut)}}
.leg{{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--mut);margin:6px 0}}
.leg i{{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px;border-radius:2px}}
.retract{{border-left:4px solid var(--bad)}}
.retract b{{color:var(--bad)}}
ul{{margin:8px 0 8px 20px}} li{{margin:5px 0}}
.check li{{list-style:none;margin:7px 0;padding-left:24px;position:relative}}
.check li:before{{content:"✓";position:absolute;left:0;color:var(--good);font-weight:800}}
a{{color:var(--txt)}}
footer{{margin-top:44px;padding-top:14px;border-top:1px solid var(--line);font-size:12px;color:var(--mut)}}
.chart{{margin:10px 0;overflow-x:auto}}
</style></head><body><div class="wrap">

<header>
<h1>The QQQ-DCA Verdict</h1>
<div class="sub">What 250+ tested strategies say about trying to beat a simple, automatic investment in QQQ — in plain English, with the receipts.</div>
</header>

<div class="card verdict" style="margin-top:18px">
<b>The verdict:</b> after years of simulated history, hundreds of strategy configurations, and multiple independent research efforts on clean, point-in-time data, <b>nothing in this research honestly and durably outperforms simply investing a fixed amount into QQQ every two weeks or every month</b> — except strategies that take <i>more risk than QQQ itself</i>, which is not out-smarting the benchmark, just out-daring it. Every apparent exception was audited, and every one either failed the audit or turned out to be leverage in disguise.
</div>

<div class="kpis">
<div class="k"><div class="v">250+</div><div class="l">configurations tested</div></div>
<div class="k"><div class="v">9</div><div class="l">independent campaigns</div></div>
<div class="k"><div class="v">3</div><div class="l">“winners” retracted on audit</div></div>
<div class="k"><div class="v">0</div><div class="l">honest skill-based beats</div></div>
</div>

<h2>1 · What exactly was tested</h2>
<p>The question, stated precisely: with the <b>same cash flows</b> (a fixed dollar amount invested on a fixed schedule — biweekly and monthly were both tested; the difference is a rounding error), can any rule for <b>picking stocks</b>, <b>rotating ETFs</b>, or <b>timing the market</b> end up with more money than just buying QQQ every time, without taking on more risk?</p>
<p>The tests used survivorship-clean, point-in-time data (delisted companies included; no knowledge of the future), realistic trading costs, and a benchmark that receives the identical contributions. Strategy families covered: machine-learning stock pickers (36 features, walk-forward), factor screens (value, quality, momentum, insider buying), momentum and mean-reversion systems, ETF rotation (sector, country, commodity, bond, leveraged), trend switches, regime gates, risk parity, dip-buying, IPO systems, and concentration plays — plus every "improvement" overlay the literature suggests.</p>

<h2>2 · The scoreboard</h2>
<p>Each bar is a strategy family's <b>final wealth as a multiple of QQQ-DCA's final wealth</b> (same money in, same period; best configuration of each family shown — the most charitable reading). Left of the 1× line = you ended up poorer than doing nothing clever.</p>
<p style="font-weight:700;font-size:13.5px;margin-bottom:2px">Pick better stocks (2015–2026 mandate harness)</p>
<div class="chart">{sb1}</div>
<p style="font-weight:700;font-size:13.5px;margin-bottom:2px">Time or rotate ETFs (2006–2026, continuous)</p>
<div class="chart">{sb2}</div>
<p style="font-weight:700;font-size:13.5px;margin-bottom:2px">Take more risk (leverage — the “exception” explained in §5)</p>
<div class="chart">{sb3}</div>
<p class="note">Faded red bars = results that <i>looked</i> like winners until audited (§4). Gray = leverage: more of QQQ's own risk, not skill. Sources: <a href="{GH}/dca/research/strategies/ascent/FINDINGS.md">ASCENT findings</a>, <a href="{GH}/leverage_etf_dca/README.md">leverage-ETF campaign</a>, <a href="{GH}/dca/README.md">SUMMIT-DCA record</a>.</p>

<h2>3 · “Maybe you just tested the wrong things?” — why this isn't selection bias</h2>
<p>A fair objection to any "nothing works" claim is that the tester quietly ignored the things that <i>do</i> work. Three facts argue the opposite happened here:</p>
<p><b>First, the breadth.</b> This wasn't one person's pet ideas: nine separate research campaigns, run at different times by independent efforts, tested 250+ configurations spanning every major approach in the practitioner and academic literature — and several invented mechanisms that appear in no literature at all. The failures were <i>published in detail</i>, not discarded: every findings file in this repository records what was tried and what it scored.</p>
<p><b>Second — and this is the important one — the winners were audited hardest, and the audits kept finding the same thing.</b> Three separate strategies were published as QQQ-beaters, then independently rebuilt, found defective, and publicly retracted:</p>
<div class="card retract"><b>WAVE (retracted).</b> Published at 21.5%/yr, Sharpe 1.41 — a machine-learning stock picker. An independent rebuild found two backtest defects: it could only "pick" stocks that were guaranteed to still exist 12 months later, and its training data leaked test-year information. Fixed honestly, it returns 12%/yr — <b>below QQQ</b>. <span class="note"><a href="{GH}/dca/research/strategies/ascent/FINDINGS.md">Full decomposition →</a></span></div>
<div class="card retract"><b>SUMMIT-DCA (retracted).</b> Published as beating QQQ-DCA in 93% of rolling windows with a 20× money multiple. A survivorship-clean rebuild found roughly <b>half the edge was survivorship bias and recency</b>; the honest version is ≈1.08× QQQ with a coin-flip win rate. It was retired from live tracking. <span class="note"><a href="{GH}/dca/README.md">Validation record →</a></span></div>
<div class="card retract"><b>PHOENIX (rebuilt).</b> A leveraged-ETF strategy whose review found leakage and stale-data defects; it required two full rebuilds (v3, v4) before its numbers were trustworthy — and the honest version's edge comes from leverage, not selection. <span class="note"><a href="{GH}/docs/phoenix.html">Factsheet →</a></span></div>
<p><b>Third, published outside claims failed the same way.</b> A well-known published IPO strategy claiming 20.5%/yr was reproduced on survivorship-clean data (6,599 IPOs including every failure): the honest result is 8.1%/yr. The pattern — impressive published number, honest rebuild deflates it — is exactly what you'd expect if the <i>industry's</i> apparent QQQ-beaters are selection effects, and exactly what a cherry-picking researcher would never volunteer.</p>

<h2>4 · “But some bars are above 1× — aren't those winners?”</h2>
<p>Four apparent winners survived to the audit stage. Each was put through five tests any honest backtest must pass: comparison against <b>random pickers</b> given the same rules, the <b>trajectory</b> of when the lead was built, <b>other eras</b>, <b>other trade days</b>, and <b>survivorship</b>. All four died:</p>
<div class="chart">{traj}</div>
<p class="note" style="margin-top:-4px">Lead vs QQQ-DCA if the same backtest had been stopped at each earlier date (strategy started 2015). Sources: trajectory tables in <a href="{GH}/dca/research/strategies/ascent/FINDINGS.md">ASCENT §5</a>.</p>
<ul>
<li><b>Mega-cap momentum (6.6×)</b> spent 2015–2019 <i>losing</i> to QQQ (0.88–0.94×). Its entire lead arrived in the 2021 and 2025–26 melt-ups — and dropping the momentum signal entirely (just holding the most-traded mega-caps) scored <i>higher</i> (7.7×). The "signal" subtracted value; the result was one era's concentration bet. In 2000–2014 the same strategy lost with −72% drawdowns.</li>
<li><b>NDX momentum (2.05×)</b>: more than half its lead was built in its final six months (one AI-hardware melt-up). Through 2023 it had beaten QQQ by nothing.</li>
<li><b>The best stock screen (1.21×)</b> landed exactly at the <i>maximum</i> of the random-picker distribution (random screens given the same universe scored 0.33–0.84×, best 1.21×) — indistinguishable from luck.</li>
<li><b>Trend-switching leveraged tech</b> produced anywhere from <b>0.74× to 3.31×</b> depending on <i>which day of the month</i> you happened to rebalance — a coin toss wearing a strategy's clothes.</li>
</ul>
<p>And on fixed three-year windows (the honest way to measure "would this have felt like winning?"), the surviving candidates beat QQQ-DCA in only <b>40–66% of windows</b> — coin-flips — with worst windows of 0.66–0.80×.</p>

<h2>5 · The one honest exception — and why it proves the rule</h2>
<p>One strategy genuinely ends with more money than QQQ-DCA in every tested era and survives every audit: <b>VOLT</b> — but it doesn't <i>pick</i> anything. It holds a 3× leveraged NASDAQ fund, scaled down when volatility rises, with the remainder in gold and bonds. It is a <b>risk dial</b>: on average ~1.6× QQQ's exposure, managed so the crashes stay QQQ-sized instead of catastrophic.</p>
<div class="chart">{hero}</div>
<div class="leg">
<span><i style="background:#9ca3af"></i>Contributions (${fin['Contributions']/1e3:,.0f}k in)</span>
<span><i style="background:#6b7280"></i>SPY-DCA → ${fin['SPY-DCA']/1e6:.1f}M</span>
<span><i style="background:#111418"></i><b>QQQ-DCA → ${fin['QQQ-DCA']/1e6:.1f}M</b></span>
<span><i style="background:#1d4ed8"></i>VOLT → ${fin['VOLT (leverage dial)']/1e6:.1f}M</span>
<span><i style="background:#b91c1c"></i>raw TQQQ → ${fin['TQQQ-DCA (raw 3×)']/1e6:.1f}M</span>
</div>
<p class="note">$1,000/month, 2006–2026, log scale, 10 bps costs; leveraged series validated against the real fund (0.999 daily correlation).</p>
<p>Look at the red dashed line: <b>raw 3× leverage ends highest of all</b>. Is buy-and-hold TQQQ therefore "the best strategy"? No — it fell <b>−84%</b> along the way in this sample, and started three years earlier it loses <b>−99.9%</b> in the dot-com crash and never recovers. That's the whole lesson in one chart: <b>in a two-decade bull market, whoever takes the most risk "wins" — until the era that kills them.</b> VOLT's contribution is not skill at beating QQQ; it's engineering that lets you hold more of QQQ's own risk without the −94% tail (dot-com: −58% vs −99.9%). Its risk-adjusted return (Sharpe 0.95 vs QQQ's 0.90) is essentially the same as the thing it "beats."</p>

<h2>6 · Why nothing works — the actual mechanism</h2>
<p>This isn't mysterious, and it isn't "markets are perfectly efficient." Four concrete reasons, each verified in the data:</p>
<ul>
<li><b>QQQ is already a momentum strategy.</b> It's capitalization-weighted: when a company wins, the index automatically holds more of it, fee-free, tax-free, with no rebalancing cost. Any picker who equal-weights, diversifies, or "takes profits" holds structurally <i>less</i> of the winners than the benchmark does. You are trying to out-momentum an automatic momentum machine.</li>
<li><b>The pond problem.</b> Stock-picking skill in this research is real and measurable (the ML models genuinely predict which stocks beat <i>each other</i>). But the picks live in a pond — mid/small caps, "quality" names — that as a whole underperformed QQQ for the entire sample. Verified directly: a <b>perfect oracle</b> choosing the best 5 stocks from every alternative pond still loses to QQQ-DCA in 3 of 5 eras. No signal quality can buy back missing pond beta.</li>
<li><b>The error-structure trap.</b> A simulated signal with only modest skill (rank-IC ≈ 0.05) but <i>random</i> errors beats QQQ-DCA in every era. The real signals have comparable IC but their errors are <i>correlated</i> — their boldest picks are all the same crowded momentum/quality bet, so they fail together: the honest ML's top picks land in the future top-decile <b>less often than random picks</b> (8.3% vs 10.2%). The skill exists; it's the wrong <i>shape</i> of skill.</li>
<li><b>Costs and taxes compound against you.</b> Every challenger trades; QQQ-DCA doesn't. 20 bps per trade, bid-ask, and (in a taxable account) short-term capital gains are a permanent headwind the benchmark never pays.</li>
</ul>

<h2>7 · What this means for an actual investor</h2>
<ul class="check">
<li><b>Automate the contribution and don't touch it.</b> Biweekly vs monthly is immaterial (tested: the conclusions reproduce at both cadences). The discipline is the edge.</li>
<li><b>Don't pay for selection.</b> Any product or newsletter claiming to beat QQQ-DCA should be assumed to fail the five audits in §4 until proven otherwise — ask for the random-null, the trajectory, other eras, other trade days, and survivorship-clean data.</li>
<li><b>The only honest dials are risk level and diversification.</b> Want more expected return? The evidence supports a <i>managed</i> leverage dial (§5) — taken with full knowledge that it's more risk, not free money, and that a −45–58% drawdown is in its history. Want a smoother ride? Diversify and accept less.</li>
<li><b>If someone beats QQQ-DCA for three years, remember the trajectory chart.</b> Mega-cap momentum was at 0.88× after five years and 6.6× after eleven. Both numbers were noise.</li>
</ul>

<h2>8 · The boundary of the claim (read this too)</h2>
<p>Honesty requires stating what this page does <i>not</i> prove. All of it is conditional on 1999–2026 U.S. market data — a sample in which technology led for two decades. QQQ-DCA "won" partly because the era belonged to exactly what QQQ holds. That cuts both ways: <b>QQQ-DCA is itself a concentrated bet</b>, and it has its own scar — in the dot-com crash a QQQ investor drew down <b>−81%</b> and waited ~15 years to break even, while boring diversification beat it. Nothing here says QQQ is safe; it says <i>no tested rule reliably improves on it without adding risk</i>. If tech leadership ends, the next twenty years' verdict may name a different benchmark — and the same audits will apply to it.</p>

<h2>Methodology & provenance</h2>
<p class="note">All results from this repository's committed research: point-in-time, delisting-inclusive price data (Tiingo, ~24k tickers incl. 8.9k delisted); SEC XBRL fundamentals and Form-4 insider data with reporting lags; identical cash flows for strategy and benchmark; 5–20 bps/side costs; delisting haircuts; walk-forward model training; locked holdouts where applicable. Key records: <a href="{GH}/dca/research/strategies/ascent/FINDINGS.md">ASCENT (stock selection, nulls &amp; gauntlets)</a> · <a href="{GH}/leverage_etf_dca/README.md">VOLT / ETF-timing campaign</a> · <a href="{GH}/leverage_etf_dca/INNOVATION_FINDINGS.md">all-regime innovation arc</a> · <a href="{GH}/dca/README.md">SUMMIT-DCA validation</a> · <a href="{GH}/dca/research/strategies/crackingmarkets_repro/FINDINGS.md">published-strategy reproductions</a> · <a href="{GH}/dca/research/strategies/METHODOLOGY_validation.md">validation methodology</a>.</p>

<footer>Research, not investment advice. Backtests are simulations; past performance does not guarantee future results. Generated from committed data by <code>scripts/gen_verdict.py</code>.</footer>
</div></body></html>"""

out = f"{ROOT}/docs/verdict.html"
open(out, "w").write(html)
print(f"written {out} ({len(html):,} bytes)")
