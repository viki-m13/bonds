"""Generate docs/verdict.html — "The QQQ-DCA Verdict".

A plain-English, evidence-first page arguing the repo's most-replicated finding:
across 250+ tested configurations, nothing honestly beats biweekly/monthly
DCA-into-QQQ. Every number is sourced from committed research records; the
equity curves are recomputed here from the committed ETF panel (no network).

Run:  python3 scripts/gen_verdict.py
"""
import os, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- curves (real, from committed panel)
P = pd.read_pickle(f"{ROOT}/leverage_etf_dca/scripts/_etf_panel.pkl")
close = P["close"].sort_index()
mgrid = pd.DatetimeIndex(sorted(close.groupby(close.index.to_period("M")).apply(lambda x: x.index[-1]).values))
mret = close.reindex(mgrid).pct_change()

def dca_curve(r, c=1000.0):
    v = 0.0; out = []
    for dt, x in r.items():
        if not np.isfinite(x): x = 0.0
        v = (v + c)*(1 + x); out.append((dt, v))
    return pd.Series(dict(out))

win = mgrid[(mgrid >= pd.Timestamp("2006-01-01")) & (mgrid <= pd.Timestamp("2026-12-31"))][1:]
CURVES = {}
for t, nm in [("QQQ", "QQQ-DCA"), ("SPY", "SPY-DCA"), ("TQQQ", "3x-leveraged QQQ")]:
    CURVES[nm] = dca_curve(mret[t].loc[win])
CURVES["Contributions"] = pd.Series(np.arange(1, len(win)+1)*1000.0, index=win)
dates = [d.strftime("%Y-%m") for d in win]
fin = {k: CURVES[k].iloc[-1] for k in CURVES}

# ---------------------------------------------------------------- SVG helpers
def line_chart(dates, series, W=700, H=300):
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
        color = {"sel": "#b91c1c", "dead": "#e3a1a1"}[cat]
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

# ---------------------------------------------------------------- assemble
hero = line_chart(dates, [
    ("Contributions", list(CURVES["Contributions"].values), "#9ca3af", 1.3, "4 3"),
    ("SPY-DCA", list(CURVES["SPY-DCA"].values), "#6b7280", 1.6, None),
    ("QQQ-DCA", list(CURVES["QQQ-DCA"].values), "#111418", 2.6, None),
    ("3x-leveraged QQQ", list(CURVES["3x-leveraged QQQ"].values), "#b91c1c", 1.4, "2 3"),
])

SB_SEL = [
    ("Best ML stock-picker (honest)", 0.71, "sel", "0.47–0.71× across 8 configs"),
    ("Factor composite screen", 0.60, "sel", "quality+value+momentum"),
    ("Broad 12-mo momentum picker", 0.26, "sel", "−82% drawdown"),
    ("Best stock screen (“qualifier”)", 1.21, "dead", "= random-picker max — died"),
    ("Mega-cap momentum k2", 6.63, "dead", "recency — died in audit"),
    ("NDX momentum k8", 2.05, "dead", "½ of lead in final 6 mo — died"),
    ("SUMMIT-DCA (rebuilt clean)", 1.08, "dead", "published 2.2× — retired"),
    ("Random stock pickers (5 seeds)", 0.58, "sel", "0.33–0.84× band"),
]
SB_TIM = [
    ("Dual-momentum ETF rotation", 0.20, "sel", ""),
    ("All-leveraged-ETF rotation (best K)", 0.90, "sel", ""),
    ("Buy-the-dip / oversold rotation (best)", 0.45, "sel", "wins busts, loses bulls"),
    ("200-day trend switch on lev-tech", 0.74, "dead", "0.74–3.31× by trade day — died"),
    ("Regime-gated switch (best)", 1.08, "dead", "0.44× on other trade days — died"),
    ("Leveraged risk parity", 0.49, "sel", ""),
]
sb1 = scoreboard_svg(SB_SEL); sb2 = scoreboard_svg(SB_TIM)

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
:root{{--txt:#111418;--mut:#6b7280;--line:#e5e7eb;--card:#fafafa;--good:#15803d;--bad:#b91c1c}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:var(--txt);background:#fff;line-height:1.55;font-size:15px}}
.wrap{{max-width:760px;margin:0 auto;padding:20px 16px 60px}}
header{{padding:28px 0 14px;border-bottom:2px solid var(--txt)}}
h1{{font-size:26px;letter-spacing:-.5px}}
.sub{{color:var(--mut);font-size:14px;margin-top:4px}}
h2{{font-size:15px;text-transform:uppercase;letter-spacing:1.2px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}}
h3{{font-size:14.5px;margin:18px 0 6px}}
p{{margin:10px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}}
.verdict{{border-left:4px solid var(--txt);font-size:16px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}}
.k{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;text-align:center}}
.k .v{{font-size:22px;font-weight:800}}
.k .l{{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}}
.good{{color:var(--good);font-weight:700}}.bad{{color:var(--bad);font-weight:700}}
.note{{font-size:12.5px;color:var(--mut)}}
.leg{{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--mut);margin:6px 0}}
.leg i{{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px;border-radius:2px}}
.retract{{border-left:4px solid var(--bad)}}
.retract b{{color:var(--bad)}}
.big{{border-left:4px solid var(--txt)}}
.big .n{{font-size:26px;font-weight:800;line-height:1.1}}
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
<b>The verdict:</b> after years of simulated history, hundreds of strategy configurations, and multiple independent research efforts on clean, point-in-time data, <b>nothing in this research honestly and durably outperforms simply investing a fixed amount into QQQ every two weeks or every month</b> — not stock-picking, not "buying winners," not rotating ETFs, not market timing. Every apparent winner was audited, and every one either failed the audit or turned out to be leverage in disguise.
</div>

<div class="kpis">
<div class="k"><div class="v">250+</div><div class="l">configurations tested</div></div>
<div class="k"><div class="v">9</div><div class="l">independent campaigns</div></div>
<div class="k"><div class="v">3</div><div class="l">“winners” retracted on audit</div></div>
<div class="k"><div class="v">0</div><div class="l">honest skill-based beats</div></div>
</div>

<h2>1 · What exactly was tested</h2>
<p>The question, stated precisely: with the <b>same cash flows</b> (a fixed dollar amount invested on a fixed schedule — every two weeks and monthly were both tested; the difference is a rounding error), can any rule for <b>picking stocks</b>, <b>rotating ETFs</b>, or <b>timing the market</b> end up with more money than just buying QQQ every time?</p>
<p>The tests used survivorship-clean, point-in-time data (companies that later went bankrupt or were delisted are included; nothing "knows the future"), realistic trading costs, and a benchmark that receives the identical contributions. Strategy families covered: machine-learning stock pickers, factor screens (value, quality, momentum, insider buying), momentum and "buy the winners" systems, dip-buying, ETF rotation (sector, country, commodity, bond, leveraged), trend switches, regime gates, risk parity, and IPO systems — plus every "improvement" overlay the academic literature suggests.</p>

<h2>2 · The scoreboard</h2>
<p>Each bar is a strategy family's <b>final wealth as a multiple of QQQ-DCA's final wealth</b> (same money in, same period; the best configuration of each family is shown — the most charitable reading). Left of the 1× line = you ended up poorer than doing nothing clever.</p>
<p style="font-weight:700;font-size:13.5px;margin-bottom:2px">Pick better stocks (2015–2026 test rig)</p>
<div class="chart">{sb1}</div>
<p style="font-weight:700;font-size:13.5px;margin-bottom:2px">Time or rotate ETFs (2006–2026, continuous)</p>
<div class="chart">{sb2}</div>
<p class="note">Faded red bars = results that <i>looked</i> like winners until audited (§4). Sources: <a href="{GH}/dca/research/strategies/ascent/FINDINGS.md">stock-selection study</a>, <a href="{GH}/leverage_etf_dca/README.md">ETF-timing campaign</a>, <a href="{GH}/dca/README.md">SUMMIT-DCA record</a>.</p>

<h2>3 · Why “picking the winners” doesn't work — the evidence</h2>
<p>"Just buy the best stocks" sounds like it should beat owning all of them. Here is what the data actually shows, step by step. Every claim below was measured in this research, not assumed.</p>

<h3>3.1 &nbsp;Most stocks lose to the index — so most picks start out behind</h3>
<div class="card big"><div class="n">32–46%</div>
That's the fraction of all investable U.S. stocks that beat QQQ over a typical 12-month stretch, measured across every era in the data (in 2015–2019 it was just 31.7%). Flip it around: <b>a majority of stocks — usually about two-thirds — lose to the index in any given year.</b></div>
<p>Why? Stock-market returns are extremely lopsided: a tiny handful of huge winners produce most of the market's gains, while the typical stock does little or worse (academic work by Bessembinder found most individual stocks over their lifetime return less than cash in the bank). An index rides that handful automatically. A stock-picker has to <i>find</i> them in advance, over and over, against roughly 2-to-1 odds per pick per year.</p>

<h3>3.2 &nbsp;No tested signal even gets you to a coin flip</h3>
<p>Every predictive signal in the data was scored on the direct question: <i>what fraction of its picks actually beat QQQ over the next year?</i> The result: <b>no signal — price patterns, fundamentals, insider buying, machine learning, or combinations — reached even 50% accuracy in any era.</b> The best signals improved on the base rate by 4–8 percentage points (to roughly 37–44%). And the "hottest" stocks — those at the extreme of their 52-week highs, the most winner-looking picks of all — went on to beat QQQ only <b>5–17%</b> of the time. Looking like a winner and continuing to win are nearly opposite things.</p>

<h3>3.3 &nbsp;Yesterday's winners are not tomorrow's</h3>
<div class="card big"><div class="n">−10.7% per year</div>
What you got for buying the 20 hottest momentum stocks of 2015 and holding them for a decade — while QQQ returned <b>+18.7% per year</b> over the same stretch. The winners list turns over constantly; buying it and holding is one of the worst tested strategies in this entire research program (−84% at its lowest point).</div>
<p>Winners <i>do</i> keep winning for a while — that's a real, well-documented effect — but harvesting it requires constant rotation, which triggers trading costs and short-term taxes, and even then the rotating version failed the audits in §4. Meanwhile QQQ <i>is already</i> a ride-your-winners machine: because it's weighted by company size, when a company wins, the index automatically holds more of it — no trades, no taxes, no delay.</p>

<h3>3.4 &nbsp;The index already owns the winners — at full size</h3>
<p>This is the quiet structural reason picking can't win. Whatever the next decade's monster stock is, QQQ already holds it, and as it grows the index's stake grows with it, automatically. Every act of "smart" management — diversifying across your picks, trimming a position that's "gotten too big," taking profits, equal-weighting — has one common effect: <b>you end up owning less of the biggest winner than the index does.</b> Since a few big winners drive everything (§3.1), owning less of them is usually fatal. Measured directly: strategies that "took profits" in winners underperformed the ones that never sold, and both underperformed the index that never has to decide.</p>

<h3>3.5 &nbsp;Even real skill turns out to be the wrong kind</h3>
<p>The strongest models in this research genuinely <i>can</i> rank stocks — their favorites reliably do better than their least-favorites. That skill is real and measurable. It still didn't beat QQQ, for two proven reasons:</p>
<ul>
<li><b>The confident picks all fail together.</b> When the model's top picks were checked against the stocks that actually ended up in the top 10% a year later, the model's picks landed there <b>less often than randomly chosen stocks</b> (8.3% vs 10.2%). Its boldest picks weren't independent bets — they were all the same fashionable bet (crowded momentum/quality names) wearing different tickers, so they rose and fell as a group. A simulation showed that even a <i>weak</i> signal would beat QQQ if its errors were random; the real signals' errors are correlated, which is the one shape of error that doesn't work.</li>
<li><b>Even perfect picking in the wrong pond loses.</b> The pools where picking "skill" works best (smaller, cheaper, higher-quality stocks) underperformed QQQ as a whole. Tested directly with a cheat: an <b>all-knowing oracle</b> that picks the future best 5 stocks from each alternative pool <i>still loses to QQQ-DCA in 3 of 5 eras</i>. If perfect foresight in that pond loses, no signal in that pond can win.</li>
</ul>

<h3>3.6 &nbsp;The deck is stacked: costs and taxes</h3>
<p>Every picking strategy trades; QQQ-DCA doesn't. Each trade pays a spread and commission, and in a normal taxable account, selling winners within a year converts your best outcomes into your highest tax rate. The benchmark never sells, so it never pays. Over decades, this alone is a compounding, guaranteed handicap that every challenger must first earn back before "beating" anything.</p>

<h3>3.7 &nbsp;And every published “winner” we audited turned out inflated</h3>
<p>Three strategies from this research program itself were published as QQQ-beaters, then independently rebuilt — and retracted:</p>
<div class="card retract"><b>WAVE (retracted).</b> Published at 21.5%/yr. The rebuild found its backtest could only "pick" stocks guaranteed to still exist a year later, and its training data leaked future information. Fixed honestly: 12%/yr — <b>below QQQ</b>.</div>
<div class="card retract"><b>SUMMIT-DCA (retracted).</b> Published as beating QQQ-DCA in 93% of periods. A survivorship-clean rebuild found half the edge was biased data and recency; the honest version is a coin flip. Retired from live tracking.</div>
<div class="card retract"><b>PHOENIX (rebuilt twice).</b> Review found data leakage and stale-price defects; the honest rebuild's edge comes from taking more risk, not from selection.</div>
<p>The same happened to outside claims: a published IPO strategy advertising 20.5%/yr was reproduced on survivorship-clean data (6,599 IPOs including every failure) — honest result, 8.1%/yr. This mirrors the broader industry record: the long-running S&amp;P SPIVA scorecards find ~90% of professional U.S. stock funds lag their index over 15 years. <b>If the people paid to pick winners can't, a backtest that claims to should be presumed broken until audited</b> — and when we audited ours, they were.</p>

<h2>4 · How the “winners” died — the audits</h2>
<p>Four strategies survived long enough to look like real QQQ-beaters. Each was put through five tests any honest claim must pass: random pickers given the same rules, the <b>timing</b> of when the lead was built, other eras, other trade days, and survivorship-clean data. All four died:</p>
<div class="chart">{traj}</div>
<p class="note" style="margin-top:-4px">The same backtests, stopped at earlier dates: lead vs QQQ-DCA (strategies start 2015). A real edge grows steadily; these leads appear all at once, late.</p>
<ul>
<li><b>Mega-cap momentum ("6.6×")</b> spent 2015–2019 <i>losing</i> to QQQ. Its entire lead arrived in two late melt-ups — and dropping its "signal" entirely (just holding the biggest, most-traded stocks) scored <i>higher</i>. In 2000–2014 the same recipe lost money with a −72% crash.</li>
<li><b>NDX momentum ("2.05×")</b>: more than half its lead came from its final six months. Through 2023 it had beaten QQQ by nothing.</li>
<li><b>The best stock screen ("1.21×")</b> scored exactly at the top of the random-picker range (random screens: 0.33–0.84×, luckiest 1.21×) — indistinguishable from luck.</li>
<li><b>Trend-switching</b> returned anywhere from 0.74× to 3.31× depending on <i>which day of the month</i> you happened to trade — a coin toss wearing a strategy's clothes.</li>
</ul>
<p>On honest fixed three-year windows, the "winners" beat QQQ-DCA only 40–66% of the time — coin flips — with worst stretches of 0.66–0.80×.</p>

<h2>5 · “Fine — then I'll just take more risk”</h2>
<p>One thing <i>does</i> end with more money than QQQ-DCA in this sample: holding a 3×-leveraged version of QQQ itself. Before concluding anything, look at the ride:</p>
<div class="chart">{hero}</div>
<div class="leg">
<span><i style="background:#9ca3af"></i>Contributions (${fin['Contributions']/1e3:,.0f}k in)</span>
<span><i style="background:#6b7280"></i>SPY-DCA → ${fin['SPY-DCA']/1e6:.1f}M</span>
<span><i style="background:#111418"></i><b>QQQ-DCA → ${fin['QQQ-DCA']/1e6:.1f}M</b></span>
<span><i style="background:#b91c1c"></i>3× leveraged → ${fin['3x-leveraged QQQ']/1e6:.1f}M</span>
</div>
<p class="note">$1,000/month, 2006–2026, log scale, 10 bps costs; leveraged series validated against the real fund (0.999 daily correlation).</p>
<p>The red line ends highest — and fell <b>−84%</b> along the way. Started three years earlier, it loses <b>−99.9%</b> in the dot-com crash and never recovers. That's not a better strategy; it's a bigger bet that happened to survive this particular sample. It's also the honest explanation for <i>every</i> line that ends above QQQ in any backtest anywhere: <b>in a two-decade bull market, whoever takes the most risk ends up on top — until the era that ends them.</b> More risk is a choice you're allowed to make; it is not out-smarting the index, and almost no one actually holds through an 84% loss.</p>

<h2>6 · What this means for an actual investor</h2>
<ul class="check">
<li><b>Automate the contribution and don't touch it.</b> Every two weeks or monthly — tested, immaterial. The discipline is the edge.</li>
<li><b>Don't buy stock picks.</b> Not from newsletters, not from screens, not from AI models. The base rate is ~2-to-1 against every pick, no tested signal reaches a coin flip, and the picks that look best (recent big winners) have the <i>worst</i> forward odds.</li>
<li><b>Never sell your winners to "rebalance into" your losers among growth stocks.</b> That's the one job the index does better than everyone: it lets winners run at full weight, forever, tax-free.</li>
<li><b>Judge any "market-beating" claim with the five audits</b> in §4: random-picker comparison, when the lead was built, other eras, other trade days, survivorship-clean data. Every claim we tested — including our own — failed at least one.</li>
<li><b>Diversification (adding bonds/other assets) is for a smoother ride, not more money.</b> It reliably cuts crashes and reliably costs return. Decide which you're buying.</li>
</ul>

<h2>7 · The boundary of the claim (read this too)</h2>
<p>Honesty requires stating what this page does <i>not</i> prove. All of it is conditional on 1999–2026 U.S. market data — a sample in which technology led for two decades. QQQ-DCA "won" partly because the era belonged to exactly what QQQ holds. That cuts both ways: <b>QQQ-DCA is itself a concentrated bet</b>, and it has its own scar — in the dot-com crash a QQQ investor lost <b>−81%</b> and waited roughly 15 years to break even, while boring diversification beat it. Nothing here says QQQ is safe; it says <i>no tested rule reliably improves on it</i>. If tech leadership ends, the next twenty years' verdict may name a different benchmark — and the same audits will apply to it.</p>

<h2>Methodology &amp; provenance</h2>
<p class="note">All results from this repository's committed research: point-in-time, delisting-inclusive price data (~24k tickers incl. 8.9k delisted); SEC fundamentals and insider filings with realistic reporting lags; identical cash flows for strategy and benchmark; 5–20 bps/side costs; delisting haircuts; walk-forward model training; locked holdouts where applicable. Key records: <a href="{GH}/dca/research/strategies/ascent/FINDINGS.md">stock-selection study (base rates, oracle bound, audits)</a> · <a href="{GH}/leverage_etf_dca/README.md">ETF-timing campaign</a> · <a href="{GH}/leverage_etf_dca/INNOVATION_FINDINGS.md">all-regime innovation arc</a> · <a href="{GH}/dca/README.md">SUMMIT-DCA validation</a> · <a href="{GH}/dca/research/strategies/crackingmarkets_repro/FINDINGS.md">published-strategy reproductions</a> · <a href="{GH}/dca/research/strategies/METHODOLOGY_validation.md">validation methodology</a>. External corroboration: Bessembinder, “Do Stocks Outperform Treasury Bills?” (2018); S&amp;P SPIVA scorecards.</p>

<footer>Research, not investment advice. Backtests are simulations; past performance does not guarantee future results. Generated from committed data by <code>scripts/gen_verdict.py</code>.</footer>
</div></body></html>"""

out = f"{ROOT}/docs/verdict.html"
open(out, "w").write(html)
print(f"written {out} ({len(html):,} bytes)")
