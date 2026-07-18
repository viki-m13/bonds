"""Generate docs/verdict.html — "Can Anyone Beat Just Buying QQQ?"

A general-audience, professional-grade research paper. Every statistic is
computed from point-in-time, delisting-inclusive data committed in this repo
(scripts/verdict_evidence.py + verdict_qqqspy.py -> /tmp/verdict_evidence.json)
or quoted from the committed research records. No internal strategy names.

Run:
  python3 scripts/verdict_evidence.py   (needs ascent panels; see build_panels.py)
  python3 scripts/verdict_qqqspy.py
  python3 scripts/gen_verdict.py
"""
import os, math, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = json.load(open("/tmp/verdict_evidence.json"))

# ---------- 3x-leverage curves from the committed ETF panel ----------
P = pd.read_pickle(f"{ROOT}/leverage_etf_dca/scripts/_etf_panel.pkl")
close = P["close"].sort_index()
mgrid = pd.DatetimeIndex(sorted(close.groupby(close.index.to_period("M")).apply(lambda x: x.index[-1]).values))
mret = close.reindex(mgrid).pct_change()
def dca_curve(r, c=1000.0):
    v = 0.0; out = []
    for x in r:
        if not np.isfinite(x): x = 0.0
        v = (v + c)*(1 + x); out.append(v)
    return np.array(out)
winL = mgrid[(mgrid >= pd.Timestamp("2006-01-01")) & (mgrid <= pd.Timestamp("2026-12-31"))][1:]
lev = {nm: dca_curve(mret[t].loc[winL].values) for t, nm in [("QQQ","QQQ"),("SPY","SPY"),("TQQQ","LEV")]}
lev["CONTRIB"] = np.arange(1, len(winL)+1)*1000.0
lev_dates = [d.strftime("%Y-%m") for d in winL]

# =======================  SVG helpers  =======================
MINW = 'style="width:100%;min-width:560px;height:auto"'
def _fmt_money(v):
    return f"${v/1e6:g}M" if v >= 1e6 else (f"${v/1e3:g}k" if v >= 1e3 else f"${v:g}")

def lines_svg(dates, series, W=700, H=290, logy=True, yfmt=_fmt_money, ylines=None, yearmod=4, shade=None):
    pad_l, pad_r, pad_t, pad_b = 52, 8, 8, 20
    allv = [v for _, vals, *_ in series for v in vals if v is not None and (not logy or v > 0)]
    lo, hi = min(allv), max(allv)
    f = (lambda v: math.log10(v)) if logy else (lambda v: v)
    flo, fhi = f(lo), f(hi)
    n = len(dates)
    def X(i): return pad_l + (W-pad_l-pad_r)*i/(n-1)
    def Y(v): return pad_t + (H-pad_t-pad_b)*(1 - (f(v)-flo)/(fhi-flo or 1))
    s = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" {MINW} role="img">']
    if shade:
        i0, i1, lab = shade
        s.append(f'<rect x="{X(i0):.1f}" y="{pad_t}" width="{X(i1)-X(i0):.1f}" height="{H-pad_t-pad_b}" fill="#fef2f2"/>')
    yl = ylines if ylines is not None else []
    if not yl and logy:
        d = int(math.floor(flo))
        while d <= math.ceil(fhi)+1:
            for m in (1, 3):
                v = m*10**d
                if lo*0.9 <= v <= hi*1.15: yl.append(v)
            d += 1
    for v in yl:
        y = Y(v)
        s.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" stroke="#eeeeee"/>'
                 f'<text x="{pad_l-5}" y="{y+3:.1f}" font-size="10" fill="#9ca3af" text-anchor="end">{yfmt(v)}</text>')
    last = ""
    for i, dt in enumerate(dates):
        yr = dt[:4]
        if yr != last and int(yr) % yearmod == 2:
            x = X(i)
            s.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{H-pad_b}" stroke="#f3f4f6"/>'
                     f'<text x="{x:.1f}" y="{H-5}" font-size="10" fill="#9ca3af" text-anchor="middle">{yr}</text>')
        last = yr
    for name, vals, color, wd, dash in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals) if v is not None and (not logy or v > 0))
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{wd}"{dd}/>')
    s.append("</svg>")
    return "".join(s)

def band_svg(dates, p10, p50, p90, W=700, H=270):
    pad_l, pad_r, pad_t, pad_b = 46, 8, 8, 20
    lo, hi = min(p10)*0.9, max(max(p90), 1.35)
    n = len(dates)
    def X(i): return pad_l + (W-pad_l-pad_r)*i/(n-1)
    def Y(v): return pad_t + (H-pad_t-pad_b)*(1 - (v-lo)/(hi-lo))
    s = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" {MINW} role="img">']
    for v, lab, em in [(0.25, "0.25×", 0), (0.5, "0.5×", 0), (0.75, "0.75×", 0), (1.0, "1× = QQQ-DCA", 1), (1.25, "1.25×", 0)]:
        if lo <= v <= hi:
            y = Y(v)
            s.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" stroke="{"#111418" if em else "#eeeeee"}" stroke-width="{1.5 if em else 1}"/>'
                     f'<text x="{pad_l-5}" y="{y+3:.1f}" font-size="10" fill="{"#111418" if em else "#9ca3af"}" text-anchor="end" font-weight="{700 if em else 400}">{lab}</text>')
    last = ""
    for i, dt in enumerate(dates):
        yr = dt[:4]
        if yr != last and int(yr) % 2 == 1:
            s.append(f'<text x="{X(i):.1f}" y="{H-5}" font-size="10" fill="#9ca3af" text-anchor="middle">{yr}</text>')
        last = yr
    poly_up = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(p90))
    poly_dn = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(p10[::-1]))
    s.append(f'<polygon points="{poly_up} {" ".join(f"{X(len(p10)-1-i):.1f},{Y(v):.1f}" for i, v in enumerate(p10[::-1]))}" fill="#fee2e2" opacity="0.8"/>')
    s.append(f'<polyline points="{" ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(p50))}" fill="none" stroke="#b91c1c" stroke-width="2.4"/>')
    s.append("</svg>")
    return "".join(s)

def hist_svg(labels, counts, marker_idx, marker_lab, median_idx, W=700, H=300):
    pad_l, pad_r, pad_t, pad_b = 40, 8, 26, 58
    n = len(labels); mx = max(counts)
    bw = (W-pad_l-pad_r)/n
    def X(i): return pad_l + bw*i
    def Y(v): return pad_t + (H-pad_t-pad_b)*(1 - v/mx)
    s = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" {MINW} role="img">']
    for i, (lab, c) in enumerate(zip(labels, counts)):
        col = "#b91c1c" if i < 5 else ("#9ca3af" if i == 5 else "#15803d")
        s.append(f'<rect x="{X(i)+2:.1f}" y="{Y(c):.1f}" width="{bw-4:.1f}" height="{(H-pad_b)-Y(c):.1f}" fill="{col}" rx="2"/>')
        s.append(f'<text x="{X(i)+bw/2:.1f}" y="{Y(c)-4:.1f}" font-size="9.5" fill="#374151" text-anchor="middle">{c}</text>')
        s.append(f'<text transform="translate({X(i)+bw/2:.1f},{H-52}) rotate(38)" font-size="8.6" fill="#6b7280" text-anchor="start">{lab}</text>')
    x = X(marker_idx) + bw*0.5
    s.append(f'<line x1="{x:.1f}" y1="{pad_t-2}" x2="{x:.1f}" y2="{H-pad_b}" stroke="#111418" stroke-width="2" stroke-dasharray="5 3"/>')
    s.append(f'<text x="{x:.1f}" y="{pad_t-8}" font-size="10.5" font-weight="700" fill="#111418" text-anchor="middle">{marker_lab}</text>')
    xm = X(median_idx) + bw*0.5
    s.append(f'<text x="{xm:.1f}" y="{pad_t-8}" font-size="10" fill="#6b7280" text-anchor="middle">median stock</text>')
    s.append(f'<line x1="{xm:.1f}" y1="{pad_t-2}" x2="{xm:.1f}" y2="{H-pad_b}" stroke="#6b7280" stroke-width="1.6" stroke-dasharray="2 3"/>')
    s.append("</svg>")
    return "".join(s)

def bars_svg(xlabels, vals, W=700, H=240, color="#b91c1c", ref=None, reflab="", fmt=lambda v: f"{v:g}"):
    pad_l, pad_r, pad_t, pad_b = 40, 8, 12, 34
    n = len(xlabels); mx = max(vals+([ref] if ref else []))
    bw = (W-pad_l-pad_r)/n
    def X(i): return pad_l + bw*i
    def Y(v): return pad_t + (H-pad_t-pad_b)*(1 - v/mx)
    s = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" {MINW} role="img">']
    if ref is not None:
        y = Y(ref)
        s.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" stroke="#111418" stroke-width="1.4" stroke-dasharray="5 3"/>'
                 f'<text x="{W-pad_r}" y="{y-4:.1f}" font-size="10" fill="#111418" text-anchor="end" font-weight="700">{reflab}</text>')
    for i, (lab, v) in enumerate(zip(xlabels, vals)):
        s.append(f'<rect x="{X(i)+1.5:.1f}" y="{Y(v):.1f}" width="{bw-3:.1f}" height="{(H-pad_b)-Y(v):.1f}" fill="{color}" rx="1.5"/>')
        if n <= 16:
            s.append(f'<text x="{X(i)+bw/2:.1f}" y="{Y(v)-4:.1f}" font-size="9" fill="#374151" text-anchor="middle">{fmt(v)}</text>')
        if n <= 30 and (n <= 16 or i % 2 == 0):
            s.append(f'<text x="{X(i)+bw/2:.1f}" y="{H-6}" font-size="9" fill="#6b7280" text-anchor="middle">{lab}</text>')
    s.append("</svg>")
    return "".join(s)

def scoreboard_svg(rows, W=620):
    rh = 26; H = len(rows)*rh + 34
    lo, hi = 0.09, 11.0
    def X(v): return 170 + (W-170-10)*(math.log10(max(v, lo))-math.log10(lo))/(math.log10(hi)-math.log10(lo))
    s = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" {MINW} role="img">']
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
        if note: s.append(f'<text x="{xb+5:.1f}" y="{y+11}" font-size="9" fill="#6b7280">{note}</text>')
        y += rh
    s.append("</svg>")
    return "".join(s)

def traj_svg(cutoffs, series, W=700, H=240):
    pad_l, pad_r, pad_t, pad_b = 40, 120, 10, 22
    lo, hi = 0.5, 7.0
    def X(i): return pad_l + (W-pad_l-pad_r)*i/(len(cutoffs)-1)
    def Y(v): return pad_t + (H-pad_t-pad_b)*(1 - (math.log10(v)-math.log10(lo))/(math.log10(hi)-math.log10(lo)))
    s = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" {MINW} role="img">']
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

def area_dd_svg(dates, dd, W=700, H=230):
    pad_l, pad_r, pad_t, pad_b = 44, 8, 8, 20
    lo = min(dd)
    n = len(dates)
    def X(i): return pad_l + (W-pad_l-pad_r)*i/(n-1)
    def Y(v): return pad_t + (H-pad_t-pad_b)*((0-v)/(0-lo))
    s = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" {MINW} role="img">']
    for v in [0, -25, -50, -75]:
        if v >= lo-5:
            y = Y(v)
            s.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" stroke="#eeeeee"/>'
                     f'<text x="{pad_l-5}" y="{y+3:.1f}" font-size="10" fill="#9ca3af" text-anchor="end">{v}%</text>')
    last = ""
    for i, dt in enumerate(dates):
        yr = dt[:4]
        if yr != last and int(yr) % 3 == 0:
            s.append(f'<text x="{X(i):.1f}" y="{H-5}" font-size="10" fill="#9ca3af" text-anchor="middle">{yr}</text>')
        last = yr
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(dd))
    s.append(f'<polygon points="{X(0):.1f},{Y(0):.1f} {pts} {X(n-1):.1f},{Y(0):.1f}" fill="#fee2e2"/>')
    s.append(f'<polyline points="{pts}" fill="none" stroke="#b91c1c" stroke-width="1.6"/>')
    s.append("</svg>")
    return "".join(s)

def dd_bars_svg(items, W=700, H=230):
    pad_l, pad_r, pad_t, pad_b = 46, 8, 30, 26
    n = len(items); bw = (W-pad_l-pad_r)/n
    def X(i): return pad_l + bw*i
    def Y(v): return pad_t + (H-pad_t-pad_b)*(v/-100.0)
    s = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" {MINW} role="img">']
    for v in [0, -50, -100]:
        y = pad_t + (H-pad_t-pad_b)*(v/-100.0)
        s.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" stroke="#eeeeee"/>'
                 f'<text x="{pad_l-5}" y="{y+3:.1f}" font-size="10" fill="#9ca3af" text-anchor="end">{v}%</text>')
    for i, it in enumerate(items):
        s.append(f'<rect x="{X(i)+6:.1f}" y="{pad_t:.1f}" width="{bw-12:.1f}" height="{(H-pad_t-pad_b)*(it["dd"]/-100.0):.1f}" fill="#b91c1c" rx="2"/>')
        s.append(f'<text x="{X(i)+bw/2:.1f}" y="{Y(it["dd"])+14:.1f}" font-size="10" font-weight="700" fill="#7f1d1d" text-anchor="middle">{it["dd"]}%</text>')
        s.append(f'<text x="{X(i)+bw/2:.1f}" y="{pad_t-14}" font-size="11" font-weight="700" fill="#111418" text-anchor="middle">{it["t"]}</text>')
        s.append(f'<text x="{X(i)+bw/2:.1f}" y="{pad_t-3}" font-size="9" fill="#15803d" text-anchor="middle">ended {it["mult"]}×</text>')
        s.append(f'<text x="{X(i)+bw/2:.1f}" y="{H-6}" font-size="9" fill="#6b7280" text-anchor="middle">worst fall</text>')
    s.append("</svg>")
    return "".join(s)

# =======================  build charts  =======================
sk = E["skew_hist"]
c_skew = hist_svg(sk["labels"], sk["counts"], marker_idx=9, marker_lab=f'QQQ: +{sk["qqq"]*100:.0f}%', median_idx=6)
c_year = bars_svg([str(y)[2:] for y in E["beat_by_year"]["years"]], E["beat_by_year"]["beat"], ref=50, reflab="coin flip",
                  color="#b91c1c", fmt=lambda v: f"{v:.0f}")
rf = E["random_fans"]
c_fans = band_svg(rf["dates"], rf["p10"], rf["p50"], rf["p90"])
hw = E["hold_winners"]
c_hold = lines_svg(hw["dates"], [("QQQ", hw["qqq"], "#111418", 2.4, None), ("winners", hw["winners"], "#b91c1c", 2.0, None)], yearmod=2, logy=True)
c_wdd = dd_bars_svg(E["winners_dd"])
dw = E["dip_wait"]
c_dip = lines_svg(dw["dates"], [("DCA", dw["dca"], "#111418", 2.4, None), ("wait", dw["wait"], "#b91c1c", 1.8, "5 3")], yearmod=4)
qs = E["qqq_scar"]
c_scar = area_dd_svg(qs["dates"], qs["dd"])
qsp = E["qqq_spy"]
c_qspy = lines_svg(qsp["dates"], [("QQQ", qsp["q_curve"], "#111418", 2.4, None), ("SPY", qsp["s_curve"], "#6b7280", 1.8, None)], yearmod=4)
c_lev = lines_svg(lev_dates, [
    ("contrib", list(lev["CONTRIB"]), "#9ca3af", 1.2, "4 3"),
    ("SPY", list(lev["SPY"]), "#6b7280", 1.5, None),
    ("QQQ", list(lev["QQQ"]), "#111418", 2.4, None),
    ("3x", list(lev["LEV"]), "#b91c1c", 1.4, "2 3")], yearmod=4)
SB_SEL = [
    ("Machine-learning picker (honest)", 0.71, "sel", "0.47–0.71× across 8 configs"),
    ("Factor screen (quality+value+mom.)", 0.60, "sel", ""),
    ("Broad momentum picker", 0.26, "sel", "−82% drawdown"),
    ("Best fundamental screen", 1.21, "dead", "= luckiest random picker — died"),
    ("Mega-cap momentum", 6.63, "dead", "recency — died in audit"),
    ("Index-pond momentum", 2.05, "dead", "½ of lead in final 6 mo — died"),
    ("Published DCA picker (rebuilt)", 1.08, "dead", "claimed 2.2× — retracted"),
    ("Random pickers (control)", 0.58, "sel", "0.33–0.84× band"),
]
SB_TIM = [
    ("Momentum ETF rotation", 0.20, "sel", ""),
    ("Leveraged-ETF rotation (best)", 0.90, "sel", ""),
    ("Buy-the-dip rotation (best)", 0.45, "sel", "wins busts, loses bulls"),
    ("Trend switch (200-day)", 0.74, "dead", "0.74–3.31× by trade day — died"),
    ("Regime switch (best)", 1.08, "dead", "0.44× on other trade days — died"),
    ("Leveraged risk parity", 0.49, "sel", ""),
]
c_sb1 = scoreboard_svg(SB_SEL); c_sb2 = scoreboard_svg(SB_TIM)
c_traj = traj_svg(["2017", "2019", "2021", "2023", "2025", "2026-06"], [
    ("Mega-cap mom.", [0.94, 0.88, 2.06, 1.41, 3.88, 6.63], "#b91c1c"),
    ("Index-pond mom.", [0.90, 0.80, 1.26, 0.97, 1.31, 2.05], "#6b7280"),
    ("Best screen", [0.98, 1.12, 0.89, 0.96, 1.26, 1.21], "#9ca3af"),
])
lk = E["luck"]["streaks"]
c_luck = bars_svg(["1 yr", "3 yrs", "5 yrs", "8 yrs", "10 yrs"], [lk["1"], lk["3"], lk["5"], lk["8"], lk["10"]],
                  color="#6b7280", fmt=lambda v: f"{v:,.0f}")
pers = E["persistence"]
conc = E["concentration"]
lot = E["lottery"]
c_lottery = bars_svg(["loses money", "makes 10×+", "beats QQQ", "makes 100×+"],
                     [round(lot["p_lose"]*100,1), round(lot["p_10x"]*100,1), round(lot["p_beat_qqq"]*100,1), round(lot["p_100x"]*100,2)],
                     color="#6b7280", fmt=lambda v: f"{v:g}%", H=220)
blend_rows = "".join(
    f"<tr><td>{r['w']}% QQQ / {100-r['w']}% bonds</td><td class='r'>${r['final']:,}</td>"
    f"<td class='r bad'>{r['dd']}%</td><td class='r'>{r['final']/E['blends']['rows'][0]['final']*100:.0f}%</td></tr>"
    for r in E["blends"]["rows"])
sat_shortfall = 1 - (E["random_fans"]["final_median"]) ** (1/11.5)      # measured annual lag of a picked satellite
drag_rows = "".join(
    f"<tr><td>{int(w*100)}% picks / {int((1-w)*100)}% QQQ</td>"
    f"<td class='r'>{((1-w) + w*((1-sat_shortfall)**20))*100:.0f}%</td>"
    f"<td class='r bad'>−{(1-((1-w) + w*((1-sat_shortfall)**20)))*100:.0f}%</td></tr>"
    for w in [0.05, 0.10, 0.20, 0.50])
era_rows = "".join(f"<tr><td>{r['era']}</td><td class='r'>${r['q']:,}</td><td class='r'>${r['s']:,}</td>"
                   f"<td class='r {'good' if r['ratio']>=1 else 'bad'}'>{r['ratio']:.2f}×</td></tr>" for r in qsp["eras"])

def faq(q, a):
    return f"<details><summary>{q}</summary><div class='fa'>{a}</div></details>"

GH = "https://github.com/viki-m13/bonds/blob/main"

FAQS_PEOPLE = [
 ("“My friend / a guy on YouTube beats the market every year.”",
  "Given how many people pick stocks, thousands of market-beaters <i>must</i> exist by pure chance. A single concentrated portfolio beats QQQ in a given year roughly 40–45% of the time (measured, §3). Start 10,000 people flipping that coin and after 5 years about 185 of them have a perfect streak — after 10 years, 3 still do — <b>with zero skill involved</b> (chart in §6). The lucky ones post; the other 9,800 don't. You only ever hear from the right tail."),
 ("“Warren Buffett beat the market for decades.”",
  "He did — mostly in the 1950s–1990s, buying tiny neglected companies and whole private businesses, using cheap insurance-float leverage: a game a retail stock-picker cannot play. Over roughly the last two decades Berkshire has performed about in line with the S&amp;P 500. Buffett himself won a famous 10-year, million-dollar bet that an index fund would beat a hand-picked group of hedge funds — and his standing advice for ordinary investors is a low-cost index fund."),
 ("“Renaissance's Medallion fund makes 60%+ a year.”",
  "True, and instructive: Medallion is capped at roughly $10–15B (returns vanish at scale), closed to outside money for decades, and earns its returns from thousands of tiny, short-lived statistical edges executed with elite infrastructure — not from 'picking good stocks.' The same firm's funds that are <i>open</i> to outsiders have performed near the market. The existence of a closed, capacity-limited machine says nothing about what an individual can do in a brokerage account."),
 ("“Professionals with Bloomberg terminals and armies of analysts must beat it.”",
  "The public scorecard says the opposite: S&amp;P's SPIVA reports have shown, year after year, that roughly <b>85–90%+ of professional U.S. stock funds lag their benchmark over 15 years</b> — after fees, with every resource money can buy. The minority who lead in one period are not reliably the same ones who lead in the next (persistence studies). If the professionals can't, the base case for anyone else is worse."),
 ("“Somebody has to be beating the market — trading is a two-sided game.”",
  "Yes — and it's worth naming who actually wins: market-makers and high-frequency firms (paid a tiny spread billions of times — a different business, not stock-picking), insiders trading their own knowledge (illegal), a small set of closed, capacity-limited quantitative funds, and the temporarily lucky. What's missing from that list is 'a person at home selecting stocks from public information.' That specific game is the one this paper measures — and it loses."),
]
FAQS_STRATEGY = [
 ("“I'll just buy NVIDIA / Apple / the obvious winner.”",
  "Today's obvious winner is obvious <i>because it already won</i> — you can no longer buy the past returns. Buying today's hottest stocks and holding was tested directly on point-in-time data: the 20 hottest stocks of 2015, held to 2026, turned $20k into <b>$53k while QQQ turned it into $160k — and 10 of the 20 no longer exist</b> (§4). And spotting the next one early doesn't solve it: the historical winners fell −67% to −87% <i>on the way</i> to their gains (§4). Almost nobody holds a single stock through an −87% loss; the index made you hold, automatically."),
 ("“Momentum investing is proven by academics.”",
  "Momentum is real <i>as a relative pattern</i> — recent winners beat recent losers on average, for a while. But converting that into beating QQQ requires constant rotation, and every rotating version tested here failed the audits (luck-level results, era-dependence, trade-day sensitivity — §5), while the buy-and-hold version is a disaster (§4). QQQ is itself a free, automatic momentum machine: winners grow their own weight with zero trades and zero taxes."),
 ("“Value investing: buy good companies when they're cheap.”",
  "Tested as a systematic rule (buy statistically cheap/quality names), it produced 0.60× QQQ's result. The deep problem: for the last two decades the market's gains came overwhelmingly from stocks that <i>never looked cheap</i> — a value discipline would have kept you out of almost every big winner the whole way up. Value works as a risk story across a century of data; it has not been a way to beat a winner-riding index in this one."),
 ("“Buy the dip — it always comes back.”",
  "Two versions tested. Buying beaten-down <i>stocks</i>: the fallen names as a group keep underperforming (their −50%+ dips are often on the way to −100%; a quarter of all stocks lost money over a decade in which the index made +638%). Waiting in cash to buy <i>the index</i> after a big dip: measured directly — waiting for −20% crashes before investing ended 30% poorer than just investing every month (§7). Dips feel buyable in hindsight because the ones that recovered are the ones you remember."),
 ("“Dividend stocks pay you to wait — that's the safe way to win.”",
  "Dividends are not extra return — the share price drops by the dividend on payment; you're taxed on it annually; and dividend-screened portfolios concentrate in slow-growth sectors that structurally lagged this index. Fine as a preference for income; not a mechanism for beating a growth index, and none of the income-tilted configurations tested came close."),
 ("“Small caps outperform over time — I'll fish in that pond.”",
  "The most important negative result in the research: an <b>all-knowing oracle</b> — allowed to pick, with perfect foresight, the future best 5 stocks from the small/mid-cap pond — <i>still lost to QQQ-DCA in 3 of 5 eras</i>, because the pond as a whole lagged so badly that even its best swimmers couldn't compensate at portfolio scale. If perfection loses in that pond, no signal wins in it."),
 ("“Equal-weighting or rebalancing my picks reduces risk and adds return.”",
  "Every act of trimming winners and topping up losers moves money <i>from</i> the stocks driving all the gains <i>into</i> the ones that aren't. In a market where the top 1% of stocks produce ~19% of all net gains and the top 10% produce ~61% (§3), owning less of the winners is usually fatal. Tested: equal-weight and profit-taking variants underperformed both their let-it-ride versions and the index."),
 ("“Sell covered calls / puts for extra income.”",
  "Stock returns are extremely right-skewed: nearly all the money is in a few huge upward runs (§3). A covered call sells exactly that upside — for a small premium. It converts the one part of the distribution that pays into income today, which is why long-run covered-call indices lag their underlying. Income strategies feel good monthly and cost you the decade."),
 ("“I'll copy Congress members / famous investors / whale trackers.”",
  "Disclosures arrive with a lag (up to 45 days for both Congress and 13F filings), the copied portfolios are stale by the time you see them, and the copy-trackers' live results — where honestly measured — fail the same audits (era-dependence, luck-range results). The information you can copy is, by construction, information the market has already had weeks to price."),
 ("“Buy IPOs — get in on the ground floor of the next Amazon.”",
  "Measured on 6,599 IPOs including every failure: a published IPO system advertising 20.5%/yr delivered 8.1%/yr when rebuilt on honest data — below the index. Most IPOs underperform for years after listing; the ground floor is, on average, above fair value because insiders choose when to sell to you."),
 ("“Follow the insiders — executives buying their own stock.”",
  "Tested with the complete filing record: insider-cluster buying is one of the <i>better</i> signals — it lifts the odds a pick beats QQQ by ~4–6 percentage points. But the base rate is ~40%, so the best insider-informed picks still beat the index less than half the time. An edge that real, applied honestly, still wasn't enough to overcome the pond and concentration problems."),
 ("“AI changes everything — a model can find the winners now.”",
  "Modern machine learning was tested extensively on this exact question with clean, point-in-time data. The models genuinely <i>can</i> rank stocks — and still lost to QQQ, because their confident picks all cluster in the same crowded trades and fail together: measured directly, the ML's top picks landed in the following year's top performers <i>less often than randomly chosen stocks</i> (§5). Also: whatever a widely available AI knows is, by definition, available to everyone — and therefore already in prices."),
]
FAQS_PREMISE = [
 ("“Backtests aren't real life.”",
  "Correct — real life is <b>worse</b> for the picker. These simulations already charge trading costs and use only information available at each decision date, but they can't charge you the behavioral tax: real investors panic-sell in crashes and chase after rallies, which studies of actual investor returns put at another 1–2%/yr of loss. Every gap between simulation and reality widens the index's lead."),
 ("“QQQ just got lucky — tech happened to win this era.”",
  "Partly true, and the paper says so openly (§9): QQQ-DCA is a concentrated bet that suffered −81% in the dot-com crash. But the anti-picking verdict does not depend on QQQ: pickers failed against SPY too, professionals fail against every benchmark (SPIVA), and the skew/concentration math (§3) holds in every equity market ever measured. Choose a broader index if you prefer — just don't hand-pick stocks against it."),
 ("“If everyone indexes, markets break and picking will work again.”",
  "A theoretical limit that is nowhere near binding: trillions still trade actively every day, and price discovery needs remarkably little active money. If indexing ever did create big, easy mispricings, the professional funds already armed and paid to exploit them would harvest them long before a retail picker could. You don't need to volunteer to be the market's price-setter."),
 ("“Isn't QQQ itself just someone's stock picks?”",
  "It's a <i>rule</i>, not a judgment: the 100 largest non-financial Nasdaq companies, weighted by size, membership refreshed mechanically. Nobody decides 'this company will win'; the weighting means whatever wins automatically becomes a bigger holding, and whatever dies falls out on its own. That mechanical winner-riding — free, tax-deferred, emotionless — is precisely the thing discretionary picking keeps failing to replicate."),
 ("“A 10-year backtest window is cherry-picked.”",
  "The core statistics were measured across <i>every</i> era in 26 years of data, including both crashes: the fraction of stocks beating QQQ was computed for each year 2000–2025 (average 42%, §3), the winner-persistence numbers span 24 annual cohorts, and the strategy tests were run and re-run across eras, both halves of the sample, and dozens of start dates. The conclusion is the same in every honest slice — with one telling exception: most stocks 'beat QQQ' in 2000–2001, when QQQ itself was collapsing. Beating the index is easiest when the index is losing."),
]
FAQS_PRACTICAL = [
 ("“Should I DCA or invest a lump sum?”",
  "If you already have a lump sum, history favors investing it immediately about two times in three (markets rise more often than not, so waiting has a cost). DCA in this paper means the thing almost everyone actually does: investing money as it's earned, every paycheck. The point is not the cadence — biweekly vs monthly was tested and is a rounding error — it's the automation: no decisions, no timing, no picks."),
 ("“What about gold, crypto, real estate, international stocks?”",
  "Different question. Those are <i>asset-allocation</i> choices — how much risk, of which kinds, you want to hold — and diversification genuinely smooths the ride (at some cost to return). This paper is about one specific claim: that you can select <i>stocks</i> to beat a stock index. Nothing here argues against owning other assets; everything here argues against paying anyone (including yourself) to pick stocks."),
 ("“Can I pick a few stocks just for fun?”",
  "Yes — with a budget. Carve out a small fixed slice (say 5%), call it entertainment, and measure it honestly against what the same dollars would have done in QQQ. Expect it to lag (that's what the evidence says it will do), enjoy it like any hobby with a cost, and never let it grow into the plan. The danger isn't the 5% — it's the promotion of a lucky year into a strategy."),
 ("“What would it actually take to beat the index?”",
  "The research measured this precisely: surprisingly little skill — <i>if</i> the errors are independent. A simulated picker with only a slight, honest edge whose mistakes were random beat QQQ in every era. What no tested signal achieved is that error-independence: every public-data signal's bold picks are the same crowded bet in different tickers, so they fail together. Genuinely independent edges come from information about individual companies that others don't have — which, for a private individual trading public information, is either unavailable or illegal to use."),
]

faq_html = (
 "<h3>“But people do beat the market…”</h3>" + "".join(faq(q, a) for q, a in FAQS_PEOPLE) +
 "<h3>“But my approach would work…”</h3>" + "".join(faq(q, a) for q, a in FAQS_STRATEGY) +
 "<h3>“But the test itself is flawed…”</h3>" + "".join(faq(q, a) for q, a in FAQS_PREMISE) +
 "<h3>Practical questions</h3>" + "".join(faq(q, a) for q, a in FAQS_PRACTICAL))

html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Can Anyone Beat Just Buying QQQ?</title>
<style>
:root{{--txt:#111418;--mut:#6b7280;--line:#e5e7eb;--card:#fafafa;--good:#15803d;--bad:#b91c1c}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:var(--txt);background:#fff;line-height:1.55;font-size:15px}}
.wrap{{max-width:780px;margin:0 auto;padding:20px 16px 60px}}
header{{padding:30px 0 16px;border-bottom:2px solid var(--txt)}}
h1{{font-size:27px;letter-spacing:-.5px;line-height:1.2}}
.sub{{color:var(--mut);font-size:14px;margin-top:6px}}
h2{{font-size:15px;text-transform:uppercase;letter-spacing:1.2px;margin:36px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}}
h3{{font-size:14.5px;margin:20px 0 6px}}
p{{margin:10px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}}
.verdict{{border-left:4px solid var(--txt);font-size:16px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}}
.k{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;text-align:center}}
.k .v{{font-size:21px;font-weight:800}}
.k .l{{font-size:10.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}}
.big{{border-left:4px solid var(--txt)}}
.big .n{{font-size:26px;font-weight:800;line-height:1.1}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;margin:10px 0}}
th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--mut);padding:6px 8px;border-bottom:1px solid var(--line)}}
td{{padding:6px 8px;border-bottom:1px solid #f0f1f3}}
td.r,th.r{{text-align:right;font-variant-numeric:tabular-nums}}
.good{{color:var(--good);font-weight:700}}.bad{{color:var(--bad);font-weight:700}}
.note{{font-size:12.5px;color:var(--mut)}}
.leg{{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--mut);margin:6px 0}}
.leg i{{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px;border-radius:2px}}
.retract{{border-left:4px solid var(--bad)}} .retract b{{color:var(--bad)}}
ul{{margin:8px 0 8px 20px}} li{{margin:5px 0}}
.check li{{list-style:none;margin:7px 0;padding-left:24px;position:relative}}
.check li:before{{content:"✓";position:absolute;left:0;color:var(--good);font-weight:800}}
details{{border:1px solid var(--line);border-radius:8px;margin:8px 0;background:#fff}}
summary{{padding:10px 12px;font-weight:600;font-size:14px;cursor:pointer}}
.fa{{padding:0 14px 12px;font-size:14px;color:#1f2937}}
a{{color:var(--txt)}}
footer{{margin-top:44px;padding-top:14px;border-top:1px solid var(--line);font-size:12px;color:var(--mut)}}
.chart{{margin:10px 0;overflow-x:auto}}
.toc{{font-size:13.5px;columns:2;column-gap:24px}} .toc a{{display:block;margin:3px 0;text-decoration:none;color:#374151}}
@media(max-width:520px){{.toc{{columns:1}}}}
</style></head><body><div class="wrap">

<header>
<h1>Can Anyone Beat Just Buying QQQ?</h1>
<div class="sub">A plain-English research review of stock-picking, market timing, and why a boring automatic index purchase keeps winning. Every statistic below was computed from point-in-time market data — the record as it actually looked on each historical day, dead companies included.</div>
</header>

<div class="card verdict" style="margin-top:18px">
<b>The verdict:</b> on 26 years of honest data, <b>no method of picking stocks or timing markets reliably ends up with more money than automatically buying QQQ every two weeks or every month.</b> Not expert selection, not "buying the winners," not value screens, not dip-buying, not machine learning. This isn't because markets are magic. It's arithmetic: a few huge winners produce nearly all stock-market wealth, the index already owns them at full size and rides them automatically, and every act of picking makes you own less of them. The rest of this page shows the evidence, then answers every objection we could find.
</div>

<div class="kpis">
<div class="k"><div class="v">6.2%</div><div class="l">of stocks beat QQQ over the decade</div></div>
<div class="k"><div class="v">2,177</div><div class="l">stocks measured, incl. every death</div></div>
<div class="k"><div class="v">250+</div><div class="l">strategies tested &amp; failed</div></div>
<div class="k"><div class="v">~90%</div><div class="l">of professional funds lag (SPIVA)</div></div>
</div>

<div class="card"><nav class="toc">
<a href="#s1">1 · The rules of the test</a>
<a href="#s2">2 · Why QQQ (and not SPY)</a>
<a href="#s3">3 · The market is a lottery with few winning tickets</a>
<a href="#s4">4 · Why buying the winners fails</a>
<a href="#s5">5 · We tried to beat it 250+ ways</a>
<a href="#s6">6 · Luck explains your market-beating friend</a>
<a href="#s7">7 · Market-timing myths</a>
<a href="#s8">8 · "Then I'll take more risk"</a>
<a href="#s9">9 · What this does NOT say</a>
<a href="#s10">10 · Every objection, answered</a>
<a href="#s11">11 · The playbook: what to do</a>
<a href="#s12">Methodology &amp; sources</a>
</nav></div>

<h2 id="s1">1 · The rules of the test</h2>
<p>The question: investing a fixed amount on a fixed schedule (every two weeks or monthly — both tested, the difference is negligible), can <i>any</i> rule for choosing stocks or timing purchases end with more money than putting every contribution into QQQ?</p>
<p>Three rules make the test honest, and they matter more than any strategy:</p>
<ul>
<li><b>Point-in-time data.</b> Every simulated decision uses only information that existed on that day. No hindsight.</li>
<li><b>The dead are counted.</b> The data includes every stock that later went bankrupt, was delisted, or was acquired — at its real final price. Most "market-beating" backtests you'll see online quietly use only companies that still exist today, which is like studying lottery winners to conclude lotteries pay well. (Of the 2,177 investable stocks in 2016, <b>666 no longer traded by 2026</b>.)</li>
<li><b>Same money, same dates.</b> The benchmark receives identical contributions on identical days, and strategies pay realistic trading costs.</li>
</ul>

<h2 id="s2">2 · Why QQQ — and not SPY?</h2>
<p>QQQ holds the 100 largest non-financial companies on the Nasdaq, weighted by size — a mechanical rule, refreshed automatically, expense ratio 0.20%. SPY holds the S&amp;P 500. Both are fine instruments; the honest comparison, same monthly contributions since 2000 (through the worst possible start — the dot-com peak):</p>
<div class="chart">{c_qspy}</div>
<div class="leg"><span><i style="background:#111418"></i><b>QQQ-DCA</b></span><span><i style="background:#6b7280"></i>SPY-DCA</span><span>$1,000/month since Jan 2000, log scale</span></div>
<table><thead><tr><th>Period</th><th class="r">QQQ-DCA ends with</th><th class="r">SPY-DCA ends with</th><th class="r">QQQ ÷ SPY</th></tr></thead>
<tbody>{era_rows}</tbody></table>
<p>Three facts from that table: <b>(1)</b> QQQ-DCA finished ahead in four of five eras and in <b>{qsp['roll5_winrate']}% of all rolling 5-year windows</b>; over the full 26 years it ends with about twice the money. <b>(2)</b> The one era it lost (2000–04) is the honest reminder that it's a technology-heavy bet. <b>(3)</b> Measured on the contribution schedule, the two had essentially the <b>same worst drawdown</b> ({qsp['q_mdd']}% vs {qsp['s_mdd']}%) — a steady contributor bought right through the crashes.</p>
<p>Why does a top-100 growth index keep beating the broader 500? Because it is a purer expression of the one force this whole paper is about: <b>letting winners run at full weight</b>. Fewer names, more of the winners, no committee judgment — just size-weighting doing the work. That said, <b>nothing in this paper depends on choosing QQQ</b>: stock-pickers lose to SPY too (and ~90% of professionals lag whatever index they're measured against). Prefer the broader bet? Take it. The mistake isn't which index — it's leaving the index to pick stocks.</p>

<h2 id="s3">3 · The market is a lottery with a few winning tickets — and the index holds them all</h2>
<p>Here is every investable U.S. stock at mid-2016 — all {sk['n']:,} of them, including the {sk['died']} that later died — and what each returned over the following decade:</p>
<div class="chart">{c_skew}</div>
<div class="leg"><span><i style="background:#b91c1c"></i>lost money</span><span><i style="background:#15803d"></i>made money</span><span>each bar = number of stocks; dashed lines = QQQ (+{sk['qqq']*100:.0f}%) and the median stock</span></div>
<div class="card big"><div class="n">6.2%</div>That's the fraction of stocks that beat QQQ over that decade — roughly <b>1 in 16</b>. A quarter lost money outright in one of the best decades markets have ever had. The <i>median</i> stock returned +{sk['median']*100:.0f}% in total while QQQ returned +{sk['qqq']*100:.0f}%.</div>
<p>And the gains that do exist are brutally concentrated: the top 1% of stocks produced <b>~{conc['share_of_net'][3]:.0f}%</b> of all net gains; the top 10% produced <b>~{conc['share_of_net'][6]:.0f}%</b>. This mirrors the famous academic result (Bessembinder, 2018) that a few dozen companies account for most of the stock market's entire net wealth creation since 1926, and <i>most individual stocks underperform cash</i>.</p>
<p>It's not one lucky decade. Year by year since 2000, the fraction of stocks beating QQQ over the following 12 months averaged <b>42%</b> — and the years it exceeded 50% are mostly 2000–2001, when "beating QQQ" meant falling less than an index that was crashing:</p>
<div class="chart">{c_year}</div>
<div class="leg"><span>% of stocks beating QQQ over the next 12 months, each year 2000–2025 (dashed line = coin flip)</span></div>
<p><b>This is the whole game.</b> Picking stocks means trading away guaranteed full-size ownership of the few tickets that pay for a handful of tickets that each probably won't. The index isn't smart — it just refuses to hand any winning ticket back.</p>

<h2 id="s4">4 · Why “buying the winners” specifically fails</h2>
<p>The most seductive idea in investing: <i>just buy what's already winning.</i> Three measured facts kill it.</p>
<h3>4.1 The winners list doesn't stay the winners list</h3>
<div class="card big"><div class="n">15%</div>Of each year's top-10% performers, that's how many repeat in the top 10% the following year. <b>{pers['below_median']:.0f}%</b> fall below the <i>median</i> stock; only <b>{pers['beat_qqq_next']:.0f}%</b> beat QQQ. (Measured across 24 annual cohorts, 2001–2024.)</div>
<h3>4.2 Buying the hot list and holding is a disaster</h3>
<p>Done literally on point-in-time data: at the start of 2015, buy the 20 hottest stocks in America (the top of the 12-month leaderboard), $1,000 each, and hold:</p>
<div class="chart">{c_hold}</div>
<div class="leg"><span><i style="background:#111418"></i>QQQ: $20k → ${hw['qqq'][-1]/1e3:.0f}k</span><span><i style="background:#b91c1c"></i>the 20 winners: $20k → ${hw['winners'][-1]/1e3:.0f}k</span><span><b>{hw['dead']} of the 20 no longer exist</b></span></div>
<p>Why so bad? Because the top of a 12-month leaderboard is where fragile, over-extended, often speculative names congregate — the measured odds of an <i>extreme</i> recent winner beating QQQ over the next year are just <b>5–17%</b>, the worst forward odds of any group studied.</p>
<h3>4.3 Even the TRUE winners would have shaken you out</h3>
<p>Suppose you somehow did buy a generational winner early. Here's what the era's best stocks did to their holders on the way to their gains:</p>
<div class="chart">{c_wdd}</div>
<p>Each of these fell by two-thirds or more — several more than once — before delivering its legendary return. Holding one stock through an −80% loss, with headlines screaming it's over, is something almost no one does. The index held them for you: no conviction required, no decision available to get wrong. <b>The index doesn't just find the winners — it forces you to keep them.</b></p>

<h2 id="s5">5 · We didn't take this on faith — we tried to beat it, 250+ ways</h2>
<p>Over multiple independent research campaigns, 250+ strategy configurations were built and tested on the honest data: machine-learning models trained on 36 predictive features, factor screens, momentum systems, insider-filing signals, dip-buyers, sector and ETF rotators, trend switches, regime detectors. The scoreboard (each family's <b>best</b> configuration, most charitable reading):</p>
<p class="note" style="font-weight:700;color:#111418;margin-bottom:2px">Stock-picking approaches (final wealth ÷ QQQ-DCA's final wealth)</p>
<div class="chart">{c_sb1}</div>
<p class="note" style="font-weight:700;color:#111418;margin-bottom:2px">Timing / rotation approaches</p>
<div class="chart">{c_sb2}</div>
<p>Faded bars looked like winners — until audited. Every apparent beat failed at least one of five checks any honest claim must pass: <b>random-picker controls</b> (does luck alone produce this?), <b>timing of the lead</b>, <b>other eras</b>, <b>other trade days</b>, and <b>survivorship-clean data</b>:</p>
<div class="chart">{c_traj}</div>
<div class="leg"><span>The same "winning" backtests, stopped at earlier dates (lead vs QQQ-DCA). Real edges grow steadily; these leads appear suddenly, late — the signature of luck riding one hot stretch.</span></div>
<p>Two details worth knowing. First, the "best" stock screen's result (1.21×) exactly equals the <i>luckiest</i> of the random-picker controls — indistinguishable from chance. Second, the strongest ML models genuinely could rank stocks, yet their most confident picks landed among the next year's top performers <b>less often than randomly chosen stocks</b> (8.3% vs 10.2%) — their bold picks were all the same crowded bet in different tickers, so they failed together. Skill existed; it was the wrong shape.</p>
<div class="card retract"><b>We retracted our own winners.</b> Three strategies from this research were themselves published as market-beaters, then independently rebuilt and audited: a machine-learning stock picker (two data flaws found — honest version lands <i>below</i> QQQ), a biweekly DCA stock-selection system (half its edge was survivorship bias and recency; honest version is a coin flip), and a leveraged-ETF timing system (data leakage; rebuilt honestly, its return came from extra risk, not skill). If we hold everyone else's claims to these audits, we hold our own to them too — <b>that is why you can trust the negative result.</b></div>

<h2 id="s6">6 · About your market-beating friend: the luck math</h2>
<p>A concentrated stock portfolio beats QQQ in a given year roughly 40–45% of the time (that's the measured base rate — losing more often than winning, but not by much in any single year). Now imagine 10,000 people each running one portfolio, with <b>zero skill</b>, just those coin odds:</p>
<div class="chart">{c_luck}</div>
<div class="leg"><span>Expected number (out of 10,000 zero-skill pickers) still holding a perfect beat-the-market streak of each length</span></div>
<p>After five years, ~185 flawless five-year track records exist by pure chance. After ten years — three. Those people are not lying about their returns; they're real, they're confident, and they're indistinguishable from skilled until the streak ends. They post on YouTube; the 9,815 others don't. <b>Survivorship bias isn't just a data problem — it's your entire social-media feed.</b></p>

<h2 id="s7">7 · The market-timing myths, measured</h2>
<p><b>"Wait for the crash, then buy."</b> Simulated directly: hold every contribution in cash until the index is at least 20% off its high, then invest it all. Since 2003:</p>
<div class="chart">{c_dip}</div>
<div class="leg"><span><i style="background:#111418"></i>invest every month → ${dw['dca'][-1]/1e6:.2f}M</span><span><i style="background:#b91c1c"></i>wait for −20% dips → ${dw['wait'][-1]/1e6:.2f}M</span></div>
<p>Waiting for the obvious bargain cost <b>~30% of final wealth</b>. Crashes are rare; cash waits for years earning nothing while the index compounds; and the "obvious" bottom is only obvious afterward. (In live crashes, the same people waiting for −20% decide at −20% to wait for −30%.)</p>
<p><b>"Get out when it looks dangerous, back in when it's safe."</b> Every tested version of this — moving-average switches, volatility triggers, regime detectors — either lost outright or turned out to depend on <i>which day of the month</i> it happened to trade (the same rule returned anywhere from 0.74× to 3.31× depending on the trade date — a coin toss wearing a lab coat). None survived the audits.</p>

<h2 id="s8">8 · “Fine — then I'll just take more risk”</h2>
<p>One thing does reliably end with more money in a rising market: <b>more exposure</b>. A 3×-leveraged version of the same index, same contributions:</p>
<div class="chart">{c_lev}</div>
<div class="leg"><span><i style="background:#9ca3af"></i>contributions</span><span><i style="background:#6b7280"></i>SPY</span><span><i style="background:#111418"></i>QQQ</span><span><i style="background:#b91c1c"></i>3× leveraged</span><span>$1,000/month, 2006–2026, log scale</span></div>
<p>The red line ends highest — and fell <b>−84%</b> along the way; started three years earlier, it hits <b>−99.9%</b> in the dot-com crash and never recovers. That's the honest decoder for every impressive line that ends above QQQ in any backtest, anywhere: look for the extra risk. More risk is a legitimate <i>choice</i>; it is not skill, it doesn't need a guru, and almost no one actually holds through the part where it's down 84%.</p>

<h2 id="s9">9 · What this paper does NOT say</h2>
<ul>
<li><b>It does not say QQQ is safe.</b> A QQQ investor in March 2000 fell <b>{qs['min']:.0f}%</b> and waited roughly 15 years to break even on a lump sum. This is a concentrated technology bet, and this century has been technology's century — the next one may not be:</li>
</ul>
<div class="chart">{c_scar}</div>
<div class="leg"><span>QQQ's distance below its own prior peak, 1999–2017 — the scar this paper refuses to hide</span></div>
<ul>
<li><b>It does not say markets are perfectly efficient</b> — only that the specific game of out-picking a winner-riding index using public information is stacked, measurably, against the picker.</li>
<li><b>It does not say never own anything else.</b> Diversifying across asset classes (bonds, international, etc.) genuinely smooths the ride — in exchange for return. That's a preference, not an error.</li>
<li><b>It does not promise the next 26 years look like the last.</b> It says: whatever index you choose, no tested method of picking stocks against it has honestly beaten contributing to it on autopilot.</li>
</ul>

<h2 id="s10">10 · Every objection we could find, answered</h2>
{faq_html}

<h2 id="s11">11 · The playbook: exactly what to do</h2>
<p>Evidence without instructions is trivia. Here is the complete, quantified playbook this research supports — including honest answers to "how much?", "what about picking a few stocks anyway?", and "when do I sell?".</p>

<h3>11.1 &nbsp;First, the order of operations (before any investing)</h3>
<ul class="check">
<li><b>Cash buffer first:</b> 3–6 months of expenses. Its job is to make sure you are never forced to sell stocks in a crash.</li>
<li><b>Free money second:</b> any employer retirement match, always, fully.</li>
<li><b>Tax-advantaged accounts before taxable</b> (401k/IRA equivalents): the same QQQ-DCA compounds meaningfully faster untaxed.</li>
<li><b>Only money you won't need for 10+ years goes into equities.</b> The index fell −81% once (§9) and −32% on a DCA account twice in 20 years. Five-year money belongs in bonds/cash regardless of what any chart says.</li>
</ul>

<h3>11.2 &nbsp;How much into QQQ? Pick the row you can sleep through</h3>
<p>Measured on the same $1,000/month, 2006–2026 (bonds = a total U.S. bond fund):</p>
<table><thead><tr><th>Allocation</th><th class="r">Ends with</th><th class="r">Worst account fall</th><th class="r">vs 100% QQQ</th></tr></thead>
<tbody>{blend_rows}</tbody></table>
<p>Read it plainly: <b>every 20% moved to bonds bought about 4 points of shallower crash and cost about a third of final wealth.</b> The right row is not the top one — it's the one whose "worst fall" you will genuinely hold through, because the single most expensive event in investing is abandoning the plan at the bottom (real investors' behavior gap costs an estimated 1–2%/yr on its own). Two honest defaults: <b>long horizon + strong stomach → 80–100% in the index</b>; any doubt about the stomach → <b>60–80%</b> with the rest in boring bonds. If the tech concentration of QQQ specifically worries you (§9), substitute a broad-market fund at the same percentages — that choice matters far less than automating it.</p>
<p><b>Should you "always and only" DCA into QQQ?</b> Always: automate contributions into your chosen index. Only: no — hold the cash buffer and the bond ballast above, and rebalance <i>with new contributions</i> (direct paycheck money toward whichever side is under target) rather than by selling. What the evidence says to never do is the third thing: buying individual stocks <i>as the plan</i>.</p>

<h3>11.3 &nbsp;“But people made fortunes on Apple and Tesla — why can't I?”</h3>
<p>Because you are looking at the winners of a lottery and asking why you can't buy winning tickets. The odds, measured over 21 years (every investable U.S. stock in 2005, followed to 2026, deaths included — {lot['n']:,} stocks):</p>
<div class="chart">{c_lottery}</div>
<div class="leg"><span>What a single randomly-chosen 2005 stock did over 21 years. QQQ over the same period: <b>{lot['qqq_mult']:.0f}×</b>.</span></div>
<ul>
<li><b>1 in 5</b> single-stock tickets lost money over two decades of a rising market. The <i>median</i> ticket made {lot['pct']['p50']:.1f}× while QQQ made <b>{lot['qqq_mult']:.0f}×</b> — so the typical stock pick trailed the index by a factor of ~7.</li>
<li><b>1 in 6</b> made 10× — respectable, still less than the index's 24×.</li>
<li><b>1 in 22</b> beat the index at all. <b>1 in ~560 (0.18%)</b> was an Apple-class 100-bagger.</li>
<li>And drawing the golden ticket is not enough: you then had to <b>size it large, hold it through −67% to −87% collapses</b> (§4.3) — several times — <b>and never take profits</b> after the first 10×. Each requirement eliminates almost everyone who starts. That triple filter — rare ticket × meaningful size × inhuman holding — is why the Apple stories are famous: <b>famous is what nearly-impossible looks like from the outside.</b></li>
</ul>
<p>Meanwhile the index holder got the Apple outcome anyway — at the index's weight, automatically, with none of the decisions. That is the quiet punchline of this entire paper: <b>owning QQQ IS owning Apple, Nvidia and Tesla — at full size, without needing to be right.</b></p>

<h3>11.4 &nbsp;If you're going to pick stocks anyway: the satellite rules</h3>
<p>Realistically, some readers will pick stocks regardless. Fine — do it in a way that caps the damage and keeps the compounding machine intact. The quantified rules:</p>
<ul>
<li><b>Cap the slice at 5–10% of your portfolio, hard.</b> The measured cost: portfolios of hand-picked stocks trailed QQQ by ≈{sat_shortfall*100:.1f}%/yr (median of 100 random-pick portfolios; skilled-looking signals did no better after audits). Over 20 years that lag costs:</li>
</ul>
<table><thead><tr><th>Slice given to picks</th><th class="r">Expected wealth vs 100% QQQ</th><th class="r">Cost of the hobby</th></tr></thead>
<tbody>{drag_rows}</tbody></table>
<ul>
<li><b>Buy once or DCA in — doesn't much matter; the cap is what matters.</b> A satellite position's fate is dominated by <i>which</i> stock it is, not how you entered it.</li>
<li><b>What to look for (measured, not vibes):</b> the only signals that tested <i>positive</i> were boring ones — sustained insider <i>cluster</i> buying (+4–6 points to the odds) and steady profitable-quality names. What tested <i>worst</i> is exactly what feels best: the 12-month leaderboard (5–17% forward odds), story stocks, and whatever is on your feed. If your pick is exciting, that's a warning, not a signal.</li>
<li><b>Diversify the slice across 8–10 names</b> — or accept that 1–3 names is a lottery ticket (§11.3 odds) and size it like one.</li>
<li><b>Never average down.</b> "It's cheaper now" is how 20% of stocks ride to zero. The measured fate of falling former winners: 56% land below the median stock the next year. Adding to losers is moving money in the exact opposite direction of everything §3 showed.</li>
<li><b>Selling rules, decided in advance:</b> the QQQ core is never sold (that's the whole edge). A satellite <i>winner</i> is left alone until it outgrows your cap — then trimmed <b>into the core</b> (winner → index-of-winners; never into your losers). A satellite <i>loser</i> needs no decision: it simply never gets another dollar, and it dies or lives on its own. This preserves the let-winners-run principle at every level while capping single-stock risk.</li>
<li><b>Measure it honestly once a year</b> against what the same dollars in QQQ did. The evidence says it will lag; when it does, you'll have paid a known, capped price for the fun — and when a pick 10×'s, you'll enjoy it without having bet the plan on it.</li>
</ul>

<h3>11.5 &nbsp;The do-not list (each one measured somewhere above)</h3>
<ul>
<li>✗ No waiting in cash for crashes (§7: cost ~30% of final wealth).</li>
<li>✗ No on/off market timing switches (§7: trade-day lottery).</li>
<li>✗ No leverage you haven't priced at −84% (§8).</li>
<li>✗ No selling winners to buy losers, anywhere, ever (§3–4).</li>
<li>✗ No strategies sold on a backtest that hasn't passed the five audits (§5).</li>
<li>✗ No acting on streaks — yours or anyone's (§6: three 10-year streaks per 10,000 coin-flippers).</li>
</ul>

<h2 id="s12">Methodology &amp; sources</h2>
<p class="note">All statistics computed from point-in-time, delisting-inclusive U.S. market data: ~24,000 tickers including ~8,900 that no longer trade, 1990–2026; prices adjusted for splits/dividends; disappeared stocks counted at their final traded price (acquisitions exit at deal price); liquidity floor (price ≥ $3, median daily volume ≥ $2M) applied at each historical date using only that date's information. Strategy tests charge 5–20 bps per side and give the benchmark identical cash flows. Charts generated by <a href="{GH}/scripts/gen_verdict.py">gen_verdict.py</a> from <a href="{GH}/scripts/verdict_evidence.py">verdict_evidence.py</a>; underlying research records: <a href="{GH}/dca/research/strategies/ascent/FINDINGS.md">stock-selection studies</a>, <a href="{GH}/leverage_etf_dca/README.md">ETF-timing studies</a>, <a href="{GH}/dca/README.md">DCA-selection validation</a>, <a href="{GH}/dca/research/strategies/METHODOLOGY_validation.md">validation playbook</a>. External: Bessembinder, <i>Do Stocks Outperform Treasury Bills?</i> (2018); S&amp;P SPIVA scorecards; Buffett's 2008–2017 index-vs-hedge-funds bet.</p>
<footer>Research, not investment advice. Backtests are simulations; past performance does not guarantee future results.</footer>
</div></body></html>"""

out = f"{ROOT}/docs/verdict.html"
open(out, "w").write(html)
print(f"written {out} ({len(html):,} bytes)")
