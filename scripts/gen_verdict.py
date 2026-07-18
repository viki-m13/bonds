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
    dd = [v if (v is not None and np.isfinite(v)) else 0.0 for v in dd]
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

def hbars_svg(rows, W=660, xmax=100, fmt=lambda v: f"{v:g}%", color="#b91c1c", H_row=30):
    """rows: (label, value, note)"""
    H = len(rows)*H_row + 16
    s2 = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" {MINW} role="img">']
    y = 8
    for row in rows:
        lab, v, note = row[0], row[1], row[2]
        rowcolor = row[3] if len(row) > 3 else color
        w = (W-230-70)*min(v,xmax)/xmax
        s2.append(f'<text x="224" y="{y+14}" font-size="10.5" fill="#111418" text-anchor="end">{lab}</text>')
        s2.append(f'<rect x="230" y="{y+4}" width="{max(w,2):.1f}" height="14" fill="{rowcolor}" rx="2"/>')
        s2.append(f'<text x="{234+max(w,2):.1f}" y="{y+15}" font-size="10.5" font-weight="700" fill="#111418">{fmt(v)}</text>')
        if note: s2.append(f'<text x="{234+max(w,2)+46:.1f}" y="{y+15}" font-size="9" fill="#6b7280">{note}</text>')
        y += H_row
    s2.append("</svg>")
    return "".join(s2)

def paired_bars_svg(items, W=680, H=250):
    """items: (label, v1, v2, note) — v1 red (theme), v2 black (QQQ), log-ish scaled by max"""
    import math as _m
    pad_l, pad_t, pad_b = 44, 26, 40
    n = len(items); gw = (W-pad_l-10)/n
    mx = max(max(v1, v2) for _, v1, v2, _ in items)
    def Y(v): return pad_t + (H-pad_t-pad_b)*(1 - (_m.log10(max(v,0.1))-_m.log10(0.1))/(_m.log10(mx*1.2)-_m.log10(0.1)))
    s2 = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" {MINW} role="img">']
    for v in [1, 10]:
        y = Y(v)
        s2.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-10}" y2="{y:.1f}" stroke="#eeeeee"/>'
                  f'<text x="{pad_l-5}" y="{y+3:.1f}" font-size="10" fill="#9ca3af" text-anchor="end">{v}×</text>')
    for i, (lab, v1, v2, note) in enumerate(items):
        x = pad_l + gw*i + 8
        bw = (gw-24)/2
        s2.append(f'<rect x="{x:.1f}" y="{Y(v1):.1f}" width="{bw:.1f}" height="{(H-pad_b)-Y(v1):.1f}" fill="#b91c1c" rx="2"/>')
        s2.append(f'<text x="{x+bw/2:.1f}" y="{Y(v1)-4:.1f}" font-size="9.5" font-weight="700" fill="#7f1d1d" text-anchor="middle">{v1:g}×</text>')
        s2.append(f'<rect x="{x+bw+4:.1f}" y="{Y(v2):.1f}" width="{bw:.1f}" height="{(H-pad_b)-Y(v2):.1f}" fill="#111418" rx="2"/>')
        s2.append(f'<text x="{x+bw+4+bw/2:.1f}" y="{Y(v2)-4:.1f}" font-size="9.5" font-weight="700" fill="#111418" text-anchor="middle">{v2:g}×</text>')
        s2.append(f'<text x="{x+bw+2:.1f}" y="{H-24}" font-size="10.5" font-weight="700" fill="#111418" text-anchor="middle">{lab}</text>')
        s2.append(f'<text x="{x+bw+2:.1f}" y="{H-12}" font-size="8.6" fill="#6b7280" text-anchor="middle">{note}</text>')
    s2.append("</svg>")
    return "".join(s2)

def conc_svg(ks, shares, ntot, W=680, H=260):
    import math as _m
    pad_l, pad_r, pad_t, pad_b = 46, 14, 12, 34
    def X(k): return pad_l + (W-pad_l-pad_r)*(_m.log10(k)-0)/(_m.log10(ntot)-0)
    def Y(v): return pad_t + (H-pad_t-pad_b)*(1-v/100.0)
    s2 = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" {MINW} role="img">']
    for v in [25, 50, 75, 100]:
        y = Y(v)
        s2.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" stroke="#eeeeee"/>'
                  f'<text x="{pad_l-5}" y="{y+3:.1f}" font-size="10" fill="#9ca3af" text-anchor="end">{v}%</text>')
    for k, lab in [(1, "top 1"), (10, "top 10"), (100, "top 100"), (ntot, f"all {ntot:,}")]:
        x = X(k)
        s2.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{H-pad_b}" stroke="#f3f4f6"/>'
                  f'<text x="{x:.1f}" y="{H-8}" font-size="9.5" fill="#6b7280" text-anchor="middle">{lab}</text>')
    pts = [(1e-9+k, v) for k, v in zip(ks, shares)] + [(ntot, 100.0)]
    poly = " ".join(f"{X(k):.1f},{Y(v):.1f}" for k, v in pts)
    s2.append(f'<polyline points="{poly}" fill="none" stroke="#b91c1c" stroke-width="2.4"/>')
    for k, v in pts[:-1]:
        s2.append(f'<circle cx="{X(k):.1f}" cy="{Y(v):.1f}" r="3" fill="#b91c1c"/>')
        s2.append(f'<text x="{X(k)+5:.1f}" y="{Y(v)-6:.1f}" font-size="9.5" font-weight="700" fill="#7f1d1d">{v:g}%</text>')
    s2.append("</svg>")
    return "".join(s2)

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
sp = E["sector_persist"]
th = E["themes"]
sky = E["skill_years"]
rf_final = {"p10": E["random_fans"]["p10"][-1], "p50": E["random_fans"]["p50"][-1], "p90": E["random_fans"]["p90"][-1]}
p10x = E["lottery"]["p_10x"]
p_any10x_8 = 1 - (1 - p10x)**8
c_profitbeat = hbars_svg([
    ("made money (2016–26)", round(E["skew_hist"]["lost_money"]*-100+100,1), "the bull market pays almost everyone"),
    ("beat QQQ (2016–26)", round(E["skew_hist"]["beat"]*100,1), "the machine is the hard part"),
], color="#6b7280")
c_ladder = hbars_svg([
    ("beat QQQ over 1 year", 42.0, "avg across 26 annual cohorts"),
    ("beat QQQ over 10 years", 6.2, "2,177 stocks, deaths included"),
    ("beat QQQ over 21 years", 4.6, "1,141 stocks"),
    ("100× over 21 years (an 'Apple')", 0.18, "≈ 1 in 560"),
])
c_skillyears = hbars_svg([
    (f"a rare true edge (IR 1.0)", 4, "years of live results needed"),
    (f"an excellent manager (IR 0.75)", 7, ""),
    (f"a good manager (IR 0.5)", 16, "longer than most careers"),
    (f"a modest real edge (IR 0.25)", 64, "longer than an investing lifetime"),
], xmax=64, fmt=lambda v: f"{v:g} yrs", color="#6b7280")
c_hedge = hbars_svg([
    ("index fund (Buffett's bet, 2008–17)", 125.8, "cumulative return"),
    ("five hedge-fund portfolios (same bet)", 36.3, "average, after fees"),
], xmax=130, fmt=lambda v: f"+{v:g}%", color="#b91c1c")
c_sector = hbars_svg([
    ("repeats as #1 next year", sp["repeat_no1"], ""),
    ("stays top-3 next year", sp["stay_top3"], ""),
    ("beats QQQ next year", sp["beat_qqq_next"], ""),
    ("falls BELOW the sector median", sp["fall_below_median"], "the most likely outcome"),
])
c_themes = paired_bars_svg([(t["t"], t["mult"], t["qqq_mult"], f"since {t['since']}, fell {t['dd']}%") for t in th])
pers = E["persistence"]
conc = E["concentration"]
lot = E["lottery"]
td = E["topdogs"]
td_rows = "".join(
    f"<tr><td>{r['y']} giants</td><td>{', '.join(n for n in r['names'][:4])}…</td>"
    f"<td class='r'>{r['med']:+d}%</td><td class='r'>{r['qqq']:+d}%</td><td class='r bad'>{r['n_beat']}/10</td></tr>"
    for r in td)
td_beat_total = sum(r["n_beat"] for r in td)
c_lottery = bars_svg(["loses money", "makes 10×+", "beats QQQ", "makes 100×+"],
                     [round(lot["p_lose"]*100,1), round(lot["p_10x"]*100,1), round(lot["p_beat_qqq"]*100,1), round(lot["p_100x"]*100,2)],
                     color="#6b7280", fmt=lambda v: f"{v:g}%", H=220)

sat_shortfall = 1 - (E["random_fans"]["final_median"]) ** (1/11.5)      # measured annual lag of a picked satellite
drag_rows = "".join(
    f"<tr><td>{int(w*100)}% picks / {int((1-w)*100)}% QQQ</td>"
    f"<td class='r'>{((1-w) + w*((1-sat_shortfall)**20))*100:.0f}%</td>"
    f"<td class='r bad'>−{(1-((1-w) + w*((1-sat_shortfall)**20)))*100:.0f}%</td></tr>"
    for w in [0.05, 0.10, 0.20, 0.50])
menu = E["menu"]
ef = E["eraflip"]
_menu_rows = [(r["name"], r["final"]/1e6, f"worst fall {r['dd']}%",
               "#111418" if r["t"] in ("QQQ", "SPY") else "#b91c1c") for r in menu["rows"]]
c_menu = hbars_svg(_menu_rows, xmax=2.6, fmt=lambda v: f"${v:.2f}M")
mach = E["machines"]
_mc = mach["curves"]
c_machines = lines_svg(_mc["dates"], [
    ("GLD", _mc["GLD"], "#b0891b", 1.5, "4 3"),
    ("EFA", _mc["EFA"], "#9ca3af", 1.4, None),
    ("EWJ", _mc["EWJ"], "#6b7280", 1.4, "2 3"),
    ("EEM", _mc["EEM"], "#d1d5db", 1.4, None),
    ("SPY", _mc["SPY"], "#374151", 1.9, None),
    ("QQQ", _mc["QQQ"], "#111418", 2.6, None),
], yearmod=4, H=300)
_ms = mach["stats"]
fvs = E.get("fans_vs_spy", {})
rel = E["qqq_rel_spy"]
c_rel = lines_svg(rel["dates"], [("QQQ vs SPY", rel["rel"], "#b91c1c", 2.2, None)],
                  logy=False, ylines=[0.5, 1.0, 1.5, 2.0], yfmt=lambda v: f"{v:g}×", yearmod=4, H=240)
c_regret = hbars_svg([
    ("chose the 'wrong' INDEX (26 yrs)", 48, "SPY instead of QQQ — still 8× your money"),
    ("median 10-stock portfolio (11.5 yrs)", 58, "vs QQQ-DCA, same dollars"),
    ("median single stock (10 yrs)", 23, "+71% vs QQQ's +638%"),
    ("worst case, single stock", 0, "−100%: 666 of 2,177 died"),
], xmax=100, fmt=lambda v: f"{v:g}%", color="#6b7280")
c_audit = hbars_svg([
    ("ML stock picker (ours)", 0, "claimed 21.5%/yr -> honest 12%/yr, below index"),
    ("DCA selection system (ours)", 7, "claimed 2.2x QQQ -> honest 1.08x"),
    ("Published IPO system", 0, "claimed 20.5%/yr -> honest 8.1%/yr, below index"),
    ("Mega-cap momentum backtest", 0, "6.63x apparent -> luck-level in audits"),
], xmax=100, fmt=lambda v: f"{v:g}%", color="#b91c1c")
c_conc = conc_svg(conc["ks"], conc["share_of_net"], conc["n"])
c_fate = hbars_svg([
    ("repeats in the top 10%", pers["repeat_top_decile"], ""),
    ("beats QQQ next year", pers["beat_qqq_next"], ""),
    ("falls below the MEDIAN stock", pers["below_median"], "the most likely outcome"),
])
c_shape = hbars_svg([
    ("random picks", 10.2, "baseline"),
    ("the ML's most confident picks", 8.3, "worse than random - crowded bets fail together"),
    ("a modest edge with RANDOM errors", 27.5, "simulated: what a usable edge looks like"),
], xmax=30, color="#6b7280")
qcurve = qsp["q_curve"]; _qs = pd.Series(qcurve)
c_ddhist = area_dd_svg(qsp["dates"], [round(float(x)*100, 1) for x in (_qs/_qs.cummax()-1).tolist()])

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
  "The public scorecard says the opposite: S&amp;P's SPIVA reports have shown, year after year, that <b>89.5% of professional U.S. large-cap funds lagged the S&amp;P 500 over the 15 years through 2024</b> [12] — after fees, with every resource money can buy. The minority who lead in one period are not reliably the same ones who lead in the next (persistence studies). If the professionals can't, the base case for anyone else is worse."),
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
FAQS_STRATEGY += [
 ("“Didn't monkeys throwing darts beat the index?”",
  "The famous dart-throwing-monkey portfolios (and the studies behind them) beat <i>cap-weighted</i> indexes historically because random picking systematically over-weights small and cheap stocks — hidden extra risk that paid off in those samples. In the recent mega-cap era that same tilt was exactly what lost (§3): our 100 random portfolios — same idea, honest modern data — ended at a median 0.58× QQQ with 6% winning. The monkeys weren't skilled; they were levered to a bet that has since stopped paying."),
 ("“What about day trading? I've seen the funded-account guys.”",
  "The best evidence is brutal: tracking <b>every person</b> who began day-trading index futures in Brazil over three years, 97% of those who stuck with it 300+ days lost money, only 1.1% out-earned minimum wage, and performance did not improve with experience [16]. U.S. brokerage data points the same way: the most active retail traders lagged the market by 6.5 points a year [14]. Day trading is the picking problem of this paper, repeated hundreds of times a year at maximum cost."),
]
FAQS_STRATEGY += [
 ("“Why not just buy the Magnificent 7 / today's giants?”",
  "Because today's giants are the previous game's winners, and the crown is heavy. We measured it directly: take the 10 most-dominant U.S. mega-caps at each of 2000, 2005, 2010 and 2015 and hold each group ten years — <b>only " + str(td_beat_total) + " of the 40 beat QQQ</b>. The 2000 giants (Cisco, Intel, Microsoft…) lost more than the crashing index; the 2005 and 2010 cohorts won 2-of-10 and 1-of-10; even the 2015 cohort — which included Meta, Netflix and Microsoft at the dawn of the mega-cap decade — went only 5-for-10. Published research finds the same 'top dog curse' in every market studied [24]. The index rode the Mag-7 up without you guessing them, and when leadership rotates, it will ride the next set too — automatically. A hand-held Mag-7 basket won't.<div class='chart' style='margin-top:8px'><table><thead><tr><th>Cohort</th><th>Names (first 4)</th><th class='r'>Median 10y</th><th class='r'>QQQ 10y</th><th class='r'>Beat QQQ</th></tr></thead><tbody>" + td_rows + "</tbody></table></div>"),
 ("“Technical analysis works — support, resistance, moving averages, patterns.”",
  "This is the single most-tested claim in finance, with the clearest verdict. Sullivan, Timmermann &amp; White tested <b>7,846 technical trading rules over 100 years</b> of Dow data with statistics that account for trying many rules: the apparent winners had no genuine predictive power out-of-sample, and nothing worked at all on more recent data [19]. A follow-up on <b>historical</b> claims of chart-pattern success found they dissolve once you correct for having searched thousands of rules [20]. Charts describe the past beautifully; a century of testing says they do not predict the future — and our own trend-rule tests (§7: results that flip on the trade date) are the same finding in miniature."),
 ("“Sell in May / the January effect / seasonal patterns.”",
  "Calendar effects were tested the same rigorous way — every seasonal rule anyone had proposed, adjusted for the fact that thousands were searched: <b>none survive</b> [21]. The seasonal patterns you read about are what randomness looks like when you sort it by month, and the famous ones stopped 'working' around the time they were published — the standard fate of published patterns (−58% after publication [10])."),
 ("“QQQ is obviously overvalued right now — I'll wait for a better price.”",
  "This is market timing wearing a valuation costume, and it has a measured record: valuation-based timing signals (like CAPE) have been persistently 'expensive' for most of the last two decades — a timer following them sat out most of the gains. The definitive study is titled, accurately, <i>Market Timing: Sin a Little</i> — even its sympathetic authors found valuation timing adds almost nothing beyond simply staying invested [22]. Our §7 shows the local version: waiting for the obvious discount cost ~30% of final wealth. Valuation tells you expected returns are lower than usual; it does not tell you when, or from what higher level, the repricing comes."),
 ("“I'll buy call options / LEAPS / 0DTE for leveraged upside.”",
  "The complete record of retail options trading is one of the most one-sided in finance: in comprehensive U.S. brokerage data, retail option <i>purchases</i> lose ~4% per trade on average, and the now-dominant same-day-expiry (0DTE) trades do several points worse [25]; an earlier study of a full brokerage's clients concluded most were, in effect, gambling and 'incurred large losses' [26]. Options add three costs stocks don't have — time decay, volatility overpricing, and wide spreads — so even a correct directional view usually loses money on the position. If §3 showed stock-picking is a lottery, bought options are the same lottery with an expiry date."),
 ("“Fine, no stock picking — I'll buy the momentum/value factor ETF instead.”",
  "Factor ETFs are the index-fund version of the strategies in §5, and they inherit the same fate plus fees: published factor premia decay ~58% after publication [10], most factor ETFs launched after the factor's famous decade, and none of the major U.S. factor ETFs (momentum, value, quality, low-vol) has beaten QQQ over its life. Specialized and thematic ETFs are worse — they launch at peak hype and underperform by ~6%/yr over their first five years, losing ~30% risk-adjusted [23]. An ETF wrapper changes the fees, not the arithmetic."),
 ("“I'll use stop-losses so my downside is capped.”",
  "A stop-loss converts a temporary decline into a permanent exit — and §4.3 showed the era's best stocks fell −67% to −87% <i>on the way to</i> their legendary gains; any stop tight enough to 'protect' you guarantees you sell every future winner during its ordinary crashes. On the index itself, stop-and-re-enter rules are just trend-timing, which failed the §7 audits (results that depend on the trade date). The honest way to cap downside is position size decided in advance (11.4), not an automated promise to sell low."),
 ("“My newsletter/analyst has a great track record.”",
  "The oldest result in empirical finance, from 1933: Alfred Cowles collected ~12,000 professional forecasts and asked <i>Can Stock Market Forecasters Forecast?</i> — his abstract answer: 'It is doubtful' [27]. Sixty years later, Graham &amp; Harvey graded 326 investment newsletters over 13 years: <b>no evidence of any timing ability as a group</b>, and the hot ones didn't stay hot [28]. Combine that with §6's luck math (track records arise by chance in the thousands) and the five audits (§5): a track record you're shown is marketing until it survives the audits — and in 90+ years of checking, essentially none have."),
]
FAQS_PEOPLE += [
 ("“People DO make money on options / 0DTE / picking — I've seen the screenshots.”",
  "Both things are true at once, and §6c's first chart is the reconciliation: <b>75% of stocks made money</b> over the decade (a bull market pays almost everyone) while <b>6.2% beat QQQ</b> — and in options, where the average retail trade <i>loses</i> ~4% [25], a large minority of trades still win, and only the wins get screenshotted. 'Made money' is the wrong test: the test is against what the same dollars did in the automatic alternative, and screenshots never show that column. The right tail of any distribution is real, visible, and not a strategy."),
 ("“There must be strategies out there that work. Somewhere. No?”",
  "Yes — and they share three properties: they're <b>capacity-limited</b> (they die if too much money uses them), <b>closed or unbuyable</b> (tolls, seats, infrastructure, control, private information — §6b), and <b>never sold to you</b> (a real edge is worth more traded than sold; what's sold is the story). Everything <i>purchasable</i> — funds, newsletters, courses, signals, ETFs — has a measured record, and it's this study. Even published academic edges decay 58% on publication [10]. The strategy that works and is available to you is the one this paper is about: own the machine and don't interrupt it."),
 ("“How often can research actually find winning stocks? Is the work worth it?”",
  "Measured directly: the base rate of a pick beating QQQ over a year is ~42%; the <i>best</i> signals found in years of systematic research — insider clusters, quality screens, machine learning over 36 feature types — lift that to ~44–48%. Never past the coin flip. So the honest wage of stock research is: <b>hundreds of hours buys ~3–6 percentage points of hit rate, landing you still below 50/50 against the free alternative</b> — before costs and taxes take their share. As work, it pays negatively; as a hobby, price it like one (11.4). The exception that CAN pay is §6c's error-structure requirement: knowledge that isn't in anyone else's model — which comes from your profession, not from research tools everyone owns."),
]
FAQS_PREMISE = [
 ("\u201cIs this just a U.S. / QQQ phenomenon?\u201d",
  "No \u2014 the lottery structure is global. Bessembinder's worldwide follow-up (64,000+ stocks across 42 countries, 1990\u20132018): the top <b>1.3% of firms produced ALL $44.7 trillion of global net wealth creation</b>; outside the U.S., <i>less than 1%</i> of firms did \u2014 and 61% of non-U.S. stocks returned less than U.S. Treasury bills [31]. SPIVA runs the professional scorecard in every region \u2014 Europe, Japan, Australia, India \u2014 with the same shape of result as the U.S. The mechanisms in this study (skew, concentration, the arithmetic of active management) are properties of equity markets, not of one country or one index."),
 ("“Backtests aren't real life.”",
  "Correct — real life is <b>worse</b> for the picker. These simulations already charge trading costs and use only information available at each decision date, but they can't charge you the behavioral tax: real investors panic-sell in crashes and chase after rallies, which studies of actual investor returns put at another 1–2%/yr of loss. Every gap between simulation and reality widens the index's lead."),
 ("“QQQ just got lucky — tech happened to win this era.”",
  "Partly true — §2.2 gives this objection a full four-step answer, and the paper concedes the era-dependence openly (§9): QQQ-DCA is a concentrated bet that suffered −81% in the dot-com crash. But the anti-picking verdict does not depend on QQQ: pickers failed against SPY too, professionals fail against every benchmark (SPIVA), and the skew/concentration math (§3) holds in every equity market ever measured. Choose a broader index if you prefer — just don't hand-pick stocks against it."),
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
  "Different question — and §2.4 now treats each one fully, with the measured record. In short: those are <i>asset-allocation</i> choices — how much risk, of which kinds, you want to hold — and diversification genuinely smooths the ride (at some cost to return). This paper is about one specific claim: that you can select <i>stocks</i> to beat a stock index. Nothing here argues against owning other assets; everything here argues against paying anyone (including yourself) to pick stocks."),
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
.lit{{border-left:4px solid #6b7280;background:#f8fafc}}
.lit .h{{font-size:10.5px;text-transform:uppercase;letter-spacing:1px;color:#6b7280;font-weight:800;margin-bottom:6px}}
.lit p{{font-size:13.5px;margin:6px 0}}
ol.refs{{margin:8px 0 8px 20px;font-size:12.5px;color:#374151}} ol.refs li{{margin:6px 0}}
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
<div class="sub" style="margin-top:8px">Prefer it visual? <a href="verdict-story.html"><b>Read the interactive story edition &rarr;</b></a></div>
</header>

<div class="card verdict" style="margin-top:18px">
<b>The verdict:</b> on 26 years of honest data, <b>no method of picking stocks or timing markets reliably ends up with more money than automatically buying QQQ every two weeks or every month.</b> Not expert selection, not "buying the winners," not value screens, not dip-buying, not machine learning. This isn't because markets are magic. It's arithmetic: a few huge winners produce nearly all stock-market wealth, the index already owns them at full size and rides them automatically, and every act of picking makes you own less of them. The rest of this page shows the evidence, then answers every objection we could find.
</div>

<div class="kpis">
<div class="k"><div class="v">6.2%</div><div class="l">of stocks beat QQQ over the decade</div></div>
<div class="k"><div class="v">2,177</div><div class="l">stocks measured, incl. every death</div></div>
<div class="k"><div class="v">250+</div><div class="l">strategies tested &amp; failed</div></div>
<div class="k"><div class="v">89.5%</div><div class="l">of pro funds lag over 15y (SPIVA)</div></div>
</div>

<div class="card"><nav class="toc">
<a href="#s1">1 · The rules of the test</a>
<a href="#s2">2 · Why QQQ (and not SPY)</a>
<a href="#s3">3 · The market is a lottery with few winning tickets</a>
<a href="#s4">4 · Why buying the winners fails</a>
<a href="#s5">5 · We tried to beat it 250+ ways</a>
<a href="#s6">6 · Luck explains your market-beating friend</a>
<a href="#s6b">6b · Who actually outperforms</a>
<a href="#s6c">6c · The argument in probabilities</a>
<a href="#s7">7 · Market-timing myths</a>
<a href="#s8">8 · "Then I'll take more risk"</a>
<a href="#s9">9 · What this does NOT say</a>
<a href="#s10">10 · The complete strategy map</a>
<a href="#s10b">10b · Every objection, answered</a>
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
<h3>2.1 &nbsp;Why a top-100 index keeps beating the broader 500</h3>
<p>Mechanically, QQQ is a purer expression of the one force this whole paper documents: <b>letting winners run at full weight</b>. Fewer names means the mega-winners are a larger share; size-weighting means their weight grows automatically as they win. No judgment is involved — just less dilution of the lottery tickets that §3 shows produce everything.</p>
<h3>2.2 &nbsp;“But choosing QQQ as the benchmark is itself hindsight — isn't this whole study circular?”</h3>
<p>This is the sharpest objection to this paper, so it gets a full logical answer. Start by separating two claims that are usually blurred together:</p>
<div class="card"><p style="margin:4px 0"><b>Claim A (benchmark-independent):</b> no method of selecting stocks or timing purchases reliably beats an automatic, cap-weighted index — <i>whichever</i> index you choose.</p>
<p style="margin:4px 0"><b>Claim B (benchmark-specific):</b> QQQ has been the better index to automate this era, and choosing it going forward is a deliberate, concentrated bet.</p>
<p style="margin:4px 0">The study's verdict rests entirely on <b>Claim A</b>. Claim B only decides <i>which machine</i> you automate — and §9 states its risk openly.</p></div>
<p><b>Why Claim A cannot be an artifact of picking a hot benchmark — four steps:</b></p>
<ul>
<li><b>1. Swap the benchmark; the verdict survives.</b> If "nothing beats QQQ" were true only because QQQ ran hot, pickers would beat a cooler benchmark. They don't: the same 100 random 10-stock portfolios that ended at a median {fvs.get("vs_qqq_med","0.58")}× of QQQ-DCA <i>also lost to SPY-DCA</i> — median {fvs.get("vs_spy_med","0.84")}×, with two-thirds trailing the broader index too. And the professional scorecard (SPIVA) doesn't use QQQ at all: it measures every fund against <i>its own</i> category benchmark — value funds vs value indexes, small-cap funds vs small-cap indexes — and finds ~90% failure everywhere over 15 years [12]. A conclusion that survives substituting the benchmark is, by definition, not benchmark selection bias; the choice of QQQ changes the <i>size</i> of the picker's shortfall, not its existence.</li>
<li><b>2. The core mechanism is an identity, not a sample.</b> For <i>any</i> index over <i>any</i> universe, the average actively-managed dollar in that universe must equal the index before costs and trail it after costs (Sharpe's arithmetic [7]) — true in 1929, true in 2000, true if technology collapses tomorrow. And the skew that makes <i>concentrated</i> picking worse than average is measured across a century of U.S. data [1] and 42 countries [31]. Neither premise mentions QQQ.</li>
<li><b>3. Even perfect benchmark hindsight doesn't rescue picking.</b> Grant the picker our exact "bias": tell them in 2015 that U.S. mega-cap tech will win. Picking <i>inside</i> the winning pond still lost — the era's dominant giants beat QQQ only 11/40 times across four decade-cohorts; momentum strategies restricted to the index's own members died in audit; the tech-theme funds trailed (§11.5). The benchmark's era explains the benchmark's return; it does not create the picker's shortfall. The shortfall comes from skew + costs, which follow the picker into any pond.</li>
<li><b>4. The rule predates the result.</b> QQQ is not a basket assembled after seeing what won — the Nasdaq-100 rule dates to 1985 and the ETF to 1999, before every window measured here. Contrast the actual hindsight construction people confuse it with: "the Magnificent 7" is a list <i>defined by</i> past winning, and §10b shows what buying such lists ex-ante actually did. This study never benchmarks against anything defined by the outcome.</li>
</ul>
<h3>2.3 &nbsp;The asymmetry that settles it: one bounded choice vs a repeated unbounded game</h3>
<p>Finally, the logical difference between choosing an index and choosing stocks — the reason the first decision is safe to make imperfectly and the second isn't:</p>
<div class="chart">{c_regret}</div>
<div class="leg"><span>Worst realistic outcome of each decision, as measured in this study (relative to the alternative, horizons as labeled).</span></div>
<p>QQQ and SPY are two <b>overlapping winner-riding machines</b>: SPY also held Apple, Microsoft and Nvidia — at somewhat smaller weights. Choose the "wrong" one and you still capture the winning tail automatically; the maximum regret over 26 years was a factor of ~2, while still multiplying your money ~8×. It is <b>one decision between two self-correcting rules, with bounded regret</b>. Stock selection is the opposite object: a <b>repeated</b> game (hundreds of decisions, each with measured sub-50% odds, costs compounding) with <b>unbounded</b> regret — the median pick captures a quarter of the index's decade and the tail outcome is zero. Choosing between trains headed the same direction is not the same kind of decision as betting you can outrun them. That is why this paper agonizes over stock-picking and treats the QQQ-vs-SPY choice as a one-paragraph preference.</p>
<h3>2.4 &nbsp;Why not international (EAFE, Japan), total-market, a sector fund, or gold?</h3>
<p>The same question, one ring further out. The record first — identical $1,000/month into each machine, 2005–2026 (${mach["contrib"]/1e3:,.0f}k contributed):</p>
<div class="chart">{c_machines}</div>
<div class="leg">
<span><i style="background:#111418"></i><b>QQQ ${_ms["QQQ"]["final"]/1e6:.2f}M</b> ({_ms["QQQ"]["dd"]}%)</span>
<span><i style="background:#374151"></i>SPY ${_ms["SPY"]["final"]/1e6:.2f}M ({_ms["SPY"]["dd"]}%)</span>
<span><i style="background:#b0891b"></i>Gold ${_ms["GLD"]["final"]/1e3:,.0f}k ({_ms["GLD"]["dd"]}%)</span>
<span><i style="background:#9ca3af"></i>Intl-developed ${_ms["EFA"]["final"]/1e3:,.0f}k ({_ms["EFA"]["dd"]}%)</span>
<span><i style="background:#6b7280"></i>Japan ${_ms["EWJ"]["final"]/1e3:,.0f}k ({_ms["EWJ"]["dd"]}%)</span>
<span><i style="background:#d1d5db"></i>Emerging ${_ms["EEM"]["final"]/1e3:,.0f}k ({_ms["EEM"]["dd"]}%)</span>
<span>(worst account fall in parentheses; log scale)</span></div>
<p>Now the logic, alternative by alternative — because the record alone would be recency, and §2.2's rules apply to us too:</p>
<ul>
<li><b>Total-market funds (VTI-style):</b> a cap-weighted total-market fund is ~85% the S&amp;P 500 by weight — the small-cap tail it adds behaved like the ponds in §5 (an <i>oracle</i> picking small-caps perfectly still lost). It is the same machine, slightly diluted; choose it or SPY interchangeably. Nothing in this study changes.</li>
<li><b>International (EAFE, emerging):</b> two facts and one concession. Fact one: the two-decade record above — roughly a quarter of QQQ's outcome at equal or worse drawdowns. Fact two, the structural one: <b>you already own the world through U.S. listings</b> — the S&amp;P 500 earns roughly 40% of its revenue abroad, and the world's dominant companies overwhelmingly choose U.S. listings; meanwhile the no-view global index (VT-style) is itself ~60–65% U.S. by cap-weight, so "maximum humility" moves your weights less than it sounds. The concession: Claim A holds <i>within</i> every market (the skew is global [31]) — if you genuinely prefer world weights, automate a global cap-weighted fund and every conclusion here still applies. What's indefensible isn't the region — it's picking and timing inside it.</li>
<li><b>Japan — the strongest warning in market history, faced directly:</b> the Nikkei's 1989 peak took about <b>three decades</b> to reclaim. That is what a single-country machine bought at bubble prices can do, and no honest study waves it away. Three replies, not one: (i) that catastrophe was a <i>lump sum at the top of one market</i> — a steady contributor kept buying Japan's bottoms for decades and recovered years earlier (the identical mechanism §11.2 shows on QQQ's own −81%); (ii) it is the case <i>for</i> broad, multi-market cap-weighting if you fear it — not for stock-picking, which §2.2 showed fails within Japan too; (iii) today's QQQ concentration is the closest modern rhyme to 1989 Tokyo — <b>which is exactly why §9 exists and why Claim B is labeled a bet, not a law.</b></li>
<li><b>A sector fund as the core:</b> the cap-weighted index <i>already is</i> a sector rotator — over 50 years its internal weights migrated from industrials and energy to technology automatically, with what looks like perfect hindsight because it requires no foresight. Committing to one sector permanently is a pond bet plus rotation risk: §11.5 measured the fate of the reigning #1 sector — <b>below the sector median 59% of the time the following year</b>. And if your sector conviction is specifically "technology keeps leading" — that conviction <i>is</i> QQQ, expressed with 100 companies of internal diversification instead of 25.</li>
<li><b>Small caps (Russell-2000-style):</b> the oldest "better pond" belief — small companies grew into the size premium of the old textbooks. The modern record: dead last among the diversified choices below (${menu["rows"][-2]["final"]/1e3:,.0f}k–${[r for r in menu["rows"] if r["t"]=="IWM"][0]["final"]/1e3:,.0f}k range for small caps this window), and the structural reason is decisive: <b>a small-cap index is a machine that sells its winners</b> — the moment a company succeeds, it graduates out of the small-cap index into the large-cap one. It is the exact inverse of the winner-riding machine this paper is about, holding the losers indefinitely and surrendering every Nvidia the day it becomes one. Add the published record — the size premium is among the anomalies that decayed after publication [10] — and §5's oracle result (perfect small-cap picking still lost 3 of 5 eras), and the pond is triple-condemned: structurally, empirically, and even under perfect foresight.</li>
<li><b>Gold:</b> honesty first — over this exact window gold out-compounded every international equity fund above (a strong decade at each end), and it crashes differently than stocks ({_ms["GLD"]["dd"]}% worst here). But it is not a compounding machine and cannot be one: <b>gold has no earnings, no cash flows, and no growth engine</b> — every dollar of its rise is repricing, not production, and its multi-century real return is near zero. It captured 41% of QQQ's result in its own good era; over horizons where compounding dominates, the gap widens without bound. As crisis insurance it's a separate, legitimate question this paper doesn't cover; as the <i>engine</i>, holding gold is conceding the engine.</li>
</ul>
<p class="note">The pattern across all five: every alternative is either the same machine in a different wrapper (total-market, global cap-weight), a bet this study already prices (sector = pond + rotation), or a different asset class doing a different job (gold). None of them changes the verdict on picking and timing — and the one that's genuinely defensible on humility grounds (global cap-weighting) is defensible precisely because it, too, is an automatic winner-riding rule.</p>
<h3>2.5 &nbsp;“Fine — so what IS the optimal thing to DCA into?” (answered without a thumb on the scale)</h3>
<p>The full menu, measured — every U.S. sector fund, small caps, and the two broad machines, identical $1,000/month over the same 21 years (${menu["contrib"]/1e3:,.0f}k in):</p>
<div class="chart">{c_menu}</div>
<div class="leg"><span><i style="background:#111418"></i>broad cap-weighted machines&nbsp;&nbsp;<i style="background:#b91c1c"></i>single slices (sectors, small caps) · {menu["window"]}, dividends reinvested</span></div>
<p>Read it honestly, in both directions:</p>
<ul>
<li><b>Yes, one slice beat QQQ:</b> the tech sector fund finished first — of course it did; this window is the tech era, and <i>the winning slice of any era beats the index that merely contains it</i>. The chart itself contains the selection trap it warns about.</li>
<li><b>The same fund, the adjacent decade:</b> DCA into that same tech fund 1999–2009 returned <b>${ef["XLK"]["final"]/1e3:,.0f}k on ${ef["XLK"]["contrib"]/1e3:,.0f}k contributed</b> — eleven years to barely break even, −{-ef["XLK"]["dd"]}% along the way — while the era's <i>actual</i> winning sector was <b>energy (${ef["XLE"]["final"]/1e3:,.0f}k)</b>… which finished <i>dead last</i> in the window above. Sector leadership didn't just fade; it <b>fully inverted</b>. Naming the next era's winning slice in advance is the same forecasting problem as naming the next Nvidia — §11.5 measured it (the reigning #1 sector ends below the median 59% of the time).</li>
<li><b>Also true and worth saying:</b> the defensive slices (healthcare −18%, staples −17%) delivered far gentler rides for their smaller outcomes. Someone who genuinely optimizes for shallow drawdowns is making a coherent choice — a <i>risk</i> choice, not a return-forecasting one.</li>
</ul>
<p><b>The unbiased answer, then, is not a ticker — it's a dial.</b> Every machine sits on a concentration spectrum:</p>
<div class="card"><p style="margin:4px 0;text-align:center"><b>global cap-weight → total-US / SPY → QQQ → single sector → single stocks</b></p>
<p style="margin:6px 0">Each step rightward means: a bigger payoff <i>if</i> the next era favors your slice, a worse outcome if it doesn't, and larger swings either way. What this study's data determines — and what it cannot:</p>
<p style="margin:6px 0"><b>It cannot name the ex-ante optimal point.</b> That would require knowing the next era's leader, and the persistence and era-flip evidence above measures that as unknowable — anyone who claims otherwise is making §11.5's bet with §6c's odds.</p>
<p style="margin:6px 0"><b>It does determine three things.</b> (1) Every point on the <i>left three positions</i> of the dial (the broad machines) beat every tested picking and timing strategy — that's Claim A, and it's the only part that isn't a bet. (2) The rightmost position (single stocks) is measurably negative (§3–§6). (3) Moving right of the broad machines is a <i>labeled era bet</i> — legitimate to make, illegitimate to mistake for skill.</p>
<p style="margin:6px 0"><b>So the operational answer:</b> the no-forecast default is the <i>broadest</i> machine you'll actually automate (total-US or global cap-weight); QQQ is the defensible-but-concentrated bet this paper itself uses as its bar while flagging its −81% scar (§9); a single sector is that same bet with less internal diversification and measured rotation risk — including the tech fund's own lost decade above. Wherever you land: <b>the choice among broad machines matters less than never leaving the dial for picking, and never abandoning the plan mid-drawdown</b> — those two errors are the measured ones.</p></div>
<h2 id="s3">3 · The market is a lottery with a few winning tickets — and the index holds them all</h2>
<p>Here is every investable U.S. stock at mid-2016 — all {sk['n']:,} of them, including the {sk['died']} that later died — and what each returned over the following decade:</p>
<div class="chart">{c_skew}</div>
<div class="leg"><span><i style="background:#b91c1c"></i>lost money</span><span><i style="background:#15803d"></i>made money</span><span>each bar = number of stocks; dashed lines = QQQ (+{sk['qqq']*100:.0f}%) and the median stock</span></div>
<div class="card big"><div class="n">6.2%</div>That's the fraction of stocks that beat QQQ over that decade — roughly <b>1 in 16</b>. A quarter lost money outright in one of the best decades markets have ever had. The <i>median</i> stock returned +{sk['median']*100:.0f}% in total while QQQ returned +{sk['qqq']*100:.0f}%.</div>
<p>And the gains that do exist are brutally concentrated: the top 1% of stocks produced <b>~{conc['share_of_net'][3]:.0f}%</b> of all net gains; the top 10% produced <b>~{conc['share_of_net'][6]:.0f}%</b>. This is the same picture the canonical academic result paints (Bessembinder [1]) — a small sliver of companies accounts for most of the stock market's entire net wealth creation since 1926, and <i>most individual stocks underperform cash</i>.</p>
<div class="card lit"><div class="h">The published record says the same thing</div>
<p><b>Bessembinder (2018)</b>, studying every U.S. stock since 1926 (~26,000 companies): the best-performing <b>4% of companies account for the stock market's entire net gain over Treasury bills</b>; the other 96% collectively matched cash. Four out of seven stocks returned <i>less than Treasury bills</i> over their lifetime, and just five companies produced 10% of all wealth ever created [1]. His global follow-up found it's even more extreme worldwide.</p>
<p><b>J.P. Morgan's “Agony &amp; Ecstasy” study</b> (Russell 3000, 1980–2014): <b>40% of all stocks suffered a catastrophic decline of 70%+ from which they never recovered</b>; two-thirds underperformed the index; the median stock lagged it by −54% over its lifetime [2].</p>
<p><b>Heaton, Polson &amp; Witte, “Why Indexing Works” (2017)</b> formalized the mechanism this section shows empirically: when a few stocks drive everything, <i>any</i> subset you pick most likely misses them, so most pickers must trail the index — before costs [3].</p></div>
<div class="chart">{c_conc}</div>
<div class="leg"><span>Cumulative share of ALL net gains (2016–2026) produced by the top N stocks, log scale. Ten stocks — 0.4% of the market — produced 10.7% of everything; 250 stocks produced 61%.</span></div>
<p>It's not one lucky decade. Year by year since 2000, the fraction of stocks beating QQQ over the following 12 months averaged <b>42%</b> — and the years it exceeded 50% are mostly 2000–2001, when "beating QQQ" meant falling less than an index that was crashing:</p>
<div class="chart">{c_year}</div>
<div class="leg"><span>% of stocks beating QQQ over the next 12 months, each year 2000–2025 (dashed line = coin flip)</span></div>
<p><b>This is the whole game.</b> Picking stocks means trading away guaranteed full-size ownership of the few tickets that pay for a handful of tickets that each probably won't. The index isn't smart — it just refuses to hand any winning ticket back.</p>

<h2 id="s4">4 · Why “buying the winners” specifically fails</h2>
<p>The most seductive idea in investing: <i>just buy what's already winning.</i> Three measured facts kill it.</p>
<h3>4.1 The winners list doesn't stay the winners list</h3>
<div class="card big"><div class="n">15%</div>Of each year's top-10% performers, that's how many repeat in the top 10% the following year. <b>{pers['below_median']:.0f}%</b> fall below the <i>median</i> stock; only <b>{pers['beat_qqq_next']:.0f}%</b> beat QQQ. (Measured across 24 annual cohorts, 2001–2024.)</div>
<div class="card lit"><div class="h">The published record</div>
<p>The same non-persistence holds for professionals: <b>S&amp;P's Persistence Scorecard</b> finds that of funds in the top quartile in any year, almost none remain top-quartile a few years later — at rates at or below what chance predicts [4]. Momentum itself is a real, Nobel-adjacent finding (<b>Jegadeesh &amp; Titman 1993</b> [5]) — but it's a short-horizon <i>relative</i> effect that decays and periodically crashes (<b>Daniel &amp; Moskowitz 2016</b> [6]); it is not "good stocks keep being good."</p></div>
<div class="chart">{c_fate}</div>
<div class="leg"><span>What happens to a top-10% winner the following year (24 annual cohorts, 2001–2024).</span></div>
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
<div class="chart">{c_shape}</div>
<div class="leg"><span>How often top picks land in the NEXT year's top 10% of stocks. The measured requirement for beating QQQ isn't more intelligence — it's <b>independent errors</b>, i.e., information not in everyone else's model.</span></div>
<div class="card retract"><b>We retracted our own winners.</b> Three strategies from this research were themselves published as market-beaters, then independently rebuilt and audited: a machine-learning stock picker (two data flaws found — honest version lands <i>below</i> QQQ), a biweekly DCA stock-selection system (half its edge was survivorship bias and recency; honest version is a coin flip), and a leveraged-ETF timing system (data leakage; rebuilt honestly, its return came from extra risk, not skill). If we hold everyone else's claims to these audits, we hold our own to them too — <b>that is why you can trust the negative result.</b>
<div class="chart" style="margin-top:10px">{c_audit}</div>
<div class="leg"><span>Share of each claimed outperformance that survived independent audit. The bars are the point.</span></div></div>

<div class="card lit"><div class="h">The published record</div>
<p><b>Sharpe's "Arithmetic of Active Management" (1991)</b> is the iron law underneath all of this: the average actively-managed dollar must earn the market return <i>before</i> costs, and less than it <i>after</i> costs — always, by accounting identity. Beating the index is a zero-sum game played against professionals, minus fees [7].</p>
<p><b>Fama &amp; French, "Luck versus Skill" (2010)</b>: across 3,000+ mutual funds, so few beat their benchmarks net of costs that the winners are statistically indistinguishable from what luck alone would produce [8]. <b>Carhart (1997)</b> found the same decades earlier: no persistent fund skill beyond fees and momentum exposure [9].</p>
<p><b>McLean &amp; Pontiff (2016)</b>: even <i>published academic anomalies</i> lose ~26% of their returns after their sample period ends and <b>~58% after publication</b> — edges die on contact with daylight [10]. <b>Harvey, Liu &amp; Zhu (2016)</b> showed most published factors likely aren't real at all once you correct for how many were tested — the same multiple-testing trap our audits caught [11].</p></div>
<h2 id="s6">6 · About your market-beating friend: the luck math</h2>
<p>A concentrated stock portfolio beats QQQ in a given year roughly 40–45% of the time (that's the measured base rate — losing more often than winning, but not by much in any single year). Now imagine 10,000 people each running one portfolio, with <b>zero skill</b>, just those coin odds:</p>
<div class="chart">{c_luck}</div>
<div class="leg"><span>Expected number (out of 10,000 zero-skill pickers) still holding a perfect beat-the-market streak of each length</span></div>
<p>After five years, ~185 flawless five-year track records exist by pure chance. After ten years — three. Those people are not lying about their returns; they're real, they're confident, and they're indistinguishable from skilled until the streak ends. They post on YouTube; the 9,815 others don't. <b>Survivorship bias isn't just a data problem — it's your entire social-media feed.</b></p>

<div class="card lit"><div class="h">The published record</div>
<p><b>SPIVA (S&amp;P, year-end 2024)</b>: <b>89.5% of professional U.S. large-cap funds underperformed the S&amp;P 500 over 15 years</b> — full-time managers, with research teams, before your tax disadvantages [12]. <b>Buffett's famous 10-year bet</b> (2008–2017): a plain S&amp;P 500 index fund returned +125.8%; the hand-picked hedge-fund portfolios averaged roughly +36% — he donated the winnings and repeated the advice: index [13].</p>
<p>And for the individual actually doing the trading: <b>Barber &amp; Odean (2000)</b>, 66,465 real brokerage households — the most active traders earned <b>11.4%/yr while the market returned 17.9%</b> [14]. <b>Morningstar's "Mind the Gap"</b>: real investors earn ~<b>1.1 percentage points per year less than the very funds they hold</b>, from timing their entries and exits [15]. A study of every Brazilian who began day-trading index futures (2013–2015): of those who persisted 300+ days, <b>97% lost money</b>, and only 1.1% earned more than minimum wage [16].</p></div>
<h2 id="s6b">6b · “So who actually DOES outperform — and how?”</h2>
<p>Markets aren't unbeatable — they're unbeatable <i>from where you're sitting</i>. The honest census of who wins, and the mechanism each uses:</p>
<ul>
<li><b>Market-makers and high-frequency firms</b> — they don't predict anything; they <i>sell liquidity</i>, collecting a fraction of a cent on billions of trades (including yours). It's a toll booth, not a forecast, and it requires infrastructure you cannot rent.</li>
<li><b>A handful of closed quantitative funds</b> (the famous one returns ~66%/yr gross) — thousands of tiny, fast statistical edges, capacity-capped at a few billion dollars, <b>closed to outside money for decades</b>. The same firm's funds that outsiders <i>can</i> buy have performed near the market. Capacity is the tell: real edges are small; anything sold to unlimited money isn't one.</li>
<li><b>Activists and private equity</b> — they buy control and <i>change the company</i>. The return comes from doing, not picking.</li>
<li><b>Corporate insiders</b> — the one real information edge, which is why trading on it is illegal (and why the legal shadow of it — cluster buying — was the best signal we measured).</li>
<li><b>Specialists in tiny, uncrowded niches</b> — genuinely possible, capacity-limited, and a full-time job with a measured failure rate, not a strategy you buy.</li>
<li><b>The lucky</b> — §6. By far the largest group, and indistinguishable from the skilled for about a decade (see below).</li>
</ul>
<h3>“Then how do hedge funds succeed?”</h3>
<p>Mostly, <b>the manager succeeds; the investor doesn't.</b> The measured record:</p>
<div class="chart">{c_hedge}</div>
<div class="leg"><span>Warren Buffett's public 10-year bet: S&amp;P 500 index fund vs five hand-picked portfolios of hedge funds, 2008–2017 [13]</span></div>
<ul>
<li><b>Corrected for reporting tricks, aggregate hedge-fund returns roughly match the index.</b> Hedge-fund databases are self-reported: funds start reporting <i>after</i> a hot streak (backfill) and stop when they die (survivorship). Correcting both removes ~5–6 points of reported return per year [29]; corrected aggregate returns came out at 9.29%/yr vs the S&amp;P's 9.38% over the classic study window [29].</li>
<li><b>The fee arithmetic is the business model:</b> one analysis of the industry's whole history found investors collectively earned <i>less than Treasury bills</i>, while managers collected hundreds of billions in fees — the classic "2-and-20" is paid on assets, win or lose [30].</li>
<li><b>What institutions actually buy</b> from the good funds isn't index-beating — it's returns <i>uncorrelated</i> with their stock portfolios (a diversification service). Judged against QQQ-DCA, that's a different product, not a refutation.</li>
<li>And the few genuinely great funds? See the census above: closed, capacity-capped, infrastructure businesses. <b>The market-beating that exists is precisely the kind you cannot buy.</b></li>
</ul>

<h2 id="s6c">6c · The whole argument in probabilities</h2>
<p>Strip away every story and the entire debate reduces to a few measured numbers. First, the one that explains every anecdote you've ever heard:</p>
<div class="chart">{c_profitbeat}</div>
<div class="leg"><span>All 2,177 investable stocks, 2016–2026, deaths included. <b>“People make money picking stocks” and “almost nobody beats QQQ” are both true.</b></span></div>
<p>Three out of four picks <i>made money</i> — a rising market pays nearly everyone, which is why every picker you know has winners to tell you about. Beating the automatic machine is a different event:</p>
<div class="chart">{c_ladder}</div>
<div class="leg"><span>The probability ladder, all point-in-time measurements from this study. Each step down is the same game, held longer.</span></div>
<p>Notice the shape: the odds <i>fall</i> as the horizon grows — time compounds the index's advantage, not the picker's. "In the long run it'll come back" is backwards: the long run is where picks go to lose.</p>
<h3>And here is why the debate never dies: skill takes decades to even detect</h3>
<div class="chart">{c_skillyears}</div>
<div class="leg"><span>Years of live results needed before performance is statistically distinguishable from luck (standard two-sigma test; IR = skill ratio). A "good" manager needs ~16 years — and the market a manager proved skill in no longer exists by the time it's proven.</span></div>
<p>This is the deepest reason the industry survives its own scorecard: <b>at realistic skill levels, one investing lifetime is not long enough to tell a skilled picker from a lucky one</b> — so belief fills the gap, marketing sells the belief, and the base rates quietly collect.</p>
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
<div class="chart">{c_rel}</div>
<div class="leg"><span>Growth of QQQ ÷ growth of SPY since 1999 (above 1× = QQQ ahead). Leadership <b>rotates</b>: QQQ spent roughly a decade below the line before its era — the next rotation is §9's whole point.</span></div>
<ul>
<li><b>It does not say markets are perfectly efficient</b> — only that the specific game of out-picking a winner-riding index using public information is stacked, measurably, against the picker.</li>
<li><b>It does not cover other asset classes.</b> This is a study of stock strategies measured against QQQ; whether you hold anything besides stocks is a separate question it deliberately doesn't answer.</li>
<li><b>It does not promise the next 26 years look like the last.</b> It says: whatever index you choose, no tested method of picking stocks against it has honestly beaten contributing to it on autopilot.</li>
</ul>

<h2 id="s10">10 · The complete map — every strategy family, where it's addressed, and the verdict</h2>
<p>A defensible "nothing works" claim must be exhaustive. Here is every family of stock-market strategy we know of, where this study addresses it, and what the evidence says:</p>
<table><thead><tr><th>Strategy family</th><th>Where</th><th>Verdict vs QQQ-DCA</th></tr></thead><tbody>
<tr><td>Stock picking — fundamental / screens</td><td>§3, §5</td><td class="bad">Loses (0.26–1.21×; best = luck-level)</td></tr>
<tr><td>Stock picking — machine learning / AI</td><td>§5, FAQ</td><td class="bad">Loses (skill real, wrong shape)</td></tr>
<tr><td>Buy the winners / momentum stocks</td><td>§4</td><td class="bad">Loses (15% repeat; hot list $53k vs $160k)</td></tr>
<tr><td>Buy today's giants (Mag-7 style)</td><td>FAQ</td><td class="bad">Loses (11/40 decade-beats across 4 cohorts)</td></tr>
<tr><td>Value / buy cheap stocks</td><td>§5, FAQ</td><td class="bad">Loses (0.60×; winners never looked cheap)</td></tr>
<tr><td>Dividend / income stocks</td><td>FAQ</td><td class="bad">Loses (total-return lag + taxes)</td></tr>
<tr><td>Small caps / other ponds</td><td>§5, FAQ</td><td class="bad">Loses (perfect-foresight oracle loses 3/5 eras)</td></tr>
<tr><td>IPOs / get in early</td><td>FAQ</td><td class="bad">Loses (6,599 IPOs: 8.1%/yr honest)</td></tr>
<tr><td>Insider-filing signals</td><td>FAQ</td><td class="bad">Best signal tested; still &lt; coin flip</td></tr>
<tr><td>Copy trading (Congress, whales, gurus)</td><td>FAQ</td><td class="bad">Loses (45-day-stale information)</td></tr>
<tr><td>Technical analysis / charts</td><td>FAQ</td><td class="bad">Loses (7,846 rules, no OOS value [19])</td></tr>
<tr><td>Seasonality (Sell-in-May etc.)</td><td>FAQ</td><td class="bad">Loses (none survive data-snooping [21])</td></tr>
<tr><td>Market timing — trend/regime switches</td><td>§7</td><td class="bad">Loses (trade-day lottery 0.74–3.31×)</td></tr>
<tr><td>Market timing — wait for the dip</td><td>§7</td><td class="bad">Loses (−30% of final wealth)</td></tr>
<tr><td>Market timing — valuation (CAPE)</td><td>FAQ</td><td class="bad">Loses (“Sin a Little” [22])</td></tr>
<tr><td>Stop-losses / downside rules</td><td>FAQ</td><td class="bad">Loses (sells every future winner in drawdown)</td></tr>
<tr><td>Sector / ETF rotation</td><td>§5</td><td class="bad">Loses (0.20–0.90×)</td></tr>
<tr><td>Factor &amp; thematic ETFs</td><td>FAQ</td><td class="bad">Loses (−6%/yr post-launch for thematic [23])</td></tr>
<tr><td>Options — income (covered calls)</td><td>FAQ</td><td class="bad">Loses (sells the only tail that pays)</td></tr>
<tr><td>Options — buying calls / 0DTE</td><td>FAQ</td><td class="bad">Loses (−4%/trade retail average [25])</td></tr>
<tr><td>Day trading</td><td>FAQ</td><td class="bad">Loses (97% of persisters lose [16])</td></tr>
<tr><td>Leverage (3× funds, margin)</td><td>§8</td><td>More money, more risk — not skill (−84%, dot-com −99.9%)</td></tr>
<tr><td>Professional fund managers</td><td>§6</td><td class="bad">89.5% lag over 15y [12]</td></tr>
<tr><td>Automatic QQQ-DCA (the benchmark)</td><td>all</td><td class="good">The thing nothing beat</td></tr>
</tbody></table>
<p class="note">If a strategy family you can name is missing from this table, the authors consider that a defect — it belongs in the next revision.</p>

<h2 id="s10b">10b · Every objection we could find, answered</h2>
{faq_html}

<h2 id="s11">11 · The playbook: exactly what to do</h2>
<p>Evidence without instructions is trivia. Here is the complete, quantified playbook this research supports — including honest answers to "how much?", "what about picking a few stocks anyway?", and "when do I sell?".</p>

<h3>11.1 &nbsp;First, the order of operations (before any investing)</h3>
<ul class="check">
<li><b>Cash buffer first:</b> 3–6 months of expenses. Its job is to make sure you are never forced to sell stocks in a crash.</li>
<li><b>Free money second:</b> any employer retirement match, always, fully.</li>
<li><b>Tax-advantaged accounts before taxable</b> (401k/IRA equivalents): the same QQQ-DCA compounds meaningfully faster untaxed.</li>
<li><b>Only money you won't need for 10+ years goes into equities.</b> The index fell −81% once (§9) and −32% on a DCA account twice in 20 years. Money you'll need within ~5 years doesn't belong in it — being forced to sell into a crash is the one unrecoverable mistake.</li>
</ul>

<h3>11.2 &nbsp;How much goes into QQQ-DCA? All of your long-term money — if you pass the stomach test</h3>
<p>This study's answer is simple: <b>every dollar you are investing for 10+ years goes into the automatic QQQ purchase.</b> Not because QQQ can't fall — but because every tested attempt to improve on it (picking, timing, rotating, waiting) ended with less money. There is no clever remainder to allocate; <b>the schedule IS the strategy.</b></p>
<p>The honest price of that answer, quantified, so you can pre-commit with open eyes:</p>
<ul>
<li>A steady QQQ-DCA account fell <b>−32% from its peak twice</b> in 20 years (2008–09, 2022) — and kept buying. Both times the automatic contributions bought the bottom, and the account went on to new highs. The people who lost were the ones who stopped.</li>
<li>The worst case on record is worse: a lump sum at the March-2000 top fell <b>−81%</b> and took ~15 years to break even (§9). A <i>contributor</i> through that same disaster recovered far sooner and ended far ahead — steady buying through the crash is precisely what repairs it (§2's table starts at that worst possible moment and still ends at 16× the money in).</li>
<li>So the sizing question is not a percentage — it's this <b>stomach test</b>: <i>when</i> (not if) the account shows −30%, will you keep the automation on? If yes: all long-term money, full weight. If honestly no: automate the largest amount for which the answer is yes — a plan you hold at the bottom beats a bigger plan you abandon there (the measured cost of self-inflicted timing is 1–2 points a year [15], and far worse when it happens inside a crash).</li>
<li>Money needed within ~5 years stays out entirely (11.1). Five-to-ten-year money: partial, scaled to how certain the need is.</li>
</ul>
<div class="chart">{c_ddhist}</div>
<div class="leg"><span>The stomach test, drawn: a QQQ-DCA account's distance below its own peak, 2000–2026. Every red valley was a moment the plan felt broken; every one repaired by continued buying.</span></div>
<p><b>Should you "always and only" do this?</b> Always — automate it and never override it; every override pathway was measured above and lost. Only — yes, for stock-market money: this entire study is the evidence that adding picking, timing, or rotation on top subtracts. The one sanctioned exception is the capped hobby slice in 11.4.</p>
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
<li><b>Selling rules, decided in advance:</b> the QQQ core is never sold (that's the whole edge). A satellite <i>winner</i> is left alone until it outgrows your cap — then trimmed <b>into the core</b> (winner → index-of-winners; never into your losers). A satellite <i>loser</i> needs no decision: it simply never gets another dollar, and it dies or lives on its own. This preserves the let-winners-run principle at every level while capping single-stock risk. (Deciding in advance matters because instinct is measurably backwards: real investors sell their winners ~50% more readily than their losers \u2014 the \u201cdisposition effect\u201d [32] \u2014 the exact opposite of what \u00a73 rewards.)</li>
<li><b>Measure it honestly once a year</b> against what the same dollars in QQQ did. The evidence says it will lag; when it does, you'll have paid a known, capped price for the fun — and when a pick 10×'s, you'll enjoy it without having bet the plan on it.</li>
</ul>

<h3>11.5 &nbsp;“I still want to find the next Nvidia / the next big sector.” The honest protocol</h3>
<p>No argument will stop this urge, so here is the evidence-based way to pursue it — with the odds on the table and the plan protected. First, what hunting the future actually looks like, measured:</p>
<p><b>Chasing the winning sector.</b> Across 27 years of the nine U.S. sector funds, last year's #1 sector:</p>
<div class="chart">{c_sector}</div>
<div class="leg"><span>What happens to the previous year's best-performing sector, 1999–2026. The winner's most likely fate is <b>below-average</b>.</span></div>
<p><b>Buying the future's theme.</b> The four defining theme funds of the era, versus QQQ over the same period:</p>
<div class="chart">{c_themes}</div>
<div class="leg"><span><i style="background:#b91c1c"></i>theme fund&nbsp;&nbsp;<i style="background:#111418"></i>QQQ, same period. </span></div>
<p>Study that chart carefully, because it contains the whole lesson:</p>
<ul>
<li><b>Solar (TAN):</b> the theme was <i>completely right</i> — solar deployment grew roughly a hundred-fold — and the fund still <b>lost money over 18 years</b> while QQQ made 28×. Being right about the future is not the same trade as making money, because the future was already in the price, competition destroyed the margins, and the winners were companies that didn't exist yet.</li>
<li><b>Innovation (ARKK):</b> the era's most famous future-picking fund: half of QQQ's return with a −75% crash.</li>
<li><b>Biotech (XBI):</b> the genomics revolution happened; the fund made half the index.</li>
<li><b>Semiconductors (SMH):</b> the one theme that won — and notice <i>which</i> one: not the exciting new story but the <b>boring incumbent toolmakers that every theme has to buy from</b> (and you had to hold through −52%). That's the only theme pattern with a winning record: picks-and-shovels incumbents, not frontier stories.</li>
</ul>
<p><b>If you're going to hunt anyway, the protocol</b> (each rule maps to a measured failure above):</p>
<ul>
<li><b>Keep it inside the 5–10% satellite</b> (11.4) — sized by the Kelly logic for a negative-expectancy bet: the measured edge of picking is negative, so the "optimal" aggressive size is zero and anything you allocate is priced as entertainment/tuition. The cap converts a plan-killer into a hobby with a known cost (−3% to −6% of lifetime wealth).</li>
<li><b>Hunt where the required edge could actually exist</b> (§6c): your own professional domain — the industry you genuinely know better than analysts covering 40 companies. That is the only <i>legal</i> channel for the uncrowded, company-specific information the math requires. If your idea comes from a feed, a ranking, or a theme ETF launch, it is by definition crowded — the measured odds of those were 5–17%.</li>
<li><b>Prefer the boring implementation of the trend</b> — the SMH pattern: incumbent suppliers with profits today, not stories with promises. Wait out the hype phase: theme funds launch at peak excitement and shed ~6%/yr for five years after [23]; the survivors are still there after the crash.</li>
<li><b>Buy 8–10 tickets, equal-sized, and hold five-plus years.</b> Here's the striking arithmetic: with the measured 17% chance each pick 10×'s over two decades, holding 8 tickets gives you a <b>{p_any10x_8*100:.0f}% chance of owning at least one 10-bagger</b> — and still only a ~6% chance the basket beats QQQ. Both at once. You will very likely get a trophy <i>and</i> lose the race — which is exactly why everyone knows someone with a great pick and almost no one who beat the index. Decide in advance which you're playing for.</li>
<li><b>Expectations, measured:</b> a 90/10 core-satellite (satellite performing at the measured picker median) ends a decade at ≈{(0.9+0.1*rf_final['p50'])*100:.0f}% of pure QQQ-DCA wealth; a lucky (90th-percentile) satellite gets you to ≈{(0.9+0.1*rf_final['p90'])*100:.0f}%; an unlucky one ≈{(0.9+0.1*rf_final['p10'])*100:.0f}%. The satellite decides your stories; the core decides your wealth.</li>
<li><b>Review yearly against QQQ; feed winners' trims to the core; never feed losers.</b> (11.4's rules apply unchanged.)</li>
</ul>

<h3>11.7 &nbsp;The do-not list (each one measured somewhere above)</h3>
<ul>
<li>✗ No waiting in cash for crashes (§7: cost ~30% of final wealth).</li>
<li>✗ No on/off market timing switches (§7: trade-day lottery).</li>
<li>✗ No leverage you haven't priced at −84% (§8).</li>
<li>✗ No selling winners to buy losers, anywhere, ever (§3–4).</li>
<li>✗ No strategies sold on a backtest that hasn't passed the five audits (§5).</li>
<li>✗ No acting on streaks — yours or anyone's (§6: three 10-year streaks per 10,000 coin-flippers).</li>
</ul>

<div class="card" style="border-left:4px solid var(--good)"><b>How to prove this study wrong.</b> A real study states its falsification conditions. This one is wrong if any of the following is produced: <b>(1)</b> a rule-based strategy, specified in advance, that beats QQQ-DCA on point-in-time delisting-inclusive data and passes all five audits (random-picker null, lead-timing, other eras, other trade days, honest costs); <b>(2)</b> a picker or fund, open to ordinary investors at ordinary size, with 15+ years of audited live returns above QQQ net of fees \u2014 enough to clear the skill-detection bar in \u00a76c; or <b>(3)</b> a demonstration that the base-rate measurements here (6.2%, 42%, 4.6%, 0.18%) are materially wrong on equivalent data. We checked every candidate we could find, including our own. The section stays until someone clears it \u2014 and the authors would genuinely like to see it cleared.</div>

<h2 id="s12">Methodology, references &amp; sources</h2>
<h3>Limitations &amp; robustness (stated plainly)</h3>
<ul class="note" style="font-size:12.5px">
<li><b>Delisting convention:</b> headline decade stats count disappeared stocks at their final traded price (acquisitions exit at deal price). Under the harshest alternative (every disappearance = −100%), the share of stocks beating QQQ moves from 6.2% to 5.9% — the conclusion is insensitive to the convention.</li>
<li><b>Window robustness:</b> the core result was measured three independent ways — one full decade (2016–26: 6.2% beat), twenty-six annual cohorts (2000–25: average 42% beat over 12 months, majority-beat only when QQQ itself was crashing), and a 21-year lottery (2005–26: 4.6% beat). All agree.</li>
<li><b>Granularity:</b> stock-level analysis is monthly; it cannot see intraday effects (which the day-trading literature covers, and which are worse for retail [14][16][25]).</li>
<li><b>Scope:</b> U.S. common stocks only; QQQ measured as the actual ETF (fees included). Taxes are not modeled — omitting them <i>flatters every challenger</i>, since the challengers trade and the benchmark doesn't.</li>
<li><b>The era caveat is §9's:</b> 1999–2026 is one long sample dominated by U.S. large-cap tech; the anti-picking mechanisms (skew, concentration, arithmetic of active management) are era-independent, but QQQ's specific margin over broader indexes is not.</li>
<li><b>Simulations are simulations:</b> every number here is a backtest or historical measurement, not a guarantee; the behavioral evidence [14][15][16] suggests live results for active approaches would be worse, not better.</li>
</ul>
<h3>Published research integrated above</h3>
<ol class="refs">
<li>Bessembinder, H. (2018). “Do Stocks Outperform Treasury Bills?” <i>Journal of Financial Economics</i> 129(3). <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2900447">SSRN</a></li>
<li>J.P. Morgan Private Bank, <i>Eye on the Market</i>: “The Agony and the Ecstasy: The Risks and Rewards of a Concentrated Stock Position.” Russell 3000, 1980–2014. <a href="https://privatebank.jpmorgan.com/content/dam/jpm-wm-aem/global/pb/en/insights/eye-on-the-market/eotm-the-agony-and-the-ecstasy.pdf">PDF</a></li>
<li>Heaton, J.B., Polson, N., &amp; Witte, J. (2017). “Why Indexing Works.” <i>Applied Stochastic Models in Business and Industry</i>.</li>
<li>S&amp;P Dow Jones Indices, <i>U.S. Persistence Scorecard</i> (annual). <a href="https://www.spglobal.com/spdji/en/spiva/article/us-persistence-scorecard/">S&amp;P</a></li>
<li>Jegadeesh, N., &amp; Titman, S. (1993). “Returns to Buying Winners and Selling Losers.” <i>Journal of Finance</i> 48(1).</li>
<li>Daniel, K., &amp; Moskowitz, T. (2016). “Momentum Crashes.” <i>Journal of Financial Economics</i> 122(2).</li>
<li>Sharpe, W.F. (1991). “The Arithmetic of Active Management.” <i>Financial Analysts Journal</i> 47(1).</li>
<li>Fama, E., &amp; French, K. (2010). “Luck versus Skill in the Cross-Section of Mutual Fund Returns.” <i>Journal of Finance</i> 65(5).</li>
<li>Carhart, M. (1997). “On Persistence in Mutual Fund Performance.” <i>Journal of Finance</i> 52(1).</li>
<li>McLean, R.D., &amp; Pontiff, J. (2016). “Does Academic Research Destroy Stock Return Predictability?” <i>Journal of Finance</i> 71(1).</li>
<li>Harvey, C., Liu, Y., &amp; Zhu, H. (2016). “…and the Cross-Section of Expected Returns.” <i>Review of Financial Studies</i> 29(1).</li>
<li>S&amp;P Dow Jones Indices, <i>SPIVA U.S. Scorecard</i>, Year-End 2024: 89.50% of large-cap funds underperformed the S&amp;P 500 over 15 years. <a href="https://www.spglobal.com/spdji/en/documents/spiva/spiva-us-year-end-2024.pdf">PDF</a></li>
<li>Buffett–Protégé Partners wager, 2008–2017; documented in Berkshire Hathaway shareholder letters (2016, 2017).</li>
<li>Barber, B., &amp; Odean, T. (2000). “Trading Is Hazardous to Your Wealth.” <i>Journal of Finance</i> 55(2). <a href="https://faculty.haas.berkeley.edu/odean/papers%20current%20versions/individual_investor_performance_final.pdf">PDF</a></li>
<li>Morningstar, <i>Mind the Gap</i> (2024): 1.1 pp/yr investor-return gap, ~15% of total returns. <a href="https://www.morningstar.com/business/insights/research/mind-the-gap">Morningstar</a></li>
<li>Chague, F., De-Losso, R., &amp; Giovannetti, B. (2020). “Day Trading for a Living?” <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3423101">SSRN</a></li>
<li>Vanguard Research: lump-sum vs cost-averaging (~two-thirds of historical periods favor immediate investment).</li>
<li>Grossman, S., &amp; Stiglitz, J. (1980). “On the Impossibility of Informationally Efficient Markets.” <i>American Economic Review</i> 70(3).</li>
<li>Sullivan, R., Timmermann, A., &amp; White, H. (1999). “Data-Snooping, Technical Trading Rule Performance, and the Bootstrap.” <i>Journal of Finance</i> 54(5) — 7,846 technical rules, no out-of-sample value. <a href="https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00163">Wiley</a></li>
<li>Bajgrowicz, P., &amp; Scaillet, O. (2012). “Technical Trading Revisited: False Discoveries, Persistence Tests, and Transaction Costs.” <i>Journal of Financial Economics</i> 106(3).</li>
<li>Sullivan, R., Timmermann, A., &amp; White, H. (2001). “Dangers of Data Mining: The Case of Calendar Effects in Stock Returns.” <i>Journal of Econometrics</i> 105(1).</li>
<li>Asness, C., Ilmanen, A., &amp; Maloney, T. (2017). “Market Timing: Sin a Little.” <i>Journal of Investment Management</i> 15(3).</li>
<li>Ben-David, I., Franzoni, F., Kim, B., &amp; Moussawi, R. (2023). “Competition for Attention in the ETF Space.” <i>Review of Financial Studies</i> 36(3) — specialized ETFs −6%/yr for five years post-launch. <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3765063">SSRN</a></li>
<li>Arnott, R., &amp; Wu, L. (2012). “The Winners Curse: Too Big to Succeed?” Research Affiliates — sector/market “top dogs” subsequently underperform, in every market studied. <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2088515">SSRN</a></li>
<li>Bogousslavsky, V., &amp; Muravyev, D. (2024). “An Anatomy of Retail Option Trading” — retail option purchases lose ~4% per trade; 0DTE worse. <a href="https://cdn.cboe.com/resources/education/research_publications/Retail_Profitability.pdf">paper</a>; Beckmeyer, H., Branger, N., &amp; Gayda, L. (2023). “Retail Traders Love 0DTE Options… But Should They?” <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4404704">SSRN</a></li>
<li>Bauer, R., Cosemans, M., &amp; Eichholtz, P. (2009). “Option Trading and Individual Investor Performance.” <i>Journal of Banking &amp; Finance</i> 33(4).</li>
<li>Cowles, A. (1933). “Can Stock Market Forecasters Forecast?” <i>Econometrica</i> 1(3) — the field's founding study; ~12,000 forecasts; answer: “It is doubtful.” <a href="https://cowles.yale.edu/sites/default/files/2022-08/cowles-forecasters33.pdf">PDF</a></li>
<li>Graham, J., &amp; Harvey, C. (1996–97). “Market Timing Ability and Volatility Implied in Investment Newsletters' Asset Allocation Recommendations.” <i>Journal of Financial Economics</i> 42; 326 newsletters, no timing ability. <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6006">SSRN</a></li>
<li>Malkiel, B., &amp; Saha, A. (2005). “Hedge Funds: Risk and Return.” <i>Financial Analysts Journal</i> 61(6) — backfill ≈ +5%/yr, survivorship ≈ +4.4%/yr; corrected aggregate ≈ index. <a href="https://www.princeton.edu/~ceps/workingpapers/104malkiel.pdf">PDF</a>; Ibbotson &amp; Chen (2006): ≈5.7%/yr combined overstatement. <a href="https://depot.som.yale.edu/icf/papers/fileuploads/2597/original/06-10.pdf">Yale ICF</a></li>
<li>Lack, S. (2012). <i>The Hedge Fund Mirage</i> (Wiley) — industry-lifetime investor profits vs ~$0.5T in fees; T-bill comparison. <a href="https://www.wiley.com/en-us/The+Hedge+Fund+Mirage:+The+Illusion+of+Big+Money+and+Why+It%27s+Too+Good+to+Be+True-p-9781118164310">Wiley</a></li>\n<li>Bessembinder, H., Chen, T.-F., Choi, G., &amp; Wei, K.C.J. (2019). \u201cDo Global Stocks Outperform US Treasury Bills?\u201d \u2014 64,000+ stocks, 42 countries: top 1.3% of firms = all $44.7T net global wealth creation; 61% of non-US stocks trail T-bills. <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3415739">SSRN</a></li>\n<li>Odean, T. (1998). \u201cAre Investors Reluctant to Realize Their Losses?\u201d <i>Journal of Finance</i> 53(5) \u2014 the disposition effect: winners sold ~50% more readily than losers.</li>
</ol>
<p class="note">All statistics computed from point-in-time, delisting-inclusive U.S. market data: ~24,000 tickers including ~8,900 that no longer trade, 1990–2026; prices adjusted for splits/dividends; disappeared stocks counted at their final traded price (acquisitions exit at deal price); liquidity floor (price ≥ $3, median daily volume ≥ $2M) applied at each historical date using only that date's information. Strategy tests charge 5–20 bps per side and give the benchmark identical cash flows. Charts generated by <a href="{GH}/scripts/gen_verdict.py">gen_verdict.py</a> from <a href="{GH}/scripts/verdict_evidence.py">verdict_evidence.py</a>; underlying research records: <a href="{GH}/dca/research/strategies/ascent/FINDINGS.md">stock-selection studies</a>, <a href="{GH}/leverage_etf_dca/README.md">ETF-timing studies</a>, <a href="{GH}/dca/README.md">DCA-selection validation</a>, <a href="{GH}/dca/research/strategies/METHODOLOGY_validation.md">validation playbook</a>. External: Bessembinder, <i>Do Stocks Outperform Treasury Bills?</i> (2018); S&amp;P SPIVA scorecards; Buffett's 2008–2017 index-vs-hedge-funds bet.</p>
<footer>Version 2.0 · data through June 2026 · U.S. markets. Research, not investment advice. Backtests are simulations; past performance does not guarantee future results.</footer>
</div></body></html>"""

out = f"{ROOT}/docs/verdict.html"
open(out, "w").write(html)
print(f"written {out} ({len(html):,} bytes)")
