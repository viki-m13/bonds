#!/usr/bin/env python3
"""Honest DCA-vs-SPY-DCA-vs-QQQ-DCA analysis of the CRT / Daily Stock Guide strategy.

Uses the *real* point-in-time return streams published in the public viki-m13/crt
repo (no re-tuning), plus QQQ total-return prices from viki-m13/bonds, all run
through one faithful monthly-DCA simulation:

  contribute $1 at the START of each month, earn that month's return.

Two layers are reported, honestly:
  1. PRODUCTION  - the v5/E2 stream the site ships
     (experiments/docs/monthly-dca/data.json -> dca_investor.growth)
  2. INDEPENDENT - a clean walk-forward PIT retrain of a comparable recipe
     (research/validation/{sp500_pit,ndx_pit}/backtest_*.csv ; REPORT.md)
"""
import json, csv, bisect, datetime as dt, statistics, os

CRT_ROOT = os.environ.get("CRT_ROOT", "/tmp/crt")
BONDS    = os.environ.get("BONDS_ROOT", "/home/user/bonds")
PROD = f"{CRT_ROOT}/experiments/docs/monthly-dca/data.json"
QQQ  = f"{BONDS}/data/etfs/QQQ.csv"
OUT  = f"{BONDS}/docs/crt_dca_data.json"

# ---------- helpers ----------
def dca_curve(rets):
    v, out = 0.0, []
    for r in rets:
        v = (v + 1.0) * (1.0 + r); out.append(v)
    return out

def xirr_monthly(rets):
    """Annualised money-weighted IRR: $1 in at start of each month, terminal V."""
    V = dca_curve(rets)[-1]; N = len(rets)
    lo, hi = -0.95, 3.0
    for _ in range(200):
        m = (lo + hi) / 2
        f = sum((1.0 + m) ** (N - k + 1) for k in range(1, N + 1)) - V
        if f > 0: hi = m
        else: lo = m
    return (1.0 + (lo + hi) / 2) ** 12 - 1.0

def max_value_dd(curve):
    peak, mdd = -1e18, 0.0
    for v in curve:
        peak = max(peak, v); mdd = min(mdd, v / peak - 1.0)
    return mdd

def block(rets):
    c = dca_curve(rets)
    return dict(moic=round(c[-1] / len(rets), 3), irr=round(xirr_monthly(rets), 4),
                max_dd=round(max_value_dd(c), 4))

# ---------- load production stream ----------
d = json.load(open(PROD))
g = d["dca_investor"]["growth"]
dates  = [m["date"] for m in g]
r_strat = [m["r"] for m in g]
r_spy   = [m["s"] for m in g]
n = len(dates)

# ---------- QQQ total return aligned to same month ends ----------
qrows = [(row["Date"], float(row["Close"])) for row in csv.DictReader(open(QQQ)) if row["Close"].strip()]
qd = [x[0] for x in qrows]; qp = [x[1] for x in qrows]
def pxle(iso):
    i = bisect.bisect_right(qd, iso) - 1
    return qp[i] if i >= 0 else None
prev = pxle((dt.date.fromisoformat(dates[0]) - dt.timedelta(days=31)).isoformat())
r_qqq = []
for iso in dates:
    p = pxle(iso); r_qqq.append(p / prev - 1.0); prev = p

cs, csp, cq = dca_curve(r_strat), dca_curve(r_spy), dca_curve(r_qqq)

# ---------- rolling windows ----------
def rolling(L):
    bs = bq = 0; ms, mp, mq = [], [], []
    for st in range(0, n - L + 1):
        a = xirr_monthly(r_strat[st:st+L]); b = xirr_monthly(r_spy[st:st+L]); c = xirr_monthly(r_qqq[st:st+L])
        bs += a > b; bq += a > c; ms.append(a); mp.append(b); mq.append(c)
    w = n - L + 1
    return dict(windows=w, beat_spy=round(bs/w,3), beat_qqq=round(bq/w,3),
                med_strat=round(statistics.median(ms),4),
                med_spy=round(statistics.median(mp),4),
                med_qqq=round(statistics.median(mq),4))
roll = {nm: rolling(L) for L, nm in [(36,"3y"),(60,"5y"),(120,"10y")]}

# ---------- eras ----------
def idx_ge(ym):
    for i, dd in enumerate(dates):
        if dd[:7] >= ym: return i
    return n
eras_def = [("2003–2009","2003-03","2010-01"),("2010–2015","2010-01","2016-01"),
            ("2016–2020","2016-01","2021-01"),("2021–2026","2021-01","2026-05")]
eras = []
for nm, a, b in eras_def:
    i0, i1 = idx_ge(a), idx_ge(b)
    ws, wp, wq = r_strat[i0:i1], r_spy[i0:i1], r_qqq[i0:i1]
    si, pi, qi = xirr_monthly(ws), xirr_monthly(wp), xirr_monthly(wq)
    eras.append(dict(era=nm, months=len(ws),
        strat_irr=round(si,4), spy_irr=round(pi,4), qqq_irr=round(qi,4),
        strat_moic=round(dca_curve(ws)[-1]/len(ws),2),
        spy_moic=round(dca_curve(wp)[-1]/len(wp),2),
        qqq_moic=round(dca_curve(wq)[-1]/len(wq),2),
        beat_spy=int(si>pi), beat_qqq=int(si>qi)))

# ---------- independent PIT retrain streams ----------
spy_by_ym = {m["date"][:7]: m["s"] for m in g}
def val_block(path, name):
    rows = [r for r in csv.DictReader(open(path)) if r.get("reason") != "insufficient_train"]
    vd = [r["date"] for r in rows]; rs = [float(r["ret_m"]) for r in rows]
    sp = [spy_by_ym.get(x[:7], 0.0) for x in vd]
    prev = pxle((dt.date.fromisoformat(vd[0]) - dt.timedelta(days=31)).isoformat()); qq = []
    for x in vd:
        p = pxle(x); qq.append(p/prev - 1.0); prev = p
    return dict(name=name, months=len(rs), window=f"{vd[0]} .. {vd[-1]}",
        strat_irr=round(xirr_monthly(rs),4), spy_irr=round(xirr_monthly(sp),4), qqq_irr=round(xirr_monthly(qq),4),
        strat_moic=round(dca_curve(rs)[-1]/len(rs),2),
        spy_moic=round(dca_curve(sp)[-1]/len(sp),2),
        qqq_moic=round(dca_curve(qq)[-1]/len(qq),2))
validation = [
    val_block(f"{CRT_ROOT}/research/validation/sp500_pit/backtest_extended_2007_2024.csv","S&P 500 PIT retrain"),
    val_block(f"{CRT_ROOT}/research/validation/ndx_pit/backtest_extended_2019_2025.csv","NASDAQ-100 PIT retrain"),
]

# ---------- write ----------
out = dict(
    as_of=d["as_of"], strategy_version=d["strategy_version"],
    window=f"{dates[0]} .. {dates[-1]}", n_months=n,
    convention="Contribute $1 at the start of each month; earn that month's return. Strategy net of 10bps. PIT S&P 500. SPY and QQQ total return.",
    source="Production stream: viki-m13/crt experiments/monthly_dca/v5 (E2). Independent retrain: crt research/validation (REPORT.md). QQQ TR: viki-m13/bonds data/etfs/QQQ.csv.",
    full_period=dict(strat=block(r_strat), spy=block(r_spy), qqq=block(r_qqq)),
    rolling=roll, eras=eras, validation=validation,
    # monthly returns let the page rebuild a fresh DCA for any start window
    growth=[dict(date=dates[i], invested=i+1, strat=round(cs[i],4), spy=round(csp[i],4), qqq=round(cq[i],4),
                 r=round(r_strat[i],6), s=round(r_spy[i],6), q=round(r_qqq[i],6)) for i in range(n)],
)
json.dump(out, open(OUT, "w"), separators=(",", ":"))
print("wrote", OUT, "| months", n, "| size", os.path.getsize(OUT))
print("FULL  strat IRR %.1f%% MoIC %.0fx | SPY %.1f%% %.2fx | QQQ %.1f%% %.2fx"%(
    out["full_period"]["strat"]["irr"]*100, out["full_period"]["strat"]["moic"],
    out["full_period"]["spy"]["irr"]*100, out["full_period"]["spy"]["moic"],
    out["full_period"]["qqq"]["irr"]*100, out["full_period"]["qqq"]["moic"]))
for v in validation:
    print("RETRAIN %s: strat %.1f%% vs SPY %.1f%% vs QQQ %.1f%%"%(v["name"],v["strat_irr"]*100,v["spy_irr"]*100,v["qqq_irr"]*100))
