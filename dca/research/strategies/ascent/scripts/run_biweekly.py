"""Biweekly-cadence check + Mode-2 ("QQQ-in-stock-form") quantification.
Biweekly grid = every 2nd Friday from the weekly panel; monthly scores ffilled;
stops checked biweekly; min-30d hold enforced by calendar days.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os as _os; HERE = _os.environ.get("ASCENT_WORK", "/tmp/ascent_work"); _os.makedirs(HERE, exist_ok=True)
REPO = _os.environ.get("BONDS_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..")))
sys.path.insert(0, HERE)
from engine import dca_run, dca_benchmark, stats, twr

t0 = time.time()
def p(*a): print(*a, flush=True)

D = pd.read_pickle(f"{HERE}/_featmat.pkl")
FEAT, liq, me, dv, bench, cols = D["FEAT"], D["liq"], D["me"], D["dv"], D["bench"], D["cols"]
M = me.index
WK = pd.read_pickle(f"{HERE}/_px_weekly.pkl")
wq = WK["QQQ"]; wspy = WK["SPY"]
WK = WK[[c for c in WK.columns if c in set(cols)]]
W = WK.index
BW = W[::2]                                    # biweekly grid
pxb = WK.loc[BW]                               # biweekly closes
PROB = pd.read_pickle(f"{HERE}/_mlprob.pkl")

def to_bw(dfm):
    """monthly (month-start label, month-end value) -> biweekly ffill, lag-safe:
    a monthly value labeled at month m becomes available from the first biweekly
    date AFTER month m ends."""
    d = dfm.copy()
    d.index = d.index + pd.offsets.MonthEnd(0)   # value known at month-end close
    return d.reindex(columns=pxb.columns).reindex(W, method="ffill", limit=9).loc[BW]

ma10w = WK.rolling(43, min_periods=43).mean().loc[BW]     # ~10-month MA on weekly
mom3w = pxb / pxb.shift(6) - 1                            # ~3 months
liq_b = to_bw(liq.astype(float)) > 0.5
dv_b = to_bw(dv)
PROB_b = to_bw(PROB.reindex(M))
accel_b = PROB_b - PROB_b.shift(4)
ELIG_b = (liq_b & (pxb >= 3.0) & (dv_b >= 2e6) & (pxb > ma10w) & (mom3w > 0))

START, END = pd.Timestamp("2015-01-02"), pd.Timestamp("2026-06-18")
dates = BW[(BW >= START) & (BW <= END)]
bq = dca_benchmark(wq.loc[BW], dates, contrib=500.0)
sq = stats(bq, freq=26)
p(f"QQQ-DCA biweekly IRR {sq['irr']:.1%} Sh {sq['sharpe']:.2f} DD {sq['maxdd']:.1%}")

def show(nm, r, freq=26):
    eq = r["equity"]; s = stats(eq, freq=freq)
    ratio = s["final"] / bq["V"].iloc[-1]
    p(f"{nm:52} IRR {s['irr']:6.1%} Sh {s['sharpe']:5.2f} DD {s['maxdd']:6.1%} vsQQQ {ratio:5.2f}x")

base = dict(dates=dates, N=12, trail=-0.30, ma=ma10w, minhold_days=30,
            cost=0.0020, delist_ret=-0.25, cash_policy="add_top_held", contrib=500.0)
show("BW: ML N12 trail30+trend", dca_run(pxb, PROB_b, ELIG_b, **base))
show("BW: ML+accel N12", dca_run(pxb, PROB_b.where(accel_b > 0), ELIG_b, **base))

# Mode-2: top-5 NDX members by dollar volume, trend-gated, exits allowed
mem = pd.read_parquet(f"{REPO}/data/pit/n100_panel_member.parquet")
mem.index = pd.to_datetime(mem.index)
memW = mem.reindex(W, method="ffill", limit=15).fillna(False)
memB = memW.loc[BW].reindex(columns=pxb.columns, fill_value=False).astype(bool)
DVSC_b = dv_b.rank(axis=1, pct=True).where(memB)
ELN_b = (liq_b & memB & (pxb > ma10w))
show("BW MODE2: NDX top-5 by $vol, trail30+trend", dca_run(pxb, DVSC_b, ELN_b, **{**base, "N": 5}))
show("BW MODE2: k3", dca_run(pxb, DVSC_b, ELN_b, **{**base, "N": 3}))
show("BW MODE2: k8", dca_run(pxb, DVSC_b, ELN_b, **{**base, "N": 8}))
show("BW MODE2: k5 minhold-only exits (no trend exit)", dca_run(pxb, DVSC_b, ELN_b, **{**base, "N": 5, "ma": None}))

# Mode-2 era extension proxy on broad pond (NDX membership unavailable pre-2015):
# top-5 by $vol among liquid names, trend-gated — 2000-2014
dv_full = dv.rank(axis=1, pct=True)
DVSC_m = dv_full
ma10m = me.rolling(10, min_periods=10).mean()
ELм = (liq & (me >= 3.0) & (dv >= 2e6) & (me > ma10m))
for st, en in [("2000-01", "2007-12"), ("2008-01", "2014-12")]:
    dts = M[(M >= pd.Timestamp(st + "-01")) & (M <= pd.Timestamp(en + "-01"))]
    rr = dca_run(me, DVSC_m, ELм, dates=dts, N=5, trail=-0.30, ma=ma10m,
                 minhold_days=30, cost=0.0010, delist_ret=-0.25, cash_policy="add_top_held")
    bb = dca_benchmark(bench["QQQ"], dts)
    s = stats(rr["equity"]); ratio = s["final"] / bb["V"].iloc[-1]
    p(f"MODE2-era {st}..{en} top5-$vol: vsQQQ {ratio:.2f}x IRR {s['irr']:+.1%} DD {s['maxdd']:.1%}")
p(f"\nDONE t={time.time()-t0:.0f}s")
