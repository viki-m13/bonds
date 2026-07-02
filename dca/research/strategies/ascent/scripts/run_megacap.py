"""Last angles: fish exactly in QQQ's pond.
  E. Mega-cap momentum + dollar-volume tilt (SUMMIT bull sleeve) with the
     sell-allowed mandate (trail/trend exits, min-30d hold).
  F. NDX-PIT-constrained ML / momentum picks (top 5-8 inside QQQ's universe).
  G. Mega-cap ML: ML score, ADV >= $200M.
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
PROB = pd.read_pickle(f"{HERE}/_mlprob.pkl").reindex(M).reindex(columns=cols)
ma10 = me.rolling(10, min_periods=10).mean()
mom3 = me / me.shift(3) - 1
qqq = bench["QQQ"]
START, END = pd.Timestamp("2015-01-01"), pd.Timestamp("2026-06-01")
dates = M[(M >= START) & (M <= END)]
bq = dca_benchmark(qqq, dates)

def show(nm, r):
    eq = r["equity"]; s = stats(eq)
    ratio = s["final"] / bq["V"].iloc[-1]
    rr = twr(eq); qr = twr(bq)
    subs = []
    for lo, hi in [("2015-01", "2021-12"), ("2022-01", "2026-06")]:
        a = rr[(rr.index >= lo) & (rr.index <= hi)]; b = qr[(qr.index >= lo) & (qr.index <= hi)]
        subs.append(f"{(1+a).prod()**(12/len(a))-1:+.1%}v{(1+b).prod()**(12/len(b))-1:+.1%}")
    p(f"{nm:56} IRR {s['irr']:6.1%} Sh {s['sharpe']:5.2f} DD {s['maxdd']:6.1%} vsQQQ {ratio:5.2f}x [{subs[0]} | {subs[1]}]")

sq = stats(bq)
p(f"QQQ-DCA IRR {sq['irr']:.1%} Sh {sq['sharpe']:.2f} DD {sq['maxdd']:.1%}\n")
base = dict(dates=dates, trail=-0.30, ma=ma10, minhold_days=30,
            cost=0.0010, delist_ret=-0.25, cash_policy="add_top_held")

# ---- E. SUMMIT-style: rank(momentum multi-horizon skip-1m) + 5*rank(dollar volume) ----
def r(df): return df.where(liq).rank(axis=1, pct=True)
mom_multi = sum(r(me.shift(1) / me.shift(1 + h) - 1) for h in [3, 6, 9, 12]) / 4
SUMMIT_SC = mom_multi + 5 * r(dv)
ELM = (liq & (me >= 3.0) & (dv >= 2e6) & (me > ma10))
for N in [2, 5, 8]:
    show(f"E: MEGACAP-MOM k{N} trail30+trend", dca_run(me, SUMMIT_SC, ELM, **{**base, "N": N}))
show("E: MEGACAP-MOM k5 never-sell", dca_run(me, SUMMIT_SC, ELM, **{**base, "N": 5, "trail": -0.99, "ma": None}))

# ---- F. NDX-PIT constrained ----
mem = pd.read_parquet(f"{REPO}/data/pit/n100_panel_member.parquet")
mem.index = pd.to_datetime(mem.index)
memM = mem.resample("MS").last().reindex(M).ffill(limit=2).fillna(False)
memM = memM.reindex(columns=cols, fill_value=False)
ELN = (liq & memM.astype(bool))
for N in [5, 8]:
    show(f"F: NDX-ML k{N} trail30+trend", dca_run(me, PROB, ELN & (me > ma10) & (mom3 > 0), **{**base, "N": N}))
    show(f"F: NDX-MOM k{N}", dca_run(me, r(me / me.shift(12) - 1), ELN & (me > ma10), **{**base, "N": N}))
show("F: NDX-MOM k5 never-sell", dca_run(me, r(me / me.shift(12) - 1), ELN & (me > ma10), **{**base, "N": 5, "trail": -0.99, "ma": None}))

# ---- G. mega-cap ML ----
EL200 = (liq & (dv >= 2e8) & (me > ma10) & (mom3 > 0))
for N in [5, 8]:
    show(f"G: ML k{N} ADV>=$200M", dca_run(me, PROB, EL200, **{**base, "N": N}))
p(f"\nDONE t={time.time()-t0:.0f}s")
