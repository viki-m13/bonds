"""ASCENT candidate race — selectors x mechanics through the honest DCA harness.
Monthly cadence, $1k/period, 20bps/side, min-30d hold, delist haircut -25%.
Reports full 2015-2026, dev 2015-2021, holdout 2022-2026, vs QQQ-DCA.
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
vol6 = FEAT["vol6"]

# eligibility: price>=$3 (prior close), median $vol >= $2M, uptrend, runner
ELIG = (liq & (me >= 3.0) & (dv >= 2e6) & (me > ma10) & (mom3 > 0))

accel = PROB - PROB.shift(2)
rk = lambda df: df.where(liq).rank(axis=1, pct=True)

# ---- selector scores ----
SC = {}
SC["ML"] = PROB
SC["ML_accel"] = PROB.where(accel > 0)
SC["FACTOR"] = (rk(FEAT["roa"]) + rk(FEAT["op_margin"]) + rk(FEAT["distHigh"])
                + rk(-FEAT["vol6"]) + rk(FEAT["mom12"]) + rk(-FEAT["share_chg"])) / 6
SC["MOM"] = rk(FEAT["mom12"])
rng = np.random.default_rng(7)
SC["RANDOM"] = pd.DataFrame(rng.random(me.shape), index=M, columns=cols).where(liq)

START, END = pd.Timestamp("2015-01-01"), pd.Timestamp("2026-06-01")
dates = M[(M >= START) & (M <= END)]
qqq = bench["QQQ"]

bq = dca_benchmark(qqq, dates)
bs = dca_benchmark(bench["SPY"], dates)
sq = stats(bq)

def show(nm, eq, sub=True):
    s = stats(eq)
    ratio = s["final"] / bq["V"].iloc[-1]
    line = (f"{nm:42} MOIC {s['moic']:5.2f}x  IRR {s['irr']:6.1%}  TWRcagr {s['twr_cagr']:6.1%}  "
            f"Sh {s['sharpe']:5.2f}  DD {s['maxdd']:6.1%}  vsQQQ {ratio:5.2f}x")
    p(line)
    if sub:
        r = twr(eq)
        for lo, hi, tag in [("2015-01", "2021-12", "dev "), ("2022-01", "2026-06", "hold")]:
            rr = r[(r.index >= lo) & (r.index <= hi)]
            qeq = twr(bq); qq = qeq[(qeq.index >= lo) & (qeq.index <= hi)]
            c = (1 + rr).prod() ** (12 / len(rr)) - 1
            s_ = rr.mean() / rr.std() * np.sqrt(12)
            qc = (1 + qq).prod() ** (12 / len(qq)) - 1
            qs = qq.mean() / qq.std() * np.sqrt(12)
            p(f"    {tag}: {c:+6.1%}/{s_:4.2f}  (QQQ {qc:+6.1%}/{qs:4.2f})")
    return s

p(f"QQQ-DCA   MOIC {sq['moic']:.2f}x IRR {sq['irr']:.1%} TWR {sq['twr_cagr']:.1%} Sh {sq['sharpe']:.2f} DD {sq['maxdd']:.1%}")
ss = stats(bs)
p(f"SPY-DCA   MOIC {ss['moic']:.2f}x IRR {ss['irr']:.1%} TWR {ss['twr_cagr']:.1%} Sh {ss['sharpe']:.2f} DD {ss['maxdd']:.1%}")
p("")

base = dict(dates=dates, N=12, trail=-0.30, ma=ma10, minhold_days=30,
            cost=0.0020, delist_ret=-0.25, cash_policy="add_top_held")

CFG = {
    "ML N12 trail30+trend (WAVE-adapted)": dict(score=SC["ML"]),
    "ML+accel N12 trail30+trend": dict(score=SC["ML_accel"]),
    "ML+accel N12 cash=hold": dict(score=SC["ML_accel"], cash_policy="hold"),
    "ML+accel N8": dict(score=SC["ML_accel"], N=8),
    "ML+accel N16": dict(score=SC["ML_accel"], N=16),
    "ML+accel trail25": dict(score=SC["ML_accel"], trail=-0.25),
    "ML+accel trail40": dict(score=SC["ML_accel"], trail=-0.40),
    "ML+accel no-trend-exit": dict(score=SC["ML_accel"], ma=None),
    "ML+accel rank-exit 36": dict(score=SC["ML_accel"], rank_exit=36),
    "ML+accel minhold60": dict(score=SC["ML_accel"], minhold_days=60),
    "FACTOR N12": dict(score=SC["FACTOR"]),
    "MOM N12 (control)": dict(score=SC["MOM"]),
}
res = {}
for nm, over in CFG.items():
    cfg = {**base, **over}
    r = dca_run(me, cfg.pop("score"), ELIG, **cfg)
    res[nm] = r
    show(nm, r["equity"])
p("")
# random null through the identical harness (5 seeds)
finals = []
for sd in range(5):
    rngs = np.random.default_rng(100 + sd)
    rsc = pd.DataFrame(rngs.random(me.shape), index=M, columns=cols).where(liq)
    r = dca_run(me, rsc, ELIG, **base)
    s = stats(r["equity"])
    finals.append(s["final"] / bq["V"].iloc[-1])
p(f"RANDOM null (5 seeds) vsQQQ ratio: mean {np.mean(finals):.2f} min {np.min(finals):.2f} max {np.max(finals):.2f}")
p(f"\nDONE t={time.time()-t0:.0f}s")
