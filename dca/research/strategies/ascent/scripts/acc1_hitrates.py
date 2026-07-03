"""ACCURACY study 1 — P(stock beats QQQ) as the objective.
  A. Base rates: P(random eligible stock beats QQQ over 3m/12m), by era.
  B. Hit-rate of top-20 selections per signal (incl. stock-level trailing
     INFORMATION RATIO vs QQQ — the accuracy-native, never-tested signal).
  C. Consistency: share of months where >50% of the 20 picks beat QQQ.
Eligible pond: top-1500 $vol, price>=$3, above 10mo MA (tradeable names).
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os as _os; HERE = _os.environ.get("ASCENT_WORK", "/tmp/ascent_work"); _os.makedirs(HERE, exist_ok=True)
sys.path.insert(0, HERE)

t0 = time.time()
def p(*a): print(*a, flush=True)
D = pd.read_pickle(f"{HERE}/_featmat.pkl")
FEAT, liq, me, dv, bench, cols = D["FEAT"], D["liq"], D["me"], D["dv"], D["bench"], D["cols"]
M = me.index
ma10 = me.rolling(10, min_periods=10).mean()
qqq = bench["QQQ"].reindex(M)
PROB = pd.read_pickle(f"{HERE}/_mlprob.pkl").reindex(M).reindex(columns=cols)
dvr = dv.rank(axis=1, ascending=False)
top1500 = dvr <= 1500
ELIG = liq & top1500 & (me >= 3.0) & (me > ma10)

ret_m = me.pct_change().clip(-0.9, 3.0)
qret_m = qqq.pct_change()
ex_m = ret_m.sub(qret_m, axis=0)                       # monthly excess vs QQQ
# trailing IR vs QQQ (12m and 24m): mean excess / std excess (monthly)
def trailing_ir(w):
    mu = ex_m.rolling(w, min_periods=int(w * 0.8)).mean()
    sd = ex_m.rolling(w, min_periods=int(w * 0.8)).std()
    return mu / (sd + 1e-9)
IR12, IR24 = trailing_ir(12), trailing_ir(24)
TE12 = ex_m.rolling(12, min_periods=10).std()

def r(df): return df.where(ELIG).rank(axis=1, pct=True)
SIGNALS = {
    "IR24 (accuracy-native)": IR24,
    "IR12": IR12,
    "IR24 & lowTE": IR24.where(TE12.rank(axis=1, pct=True) <= 0.4),
    "mom12": me / me.shift(12) - 1,
    "lowvol": -FEAT["vol6"],
    "quality ROA": FEAT["roa"],
    "near 52w high": FEAT["distHigh"],
    "ML prob": PROB,
    "dollar volume": dv,
    "low beta-ish (lowTE)": -TE12,
    "buyback": -FEAT["share_chg"],
}

for H, tag in [(3, "3m"), (12, "12m")]:
    beat = ((me.shift(-H) / me) > (qqq.shift(-H) / qqq).values[:, None])
    beat = beat.where(ELIG & me.shift(-H).notna())
    ERAS = [("2005-01", "2009-12"), ("2010-01", "2014-12"), ("2015-01", "2019-12"),
            ("2020-01", str(pd.Timestamp("2026-06-01") - pd.DateOffset(months=H))[:7])]
    p(f"\n=== horizon {tag}: P(beat QQQ) — base rate then top-20 hit rates ===")
    hdr = f"{'signal':26} " + " ".join(f"{a[:7]:>8}" for a, b in ERAS) + "   consist"
    p(hdr)
    # base rate
    vals = []
    for st, en in ERAS:
        dts = M[(M >= pd.Timestamp(st + "-01")) & (M <= pd.Timestamp(en + "-01"))]
        vals.append(beat.loc[dts].stack().mean())
    p(f"{'BASE RATE (all eligible)':26} " + " ".join(f"{v:>8.1%}" for v in vals))
    for nm, sig in SIGNALS.items():
        sg = sig.where(ELIG)
        rk = sg.rank(axis=1, ascending=False)
        sel = rk <= 20
        vals = []; consist = []
        for st, en in ERAS:
            dts = M[(M >= pd.Timestamp(st + "-01")) & (M <= pd.Timestamp(en + "-01"))]
            hits = []
            for dt in dts:
                b = beat.loc[dt][sel.loc[dt]].dropna()
                if len(b) >= 10: hits.append(b.mean())
            vals.append(np.mean(hits) if hits else np.nan)
            consist.extend([h > 0.5 for h in hits])
        p(f"{nm:26} " + " ".join(f"{v:>8.1%}" for v in vals) +
          f"   {np.mean(consist):.0%}" if consist else "")
p(f"\nDONE t={time.time()-t0:.0f}s")
