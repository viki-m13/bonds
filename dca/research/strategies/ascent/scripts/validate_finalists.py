"""Finalist gauntlet: MEGACAP-MOM k2/k5, NDX-MOM k8, QUALIFIER N20.
  1. Proper nulls: dv-only rank control; random-in-pond controls.
  2. Cutoff-date trajectory (recency test).
  3. Pre-2015 era extension for price-only strategies (2000-2015, incl dot-com).
  4. DCA start-date window grid for survivors.
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
ma10 = me.rolling(10, min_periods=10).mean()
qqq = bench["QQQ"]

def r(df): return df.where(liq).rank(axis=1, pct=True)
mom_multi = sum(r(me.shift(1) / me.shift(1 + h) - 1) for h in [3, 6, 9, 12]) / 4
SUMMIT_SC = mom_multi + 5 * r(dv)
DV_ONLY = r(dv)
MOM12 = r(me / me.shift(12) - 1)
ELM = (liq & (me >= 3.0) & (dv >= 2e6) & (me > ma10))

mem = pd.read_parquet(f"{REPO}/data/pit/n100_panel_member.parquet")
mem.index = pd.to_datetime(mem.index)
memM = mem.resample("MS").last().reindex(M).ffill(limit=2).fillna(False)
memM = memM.reindex(columns=cols, fill_value=False).astype(bool)
ELN = (liq & memM & (me > ma10))

rev_accel, rev_yoy, insn = FEAT["rev_accel"], FEAT["rev_yoy"], FEAT["ins_clustern"]
hi_yoy = rev_yoy.rank(axis=1, pct=True) >= 0.9
QSCORE = FEAT["mom6"].where(((rev_accel > 0.5) | hi_yoy) & (insn >= 2) & (me > ma10))
ELQ = (liq & (me >= 3.0) & (dv >= 2e6))

def run(score, elig, start, end, N, cost=0.0010, **over):
    dates = M[(M >= start) & (M <= end)]
    cfg = dict(dates=dates, N=N, trail=-0.30, ma=ma10, minhold_days=30,
               cost=cost, delist_ret=-0.25, cash_policy="add_top_held")
    cfg.update(over)
    res = dca_run(me, score, elig, **cfg)
    b = dca_benchmark(qqq, dates)
    return res["equity"]["V"].iloc[-1] / b["V"].iloc[-1], stats(res["equity"]), res

S15, E = pd.Timestamp("2015-01-01"), pd.Timestamp("2026-06-01")

# ---- 1. nulls ----
p("=== Nulls (2015-2026, vsQQQ terminal ratio) ===")
ratio, s, _ = run(SUMMIT_SC, ELM, S15, E, 2)
p(f"MEGACAP-MOM k2:              {ratio:.2f}x  IRR {s['irr']:.1%}")
ratio, s, _ = run(SUMMIT_SC, ELM, S15, E, 5)
p(f"MEGACAP-MOM k5:              {ratio:.2f}x  IRR {s['irr']:.1%}")
ratio, s, _ = run(DV_ONLY, ELM, S15, E, 2)
p(f"DV-ONLY k2 (no momentum):    {ratio:.2f}x  IRR {s['irr']:.1%}")
ratio, s, _ = run(DV_ONLY, ELM, S15, E, 5)
p(f"DV-ONLY k5 (no momentum):    {ratio:.2f}x  IRR {s['irr']:.1%}")
# random among top-20 dv names
rs = []
for sd in range(10):
    rng = np.random.default_rng(300 + sd)
    noise = pd.DataFrame(rng.random(me.shape), index=M, columns=cols)
    top20 = DV_ONLY.rank(axis=1, ascending=False) <= 20
    sc = noise.where(top20)
    ratio, _, _ = run(sc, ELM, S15, E, 5)
    rs.append(ratio)
p(f"RANDOM-in-top20dv k5 (10sd): mean {np.mean(rs):.2f}x  min {np.min(rs):.2f} max {np.max(rs):.2f}")

ratio, s, _ = run(MOM12, ELN, S15, E, 8)
p(f"NDX-MOM k8:                  {ratio:.2f}x  IRR {s['irr']:.1%}")
rs = []
for sd in range(10):
    rng = np.random.default_rng(400 + sd)
    sc = pd.DataFrame(rng.random(me.shape), index=M, columns=cols).where(memM)
    ratio, _, _ = run(sc, ELN, S15, E, 8)
    rs.append(ratio)
p(f"NDX-RANDOM k8 (10 seeds):    mean {np.mean(rs):.2f}x  min {np.min(rs):.2f} max {np.max(rs):.2f}")
ratio, s, _ = run(pd.DataFrame(1.0, index=M, columns=cols).where(memM), ELN, S15, E, 100)
p(f"NDX-EQUALWEIGHT-ALL:         {ratio:.2f}x  IRR {s['irr']:.1%}")
ratio, s, _ = run(QSCORE, ELQ, S15, E, 20)
p(f"QUALIFIER N20:               {ratio:.2f}x  IRR {s['irr']:.1%}")
rs = []
for sd in range(10):
    rng = np.random.default_rng(500 + sd)
    qual_pond = QSCORE.notna()
    sc = pd.DataFrame(rng.random(me.shape), index=M, columns=cols).where(qual_pond)
    ratio, _, _ = run(sc, ELQ, S15, E, 20)
    rs.append(ratio)
p(f"RANDOM-in-qualpond N20:      mean {np.mean(rs):.2f}x  min {np.min(rs):.2f} max {np.max(rs):.2f}")

# ---- 2. cutoff trajectory ----
p("\n=== Cutoff trajectory (start 2015-01) ===")
p(f"{'cutoff':8} {'MEGA-k2':>8} {'MEGA-k5':>8} {'NDX-M-k8':>9} {'QUAL-20':>8}")
for cut in ["2017-12", "2019-12", "2021-12", "2023-12", "2025-12", "2026-06"]:
    c = pd.Timestamp(cut + "-01")
    vals = []
    for sc, el, N in [(SUMMIT_SC, ELM, 2), (SUMMIT_SC, ELM, 5), (MOM12, ELN, 8), (QSCORE, ELQ, 20)]:
        ratio, _, _ = run(sc, el, S15, c, N)
        vals.append(ratio)
    p(f"{cut:8} " + " ".join(f"{v:>8.2f}" for v in vals[:2]) + f" {vals[2]:>9.2f} {vals[3]:>8.2f}")

# ---- 3. pre-2015 era extension (price-only strategies) ----
p("\n=== Era extension 2000-2015 (price-only; QQQ data starts 1999) ===")
for st, en in [("2000-01", "2007-12"), ("2008-01", "2014-12"), ("2000-01", "2014-12")]:
    stt, enn = pd.Timestamp(st + "-01"), pd.Timestamp(en + "-01")
    for nm, sc, el, N in [("MEGA-k2", SUMMIT_SC, ELM, 2), ("MEGA-k5", SUMMIT_SC, ELM, 5),
                          ("MOM12-k8", MOM12, ELM, 8)]:
        ratio, s, _ = run(sc, el, stt, enn, N)
        p(f"  {st}..{en} {nm:9}: {ratio:.2f}x  IRR {s['irr']:+.1%}")

# ---- 4. window grid for survivors ----
p("\n=== Window grid (quarterly starts 2015-2023, to-end + 3y) ===")
for nm, sc, el, N in [("MEGA-k5", SUMMIT_SC, ELM, 5), ("NDX-MOM-k8", MOM12, ELN, 8),
                      ("QUAL-20", QSCORE, ELQ, 20)]:
    res3, rese = [], []
    for yr in range(2015, 2024):
        for q in (1, 4, 7, 10):
            stt = pd.Timestamp(f"{yr}-{q:02d}-01")
            ratio, _, _ = run(sc, el, stt, E, N)
            rese.append(ratio)
            en3 = stt + pd.DateOffset(months=35)
            if en3 <= E:
                ratio3, _, _ = run(sc, el, stt, en3, N)
                res3.append(ratio3)
    res3, rese = np.array(res3), np.array(rese)
    p(f"  {nm:11}: 3y beat {np.mean(res3 > 1):.0%} med {np.median(res3):.2f} worst {res3.min():.2f} | "
      f"to-end beat {np.mean(rese > 1):.0%} med {np.median(rese):.2f} worst {rese.min():.2f}")
p(f"\nDONE t={time.time()-t0:.0f}s")
