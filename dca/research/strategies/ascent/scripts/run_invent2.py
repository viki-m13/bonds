"""INV-6 "concentration timing": hold the top-5 leaders (by $ volume, in the
NDX/top-100 pond) ONLY while concentration is being paid — trailing 6m return
of the current leaders basket > QQQ's trailing 6m return (+ optional persistence
gate). Otherwise hold QQQ. Always invested; min-30d hold on every position.

DISCLOSURE: designed AFTER observing the 2003-09 era failure of the static
leaders basket — so the era tests below (incl. 2000-02, never touched before)
and the nulls are the arbiters, and thresholds are standard (spread>0, 6m).
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os as _os; HERE = _os.environ.get("ASCENT_WORK", "/tmp/ascent_work"); _os.makedirs(HERE, exist_ok=True)
REPO = _os.environ.get("BONDS_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..")))
sys.path.insert(0, HERE)
from engine import dca_benchmark, stats, twr

t0 = time.time()
def p(*a): print(*a, flush=True)

D = pd.read_pickle(f"{HERE}/_featmat.pkl")
FEAT, liq, me, dv, bench, cols = D["FEAT"], D["liq"], D["me"], D["dv"], D["bench"], D["cols"]
M = me.index
ma10 = me.rolling(10, min_periods=10).mean()
qqq = bench["QQQ"].reindex(M)

mem = pd.read_parquet(f"{REPO}/data/pit/n100_panel_member.parquet")
mem.index = pd.to_datetime(mem.index)
memM = mem.resample("MS").last().reindex(M).ffill(limit=2).fillna(False)
memM = memM.reindex(columns=cols, fill_value=False).astype(bool)
top100dv = dv.rank(axis=1, ascending=False) <= 100
POND = memM.copy()
pre15 = M < pd.Timestamp("2015-01-01")
POND.loc[pre15] = top100dv.loc[pre15]

DVR = dv.rank(axis=1, ascending=False).where(POND)
mom6 = me / me.shift(6) - 1
qmom6 = qqq / qqq.shift(6) - 1

# causal concentration-premium signal: mean 6m return of CURRENT top-5 leaders minus QQQ 6m
lead_mask = DVR <= 5
SPREAD = mom6.where(lead_mask).mean(axis=1) - qmom6
SPREAD3 = SPREAD.rolling(3, min_periods=1).mean()

# persistence (from INV-2)
def topset(row, k=10):
    s = row.dropna().sort_values(ascending=False)
    return set(s.index[:k])
tops = {dt: topset(dv.where(POND.astype(bool)).loc[dt]) for dt in M}
pers = pd.Series({dt: (len(tops[dt] & tops[dt - pd.DateOffset(months=12)]) / 10.0
                       if (dt - pd.DateOffset(months=12)) in tops and len(tops[dt]) == 10 else np.nan)
                  for dt in M}).reindex(M)
PERS = pers.rolling(3, min_periods=1).mean()

DVSC = dv.rank(axis=1, pct=True).where(POND)
ELN = (liq & POND & (me > ma10))

def run_conc(score, k=5, regime=None, start=None, end=None, contrib=1000.0,
             trail=-0.30, cost=0.002):
    dates = M[(M >= start) & (M <= end)]
    pos = {}; qqq_units = 0.0; qqq_entry = None; cash = 0.0; contributed = 0.0; rows = []
    for dt in dates:
        prow = me.loc[dt]; qp = qqq.loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]; cp = prow.get(tk, np.nan)
            if np.isfinite(cp):
                e["val"] *= cp / e["last_px"]; e["last_px"] = cp; e["peak_px"] = max(e["peak_px"], cp)
            else:
                e["val"] *= 0.75; cash += e["val"] * (1 - cost); pos.pop(tk)
        on = bool(regime.loc[dt]) if regime is not None else True
        srow = score.loc[dt]; marow = ma10.loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]
            if (dt - e["entry_date"]).days < 30: continue
            cp = e["last_px"]
            cut = (cp / e["peak_px"] - 1) <= trail or (np.isfinite(marow.get(tk, np.nan)) and cp < marow[tk])
            if cut or not on:
                cash += e["val"] * (1 - cost); pos.pop(tk)
        cash += contrib; contributed += contrib
        if on:
            if qqq_units > 0 and qqq_entry is not None and (dt - qqq_entry).days >= 30:
                cash += qqq_units * qp * (1 - 0.0005); qqq_units = 0.0; qqq_entry = None
            erow = ELN.loc[dt]
            cand = srow[erow.reindex(srow.index).fillna(False).astype(bool)].dropna()
            cand = cand[~cand.index.isin(pos)].sort_values(ascending=False)
            need = k - len(pos)
            if need > 0 and cash > 1e-9 and len(cand):
                picks = list(cand.index[:need]); amt = cash / len(picks)
                for tk in picks:
                    pos[tk] = {"val": amt * (1 - cost), "last_px": prow[tk], "peak_px": prow[tk], "entry_date": dt}
                cash = 0.0
            elif cash > 1e-9 and len(pos):
                hs = {tk: srow.get(tk, np.nan) for tk in pos}
                tops_ = sorted(hs, key=lambda t: -(hs[t] if np.isfinite(hs[t]) else -1))[:3]
                for tk in tops_:
                    pos[tk]["val"] += (cash / len(tops_)) * (1 - cost)
                cash = 0.0
        if cash > 1e-9 and np.isfinite(qp):
            qqq_units += cash * (1 - 0.0005) / qp
            if qqq_entry is None: qqq_entry = dt
            cash = 0.0
        V = cash + sum(e["val"] for e in pos.values()) + (qqq_units * qp if np.isfinite(qp) else 0)
        rows.append((dt, V, contributed))
    return pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date"), dates

def show(nm, eq, dates):
    b = dca_benchmark(bench["QQQ"], dates); s = stats(eq)
    ratio = s["final"] / b["V"].iloc[-1]
    p(f"{nm:52} IRR {s['irr']:6.1%} Sh {s['sharpe']:5.2f} DD {s['maxdd']:6.1%} vsQQQ {ratio:5.2f}x")
    return ratio

REG_S = (SPREAD3 > 0)
REG_SP = (SPREAD3 > 0) & (PERS >= 0.6)
S15, E = pd.Timestamp("2015-01-01"), pd.Timestamp("2026-06-01")
S00 = pd.Timestamp("2000-01-01")

p("=== INV-6 concentration timing, 2015-2026 ===")
eq, d = run_conc(DVSC, 5, REG_S, S15, E);  show("INV6 spread-only", eq, d)
eq, d = run_conc(DVSC, 5, REG_SP, S15, E); show("INV6 spread & pers", eq, d)
eq, d = run_conc(DVSC, 5, None, S15, E);   show("static leaders k5 (no regime)", eq, d)
p(f"   regime ON share 2015-26: spread {REG_S[(M>=S15)&(M<=E)].mean():.0%}, spread&pers {REG_SP[(M>=S15)&(M<=E)].mean():.0%}")

p("\n=== Full-history eras (incl untouched 2000-02) ===")
for st, en in [("2000-01", "2002-12"), ("2003-01", "2009-12"), ("2010-01", "2014-12"),
               ("2015-01", "2019-12"), ("2020-01", "2026-06"), ("2000-01", "2026-06")]:
    stt, enn = pd.Timestamp(st + "-01"), pd.Timestamp(en + "-01")
    eq, d = run_conc(DVSC, 5, REG_S, stt, enn)
    b = dca_benchmark(bench["QQQ"], d); s = stats(eq)
    eq2, d2 = run_conc(DVSC, 5, None, stt, enn); s2 = stats(eq2)
    p(f"  {st}..{en}: INV6 {s['final']/b['V'].iloc[-1]:5.2f}x (IRR {s['irr']:+6.1%} DD {s['maxdd']:6.1%})"
      f"   static {s2['final']/b['V'].iloc[-1]:5.2f}x (DD {s2['maxdd']:6.1%})  ON {REG_S[(M>=stt)&(M<=enn)].mean():.0%}")

p("\n=== Cutoff trajectory (start 2015-01) ===")
for cut in ["2017-12", "2019-12", "2021-12", "2023-12", "2025-12", "2026-06"]:
    c = pd.Timestamp(cut + "-01")
    eq, d = run_conc(DVSC, 5, REG_S, S15, c)
    b = dca_benchmark(bench["QQQ"], d)
    p(f"  {cut}: {stats(eq)['final']/b['V'].iloc[-1]:.2f}x")
p("=== Cutoff trajectory (start 2003-01) ===")
for cut in ["2007-12", "2010-12", "2014-12", "2019-12", "2026-06"]:
    c = pd.Timestamp(cut + "-01")
    eq, d = run_conc(DVSC, 5, REG_S, pd.Timestamp("2003-01-01"), c)
    b = dca_benchmark(bench["QQQ"], d)
    p(f"  {cut}: {stats(eq)['final']/b['V'].iloc[-1]:.2f}x")

p("\n=== Threshold / param sweeps (2000-2026, ratio) ===")
for thr in [-0.02, 0.0, 0.02]:
    for smooth in [1, 3, 6]:
        R = (SPREAD.rolling(smooth, min_periods=1).mean() > thr)
        eq, d = run_conc(DVSC, 5, R, S00, E)
        b = dca_benchmark(bench["QQQ"], d)
        p(f"  thr {thr:+.2f} smooth {smooth}: {stats(eq)['final']/b['V'].iloc[-1]:.2f}x")
for k in [3, 5, 8]:
    eq, d = run_conc(DVSC, k, REG_S, S00, E)
    b = dca_benchmark(bench["QQQ"], d)
    p(f"  k={k}: {stats(eq)['final']/b['V'].iloc[-1]:.2f}x")

p("\n=== Null: random-basket with the same spread-timing overlay (10 seeds, 2000-2026) ===")
rs = []
for sd in range(10):
    rng = np.random.default_rng(900 + sd)
    sc = pd.DataFrame(rng.random(me.shape), index=M, columns=cols).where(POND)
    # spread recomputed on the random basket's own top-5 would be circular; the
    # honest null keeps the SAME regime series but random picks in the pond
    eq, d = run_conc(sc, 5, REG_S, S00, E)
    b = dca_benchmark(bench["QQQ"], d)
    rs.append(stats(eq)["final"] / b["V"].iloc[-1])
p(f"  random-picks + same regime: mean {np.mean(rs):.2f} min {np.min(rs):.2f} max {np.max(rs):.2f}")
p(f"\nDONE t={time.time()-t0:.0f}s")
