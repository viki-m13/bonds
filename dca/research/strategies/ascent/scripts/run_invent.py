"""Invention round 2 — mechanisms that can HONESTLY raise expected CAGR vs QQQ-DCA:
  INV-1  Beta-harvest: top-k highest trailing-beta NDX members (compensated risk =
         implicit leverage on QQQ's own engine), trend-gated, cut-losers.
  INV-2  Leadership-persistence switch: hold the leaders basket (top-5 by $vol)
         while index leadership is STABLE (top-10 $vol set overlap vs 12m ago),
         else hold QQQ. Novel causal regime target: WHICH pond, not in/out.
  INV-3  Beta x quality: high-beta gated to profitable (ROA>median) names.
  INV-4  Faithful exp67 qualifier: equal-weight ALL qualifying names, staged
         confirm-or-cull ladder (+10%@3m/+30%@6m), hard-stop -25%, trail -35%,
         trend exit — all deferred to the 30d embargo.
  INV-5  INV-2 switch applied to INV-1 (beta basket when stable, QQQ else).
All under the mandate harness (monthly, $1k, 20bps, min-30d hold, delist -25%).
Gauntlet: dev/holdout split, era extension 2003-2014, cutoff trajectory, nulls.
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
mom3 = me / me.shift(3) - 1
qqq = bench["QQQ"]

# ---------- trailing beta vs QQQ (36m monthly, min 24) ----------
r_m = me.pct_change().clip(-0.9, 3.0)
r_q = qqq.reindex(M).pct_change()
cov = r_m.rolling(36, min_periods=24).cov(r_q)
var = r_q.rolling(36, min_periods=24).var()
BETA = cov.div(var, axis=0)

# ---------- NDX membership (2015+) and pre-2015 proxy pond (top-100 by $vol) ----------
mem = pd.read_parquet(f"{REPO}/data/pit/n100_panel_member.parquet")
mem.index = pd.to_datetime(mem.index)
memM = mem.resample("MS").last().reindex(M).ffill(limit=2).fillna(False)
memM = memM.reindex(columns=cols, fill_value=False).astype(bool)
top100dv = dv.rank(axis=1, ascending=False) <= 100
POND = memM.copy()
pre15 = M < pd.Timestamp("2015-01-01")
POND.loc[pre15] = top100dv.loc[pre15]

# ---------- leadership-persistence detector (causal) ----------
def topset(row, k=10):
    s = row.dropna().sort_values(ascending=False)
    return set(s.index[:k])
tops = {dt: topset(dv.where(POND.astype(bool)).loc[dt]) for dt in M}
pers = pd.Series({dt: (len(tops[dt] & tops[dt - pd.DateOffset(months=12)]) / 10.0
                       if (dt - pd.DateOffset(months=12)) in tops and len(tops[dt]) == 10 else np.nan)
                  for dt in M}).reindex(M)
PERS = pers.rolling(3, min_periods=1).mean()          # smooth
p("leadership persistence: dev distribution (2015-2021):")
dvp = PERS[(PERS.index >= "2015-01-01") & (PERS.index <= "2021-12-01")]
p(f"  mean {dvp.mean():.2f} p25 {dvp.quantile(.25):.2f} p50 {dvp.quantile(.5):.2f} p75 {dvp.quantile(.75):.2f}")

# scores
rk = lambda df: df.where(liq).rank(axis=1, pct=True)
DVSC = dv.rank(axis=1, pct=True).where(POND)
BSC = BETA.rank(axis=1, pct=True).where(POND)
roa_med = FEAT["roa"].rank(axis=1, pct=True)
BQ = BSC.where(roa_med >= 0.5)

ELN = (liq & POND & (me > ma10))
S15, S03, E = pd.Timestamp("2015-01-01"), pd.Timestamp("2003-01-01"), pd.Timestamp("2026-06-01")

def bench_of(dates): return dca_benchmark(qqq, dates)

def show(nm, eq, dates, freq=12):
    b = bench_of(dates); s = stats(eq, freq=freq)
    ratio = s["final"] / b["V"].iloc[-1]
    rr = twr(eq); qr = twr(b)
    subs = []
    for lo, hi in [("2015-01", "2021-12"), ("2022-01", "2026-06")]:
        a = rr[(rr.index >= lo) & (rr.index <= hi)]; bb = qr[(qr.index >= lo) & (qr.index <= hi)]
        if len(a) > 11:
            subs.append(f"{(1+a).prod()**(12/len(a))-1:+.1%}v{(1+bb).prod()**(12/len(bb))-1:+.1%}")
    p(f"{nm:52} IRR {s['irr']:6.1%} Sh {s['sharpe']:5.2f} DD {s['maxdd']:6.1%} vsQQQ {ratio:5.2f}x " +
      (f"[{' | '.join(subs)}]" if subs else ""))
    return ratio

base = dict(N=8, trail=-0.30, ma=ma10, minhold_days=30, cost=0.0020,
            delist_ret=-0.25, cash_policy="add_top_held")

def run(score, elig, start, end, **over):
    dates = M[(M >= start) & (M <= end)]
    cfg = {**base, **over, "dates": dates}
    return dca_run(me, score, elig, **cfg)["equity"], dates

p("\n=== INV-1 beta-harvest (NDX pond) 2015-2026 ===")
for k in [5, 8, 12]:
    eq, dts = run(BSC, ELN, S15, E, N=k)
    show(f"INV1 beta-top{k} trail30+trend", eq, dts)
eq, dts = run(BSC, ELN, S15, E, N=8, trail=-0.40)
show("INV1 beta-top8 trail40", eq, dts)

p("\n=== INV-3 beta x quality 2015-2026 ===")
eq, dts = run(BQ, ELN, S15, E, N=8)
show("INV3 beta-top8 & ROA>med", eq, dts)

p("\n=== INV-2 leadership switch (thresh 0.60 fixed on dev p25) 2015-2026 ===")
def run_switch(basket_score, k, thresh=0.60, start=S15, end=E, contrib=1000.0):
    """personal engine: leaders basket while PERS>=thresh, else QQQ. min-hold both."""
    dates = M[(M >= start) & (M <= end)]
    pos = {}; qqq_units = 0.0; qqq_entry = None; cash = 0.0; contributed = 0.0; rows = []
    for dt in dates:
        prow = me.loc[dt]; qp = qqq.reindex(M).loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]; cp = prow.get(tk, np.nan)
            if np.isfinite(cp):
                e["val"] *= cp / e["last_px"]; e["last_px"] = cp; e["peak_px"] = max(e["peak_px"], cp)
            else:
                e["val"] *= 0.75; cash += e["val"] * 0.998; pos.pop(tk)
        stable = PERS.loc[dt] >= thresh
        srow = basket_score.loc[dt]; marow = ma10.loc[dt]
        # exits: standard cuts always; if regime flipped to QQQ, liquidate stocks (post-embargo)
        for tk in list(pos.keys()):
            e = pos[tk]
            if (dt - e["entry_date"]).days < 30: continue
            cp = e["last_px"]
            cut = (cp / e["peak_px"] - 1) <= -0.30 or (np.isfinite(marow.get(tk, np.nan)) and cp < marow[tk])
            if cut or not stable:
                cash += e["val"] * 0.998; pos.pop(tk)
        cash += contrib; contributed += contrib
        if stable:
            # fund stock buys from QQQ sleeve if eligible to sell
            if qqq_units > 0 and qqq_entry is not None and (dt - qqq_entry).days >= 30:
                cash += qqq_units * qp * 0.9995; qqq_units = 0.0; qqq_entry = None
            erow = ELN.loc[dt]
            cand = srow[erow.reindex(srow.index).fillna(False).astype(bool)].dropna()
            cand = cand[~cand.index.isin(pos)].sort_values(ascending=False)
            need = k - len(pos)
            if need > 0 and cash > 1e-9 and len(cand):
                picks = list(cand.index[:need]); amt = cash / len(picks)
                for tk in picks:
                    pos[tk] = {"val": amt * 0.998, "last_px": prow[tk], "peak_px": prow[tk], "entry_date": dt}
                cash = 0.0
            elif cash > 1e-9 and len(pos):
                hs = {tk: srow.get(tk, np.nan) for tk in pos}
                tops_ = sorted(hs, key=lambda t: -(hs[t] if np.isfinite(hs[t]) else -1))[:3]
                for tk in tops_:
                    pos[tk]["val"] += (cash / len(tops_)) * 0.998
                cash = 0.0
        if cash > 1e-9 and np.isfinite(qp):
            qqq_units += cash * 0.9995 / qp
            if qqq_entry is None: qqq_entry = dt
            cash = 0.0
        V = cash + sum(e["val"] for e in pos.values()) + (qqq_units * qp if np.isfinite(qp) else 0)
        rows.append((dt, V, contributed))
    return pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date"), dates

eq, dts = run_switch(DVSC, 5)
show("INV2 switch: leaders-k5 <-> QQQ (thr .60)", eq, dts)
eq, dts = run_switch(DVSC, 5, thresh=0.70)
show("INV2 thr .70", eq, dts)
eq, dts = run_switch(DVSC, 5, thresh=0.50)
show("INV2 thr .50", eq, dts)

p("\n=== INV-5 switch on beta basket ===")
eq, dts = run_switch(BSC, 8)
show("INV5 switch: beta-k8 <-> QQQ (thr .60)", eq, dts)

p("\n=== INV-4 faithful staged-exit qualifier 2015-2026 ===")
rev_accel, rev_yoy, insn = FEAT["rev_accel"], FEAT["rev_yoy"], FEAT["ins_clustern"]
hi_yoy = rev_yoy.rank(axis=1, pct=True) >= 0.9
QMASK = ((rev_accel > 0.5) | hi_yoy) & (insn >= 2) & (me > ma10) & liq & (me >= 3.0) & (dv >= 2e6)

def run_qual_staged(start, end, contrib=1000.0, maxpos=25):
    dates = M[(M >= start) & (M <= end)]
    pos = {}; cash = 0.0; contributed = 0.0; rows = []
    for dt in dates:
        prow = me.loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]; cp = prow.get(tk, np.nan)
            if np.isfinite(cp):
                e["val"] *= cp / e["last_px"]; e["last_px"] = cp; e["peak_px"] = max(e["peak_px"], cp)
            else:
                e["val"] *= 0.75; cash += e["val"] * 0.998; pos.pop(tk)
        marow = ma10.loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]
            days = (dt - e["entry_date"]).days
            if days < 30: continue
            cp = e["last_px"]; gain = cp / e["entry_px"] - 1
            cut = (gain <= -0.25 or (cp / e["peak_px"] - 1) <= -0.35
                   or (np.isfinite(marow.get(tk, np.nan)) and cp < marow[tk])
                   or (85 <= days <= 100 and gain < 0.10) or (175 <= days <= 190 and gain < 0.30))
            if cut:
                cash += e["val"] * 0.998; pos.pop(tk)
        cash += contrib; contributed += contrib
        qm = QMASK.loc[dt]
        cand = [t for t in qm[qm].index if t not in pos and np.isfinite(prow.get(t, np.nan))]
        need = maxpos - len(pos)
        if need > 0 and cash > 1e-9 and cand:
            picks = cand[:need]; amt = cash / len(picks)
            for tk in picks:
                pos[tk] = {"val": amt * 0.998, "last_px": prow[tk], "peak_px": prow[tk],
                           "entry_px": prow[tk], "entry_date": dt}
            cash = 0.0
        elif cash > 1e-9 and len(pos):
            for tk in pos: pos[tk]["val"] += (cash / len(pos)) * 0.998
            cash = 0.0
        V = cash + sum(e["val"] for e in pos.values())
        rows.append((dt, V, contributed))
    return pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date"), dates

eq, dts = run_qual_staged(S15, E)
show("INV4 staged qualifier cap25", eq, dts)

p("\n=== Era extension 2003-2014 (all inventions; pond=top100 $vol pre-2015) ===")
for nm, fn in [
    ("INV1 beta-top8", lambda s, e: run(BSC, ELN, s, e, N=8)),
    ("INV2 switch leaders-k5", lambda s, e: run_switch(DVSC, 5, start=s, end=e)),
    ("INV5 switch beta-k8", lambda s, e: run_switch(BSC, 8, start=s, end=e)),
    ("INV4 staged qualifier", lambda s, e: run_qual_staged(s, e)),
]:
    for st, en in [("2003-01", "2009-12"), ("2010-01", "2014-12"), ("2003-01", "2014-12")]:
        eq, dts = fn(pd.Timestamp(st + "-01"), pd.Timestamp(en + "-01"))
        b = bench_of(dts); s = stats(eq)
        p(f"  {nm:24} {st}..{en}: vsQQQ {s['final']/b['V'].iloc[-1]:5.2f}x IRR {s['irr']:+6.1%} DD {s['maxdd']:6.1%}")

p("\n=== Cutoff trajectory 2015-01 start ===")
p(f"{'cutoff':8} {'INV1b8':>7} {'INV2':>7} {'INV5':>7} {'INV4':>7}")
for cut in ["2017-12", "2019-12", "2021-12", "2023-12", "2025-12", "2026-06"]:
    c = pd.Timestamp(cut + "-01"); vals = []
    eq, dts = run(BSC, ELN, S15, c, N=8); vals.append(stats(eq)["final"] / bench_of(dts)["V"].iloc[-1])
    eq, dts = run_switch(DVSC, 5, start=S15, end=c); vals.append(stats(eq)["final"] / bench_of(dts)["V"].iloc[-1])
    eq, dts = run_switch(BSC, 8, start=S15, end=c); vals.append(stats(eq)["final"] / bench_of(dts)["V"].iloc[-1])
    eq, dts = run_qual_staged(S15, c); vals.append(stats(eq)["final"] / bench_of(dts)["V"].iloc[-1])
    p(f"{cut:8} " + " ".join(f"{v:>7.2f}" for v in vals))

p("\n=== Nulls: random-in-pond with same mechanics (10 seeds), 2015-2026 ===")
rs = []
for sd in range(10):
    rng = np.random.default_rng(700 + sd)
    sc = pd.DataFrame(rng.random(me.shape), index=M, columns=cols).where(POND)
    eq, dts = run(sc, ELN, S15, E, N=8)
    rs.append(stats(eq)["final"] / bench_of(dts)["V"].iloc[-1])
p(f"random-in-pond k8: mean {np.mean(rs):.2f} min {np.min(rs):.2f} max {np.max(rs):.2f}")
rs = []
for sd in range(10):
    rng = np.random.default_rng(800 + sd)
    sc = pd.DataFrame(rng.random(me.shape), index=M, columns=cols).where(POND)
    eq, dts = run_switch(sc, 5)
    rs.append(stats(eq)["final"] / bench_of(dts)["V"].iloc[-1])
p(f"random switch-basket k5: mean {np.mean(rs):.2f} min {np.min(rs):.2f} max {np.max(rs):.2f}")
p(f"\nDONE t={time.time()-t0:.0f}s")
