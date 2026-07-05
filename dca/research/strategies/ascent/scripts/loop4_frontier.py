"""LOOP4 — HONEST ACHIEVABLE FRONTIER.
Map the best monthly-DCA stock strategy that beats QQQ-DCA in the MODERN
INVESTABLE eras (2010-14, 2015-19, 2020-26, and full 2010-26) where beating
QQQ is genuinely hard, WITHOUT blowing up if 2003-09-style conditions recur.

Candidates:
 1. ASCENT concentration-timed leaders basket (top-5 NDX by $vol while leaders'
    6m return > QQQ's, else route to QQQ). Repo champion -> baseline to beat.
 2. Quality-momentum composite: top-N by rank(roa)+rank(mom12)+rank(-vol6)
    +rank(-share_chg) (mean of available legs; price-only fallback pre-2012),
    trend exit + cut losers, N in {10,20}.
 3. QQQ-core + concentrated-tilt blend: X*QQQ + (1-X)*top-5 leaders sleeve,
    monthly-rebalanced, X in {0.4,0.6}(+0.5,0.7).
 4. Extra: QQQ-core + ASCENT-sleeve blend (regime-timed tilt).

Each: ratio vs QQQ-DCA for 2003-09, 2010-14, 2015-19, 2020-26, 2010-26,
with random-in-pond null (5 seeds). Deliverable: strategy with highest MINIMUM
ratio across the 3 modern eras that also clears its null.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import dca_run, dca_benchmark, stats, twr

t0 = time.time()
def p(*a): print(*a, flush=True)

D = pd.read_pickle(f"{HERE}/_featmat.pkl")
FEAT, liq, me, dv, bench, cols = D["FEAT"], D["liq"], D["me"], D["dv"], D["bench"], D["cols"]
M = me.index
ma10 = me.rolling(10, min_periods=10).mean()
qqq = bench["QQQ"].reindex(M)

# ---------- NDX / top-100 pond (same construction as ASCENT champion) ----------
mem = pd.read_parquet("/home/user/bonds/data/pit/n100_panel_member.parquet")
mem.index = pd.to_datetime(mem.index)
memM = mem.resample("MS").last().reindex(M).ffill(limit=2).fillna(False)
memM = memM.reindex(columns=cols, fill_value=False).astype(bool)
top100dv = dv.rank(axis=1, ascending=False) <= 100
POND = memM.copy()
pre15 = M < pd.Timestamp("2015-01-01")
POND.loc[pre15] = top100dv.loc[pre15]

DVR = dv.rank(axis=1, ascending=False).where(POND)      # dollar-vol rank in pond
mom6 = me / me.shift(6) - 1
qmom6 = qqq / qqq.shift(6) - 1
lead_mask = DVR <= 5
SPREAD = mom6.where(lead_mask).mean(axis=1) - qmom6
REG_S = (SPREAD.rolling(3, min_periods=1).mean() > 0)   # ASCENT regime: concentration paying

DVSC = dv.rank(axis=1, pct=True).where(POND)            # leaders score (higher $vol = better)
ELN = (liq & POND & (me > ma10))                        # leaders eligibility (trend gated)

# ---------- standard broad pond for quality-momentum ----------
ELIG_STD = (liq & (me >= 3.0) & (dv >= 2e6) & (me > ma10)).fillna(False)

# ================= ERAS =================
ERAS = [("2003-09", "2003-01", "2009-12"),
        ("2010-14", "2010-01", "2014-12"),
        ("2015-19", "2015-01", "2019-12"),
        ("2020-26", "2020-01", "2026-06"),
        ("2010-26", "2010-01", "2026-06")]
MODERN = ["2010-14", "2015-19", "2020-26"]
def era_dates(st, en):
    return M[(M >= pd.Timestamp(st + "-01")) & (M <= pd.Timestamp(en + "-01"))]
def qfinal(dates):
    return dca_benchmark(qqq, dates)["V"].iloc[-1]

# ================= ASCENT concentration-timing runner (from run_invent2) =====
def run_conc(score, k=5, regime=None, dates=None, contrib=1000.0, trail=-0.30, cost=0.002):
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
    return pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date")

# ================= Quality-momentum composite =================
def rk(df):
    return df.where(ELIG_STD).rank(axis=1, pct=True)
r_roa   = rk(FEAT["roa"])
r_mom12 = rk(FEAT["mom12"])
r_lvol  = rk(-FEAT["vol6"])
r_shr   = rk(-FEAT["share_chg"])
# mean of available legs (price-only fallback pre-2012 where roa/share_chg NaN)
legs = [r_roa, r_mom12, r_lvol, r_shr]
stack = np.stack([l.values for l in legs])
QM = pd.DataFrame(np.nanmean(stack, axis=0), index=M, columns=cols).where(ELIG_STD)

# ================= sleeve return streams (full history) for blends ============
def sleeve_returns_leaders(k=5):
    """static top-k NDX-by-$vol trend-gated sleeve, full history -> monthly TWR."""
    dts = M[(M >= pd.Timestamp("2000-01-01")) & (M <= pd.Timestamp("2026-06-01"))]
    eq = dca_run(me, DVSC, ELN, dts, N=k, trail=-0.30, ma=ma10, minhold_days=30,
                 cost=0.0020, cash_policy="add_top_held")["equity"]
    return twr(eq)
def sleeve_returns_ascent(k=5):
    dts = M[(M >= pd.Timestamp("2000-01-01")) & (M <= pd.Timestamp("2026-06-01"))]
    eq = run_conc(DVSC, k=k, regime=REG_S, dates=dts)
    return twr(eq)

r_qqq = qqq.pct_change()
R_LEAD = sleeve_returns_leaders(5).reindex(M)
R_ASC  = sleeve_returns_ascent(5).reindex(M)

def dca_blend_eq(rA, rB, X, dates, contrib=1000.0):
    """monthly-rebalanced blend: portfolio return = X*rA + (1-X)*rB each month."""
    V = 0.0; contributed = 0.0; rows = []
    for dt in dates:
        a = rA.get(dt, np.nan); b = rB.get(dt, np.nan)
        a = 0.0 if not np.isfinite(a) else a
        b = 0.0 if not np.isfinite(b) else b
        V *= (1 + X * a + (1 - X) * b)
        V += contrib; contributed += contrib
        rows.append((dt, V, contributed))
    return pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date")

# ================= NULL helpers (5 seeds) =================
def null_dca(elig, N, trail, ma, dates, qf, nseed=5, pond_mask=None):
    rr = []
    pm = pond_mask if pond_mask is not None else elig
    for sd in range(nseed):
        g = np.random.default_rng(4000 + sd)
        noise = pd.DataFrame(g.random(me.shape), index=M, columns=cols).where(pm.astype(bool))
        eq = dca_run(me, noise, elig, dates, N=N, trail=trail, ma=ma, minhold_days=30,
                     cost=0.0020, cash_policy="add_top_held")["equity"]
        rr.append(eq["V"].iloc[-1] / qf)
    return np.mean(rr), np.max(rr)

def null_ascent(dates, qf, nseed=5):
    rr = []
    for sd in range(nseed):
        g = np.random.default_rng(900 + sd)
        sc = pd.DataFrame(g.random(me.shape), index=M, columns=cols).where(POND)
        eq = run_conc(sc, k=5, regime=REG_S, dates=dates)
        rr.append(eq["V"].iloc[-1] / qf)
    return np.mean(rr), np.max(rr)

def null_blend(X, base_r, dates, qf, nseed=5):
    """X*QQQ + (1-X)*random-leaders sleeve (random score in POND, trend-gated)."""
    rr = []
    fulld = M[(M >= pd.Timestamp("2000-01-01")) & (M <= pd.Timestamp("2026-06-01"))]
    for sd in range(nseed):
        g = np.random.default_rng(4200 + sd)
        sc = pd.DataFrame(g.random(me.shape), index=M, columns=cols).where(POND)
        eqs = dca_run(me, sc, ELN, fulld, N=5, trail=-0.30, ma=ma10, minhold_days=30,
                      cost=0.0020, cash_policy="add_top_held")["equity"]
        rr_sleeve = twr(eqs).reindex(M)
        eq = dca_blend_eq(base_r, rr_sleeve, X, dates)
        rr.append(eq["V"].iloc[-1] / qf)
    return np.mean(rr), np.max(rr)

# ================= evaluate a candidate over all eras =================
def evaluate(name, kind, **kw):
    row = {"name": name}
    p(f"\n  {name}")
    p(f"    {'era':8}  {'ratio':>6} {'IRR':>6} {'Sh':>5} {'DD':>6} | {'nullμ':>5} {'nullmax':>6}")
    for lab, st, en in ERAS:
        dts = era_dates(st, en); qf = qfinal(dts)
        if kind == "dca":
            eq = dca_run(me, kw["score"], kw["elig"], dts, N=kw["N"], trail=kw["trail"],
                         ma=kw["ma"], minhold_days=30, cost=0.0020, cash_policy="add_top_held")["equity"]
            nmu, nmx = null_dca(kw["elig"], kw["N"], kw["trail"], kw["ma"], dts, qf, pond_mask=kw.get("pond"))
        elif kind == "ascent":
            eq = run_conc(DVSC, k=kw["k"], regime=REG_S, dates=dts)
            nmu, nmx = null_ascent(dts, qf)
        elif kind == "blend":
            eq = dca_blend_eq(kw["rA"], kw["rB"], kw["X"], dts)
            nmu, nmx = null_blend(kw["X"], kw["rA"], dts, qf)
        s = stats(eq); ratio = s["final"] / qf
        row[lab] = (ratio, s["irr"], s["sharpe"], s["maxdd"], nmu, nmx)
        p(f"    {lab:8}  {ratio:6.2f} {s['irr']*100:5.0f}% {s['sharpe']:5.2f} {s['maxdd']*100:5.0f}% |"
          f" {nmu:5.2f} {nmx:6.2f}")
    mins = min(row[e][0] for e in MODERN)
    clr = all(row[e][0] > row[e][5] for e in MODERN)   # clears null MAX in all modern eras
    row["min_modern"] = mins; row["clears_null_modern"] = clr
    p(f"    -> min modern ratio {mins:.2f}   clears null-max in all 3 modern: {clr}")
    return row

# ================= run the battery =================
RESULTS = {}
p("="*82)
p("CANDIDATE 1 — ASCENT concentration-timed leaders (repo champion baseline)")
RESULTS["ASCENT_k5"] = evaluate("ASCENT k5 (spread-timed leaders<->QQQ)", "ascent", k=5)

p("\n" + "="*82)
p("CANDIDATE 2 — Quality-momentum composite (roa+mom12-vol6-share_chg)")
for N in (10, 20):
    RESULTS[f"QM_N{N}"] = evaluate(f"QualMom top-{N} (trend exit + cut losers)", "dca",
                                   score=QM, elig=ELIG_STD, N=N, trail=-0.30, ma=ma10, pond=ELIG_STD)

p("\n" + "="*82)
p("CANDIDATE 3 — QQQ-core + top-5 leaders tilt blend (monthly rebalanced)")
for X in (0.4, 0.5, 0.6, 0.7):
    RESULTS[f"BLEND_lead_X{X}"] = evaluate(f"blend {int(X*100)}% QQQ + {int((1-X)*100)}% top-5 leaders",
                                           "blend", rA=r_qqq, rB=R_LEAD, X=X)

p("\n" + "="*82)
p("CANDIDATE 4 — QQQ-core + ASCENT-sleeve tilt blend (regime-timed tilt)")
for X in (0.4, 0.5, 0.6):
    RESULTS[f"BLEND_asc_X{X}"] = evaluate(f"blend {int(X*100)}% QQQ + {int((1-X)*100)}% ASCENT-sleeve",
                                          "blend", rA=r_qqq, rB=R_ASC, X=X)

# ================= FRONTIER SUMMARY =================
p("\n" + "="*82)
p("FRONTIER SUMMARY — ranked by MIN modern-era ratio (2010-14/15-19/20-26)")
p(f"{'strategy':40} {'min_mod':>7} {'10-14':>6} {'15-19':>6} {'20-26':>6} {'10-26':>6} {'03-09':>6} {'null?':>5}")
ranked = sorted(RESULTS.values(), key=lambda r: -r["min_modern"])
for r in ranked:
    p(f"{r['name']:40} {r['min_modern']:7.2f} {r['2010-14'][0]:6.2f} {r['2015-19'][0]:6.2f} "
      f"{r['2020-26'][0]:6.2f} {r['2010-26'][0]:6.2f} {r['2003-09'][0]:6.2f} "
      f"{'Y' if r['clears_null_modern'] else 'n':>5}")

# best = highest min modern ratio that clears null in all 3 modern eras
elig_best = [r for r in ranked if r["clears_null_modern"]]
best = elig_best[0] if elig_best else ranked[0]
p(f"\nBEST (highest min-modern that clears null in all 3): {best['name']}")
p(f"  modern ratios: 2010-14 {best['2010-14'][0]:.2f}, 2015-19 {best['2015-19'][0]:.2f}, "
  f"2020-26 {best['2020-26'][0]:.2f}  (min {best['min_modern']:.2f})")
p(f"  2003-09 downside: {best['2003-09'][0]:.2f}x  (IRR {best['2003-09'][1]*100:.0f}%, DD {best['2003-09'][3]*100:.0f}%)")
p(f"  >1.2 in all 3 modern? {all(best[e][0] > 1.2 for e in MODERN)}")
p(f"  >1.2 in how many modern eras: {sum(best[e][0] > 1.2 for e in MODERN)}/3")

pd.to_pickle({"results": RESULTS, "best_name": best["name"], "eras": ERAS, "modern": MODERN,
              "note": "loop4 frontier: ASCENT champion vs quality-momentum vs QQQ-core tilt blends. "
                      "Ratios vs QQQ-DCA, random-in-pond null 5 seeds. Best = highest min modern-era "
                      "ratio clearing null-max in all 3 modern eras."},
             f"{HERE}/loop4_frontier.pkl")
p(f"\nsaved loop4_frontier.pkl  DONE t={time.time()-t0:.0f}s")
