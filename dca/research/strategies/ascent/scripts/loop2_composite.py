"""LOOP iter 2 — DEFENDED COMPOSITE.
Combine durable SELECTION (rank-avg of coiled-quality score + twin-momentum
AGREEMENT score, among uptrend names) with DRAWDOWN DEFENSE (Faber regime
switch: when QQQ < its 10-month MA, route the monthly contribution to QQQ
instead of concentrated single-name picks; when QQQ > 10mo MA, buy top-N
composite). Two exit variants:
   A  = cut losers  (trail -30% + trend-break exit)   [symmetric-ish]
   B  = never sell winners (no trail; trend-break exit ONLY on underwater
        positions -> winners ride, losers cut at trend-break)
Fundamentals (coiled + twin fund leg) only exist 2012+, so a PRICE-ONLY
FALLBACK composite (coiled proxy = low-vol + near-high using vol6/distHigh;
momentum = twin price-momentum leg) makes 2000-09 testable. Merged panel uses
the full composite in months where fundamentals exist, else the price-only
fallback -- each era is labelled full vs price-only-fallback.
Era-sliced ratio vs QQQ-DCA + random-in-pond null (5 seeds, SAME regime
mechanics so the null measures selection skill under identical defense).
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import dca_benchmark, stats

t0 = time.time()
def p(*a): print(*a, flush=True)

D = pd.read_pickle(f"{HERE}/_featmat.pkl")
FEAT, liq, me, dv, bench, cols = D["FEAT"], D["liq"], D["me"], D["dv"], D["bench"], D["cols"]
M = me.index
ma10 = me.rolling(10, min_periods=10).mean()
elig = (liq & (me >= 3.0) & (dv >= 2e6) & (me > ma10)).fillna(False)

TW = pd.read_pickle(f"{HERE}/loop1_twinmom.pkl")
twin      = TW["twin"].reindex(index=M, columns=cols)          # agreement (price+fund), price-only pre-2011
priceonly = TW["priceonly"].reindex(index=M, columns=cols)     # price-momentum top tercile (all eras)
coiled    = pd.read_pickle(f"{HERE}/loop1_expectancy.pkl").reindex(index=M, columns=cols)  # fund coiled, 2012+

def rk(df):
    return df.where(elig).rank(axis=1, pct=True)

# ---- FULL composite (2012+): rank-avg of coiled-quality & twin-agreement ----
r_coiled = rk(coiled)
r_twin   = rk(twin)
comp_full = (r_coiled + r_twin) / 2.0        # NaN unless BOTH present => agreement+fund pond

# ---- PRICE-ONLY fallback composite (all eras) ----
# coiled proxy, price-only = low realized vol + proximity to 52w high ("coiled")
coiled_po = rk(-FEAT["vol6"]) + rk(FEAT["distHigh"])
r_coiled_po = rk(coiled_po)
r_price     = rk(priceonly)
comp_po = (r_coiled_po + r_price) / 2.0      # candidate pond = price-momentum top tercile, ranked by low-vol quality

# ---- MERGED tradable: full where fundamentals exist that month, else price-only ----
fund_avail = r_coiled.notna().sum(axis=1) >= 30
comp_merged = comp_po.copy()
comp_merged.loc[fund_avail] = comp_full.loc[fund_avail]
p(f"fundamentals (full composite) available from {M[fund_avail].min().date()} "
  f"({int(fund_avail.sum())}/{len(M)} months); earlier months use price-only fallback")

# ---- Faber regime: QQQ vs its own 10-month MA ----
qqq = bench["QQQ"]
qqq_ma10 = qqq.rolling(10, min_periods=10).mean()
regime_on = (qqq > qqq_ma10)                 # True = risk-on (buy picks); False = route to QQQ

# augmented price panel with a QQQ column so the defense can actually hold QQQ
COLS2 = list(cols) + ["QQQ"]
px_aug = me.reindex(columns=COLS2)
px_aug["QQQ"] = qqq
ma_aug = px_aug.rolling(10, min_periods=10).mean()

def make_defended_score(comp):
    """risk-on months -> composite over stocks (QQQ NaN);
       risk-off months -> all stocks NaN, QQQ is the only candidate."""
    s = comp.reindex(columns=COLS2).copy()
    off = ~regime_on.reindex(M).fillna(False)
    s.loc[off, cols] = np.nan          # suppress single-name picks in risk-off
    s["QQQ"] = np.nan
    s.loc[off, "QQQ"] = 1e6            # QQQ becomes the sole candidate risk-off
    return s

def make_plain_score(comp):
    return comp.reindex(columns=COLS2)

elig_aug = elig.reindex(columns=COLS2).fillna(False)
elig_aug["QQQ"] = True                 # QQQ always buyable; score gates when it's chosen

# ---------------- local backtester (copy of engine.dca_run + winner_safe) ----------------
def run(px, score, el, dates, N=10, trail=-0.30, ma=None, minhold_days=30,
        cost=0.0020, delist_ret=-0.25, cash_policy="add_top_held", winner_safe=False):
    didx = list(px.index); pos = {}; cash = 0.0; rows = []; contributed = 0.0; trades = []
    dset = set(dates)
    for i, dt in enumerate(didx):
        if dt < dates[0] or dt > dates[-1]:
            continue
        prow = px.loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]; cp = prow.get(tk, np.nan)
            if np.isfinite(cp):
                e["val"] *= cp / e["last_px"]; e["last_px"] = cp; e["peak_px"] = max(e["peak_px"], cp)
            else:
                e["val"] *= (1.0 + delist_ret); cash += e["val"] * (1 - cost)
                trades.append((dt, tk, "delist", e["val"], e["val"]/e["cost_basis"]-1)); pos.pop(tk)
        if dt not in dset:
            rows.append((dt, cash + sum(e["val"] for e in pos.values()), contributed)); continue
        srow = score.loc[dt] if dt in score.index else pd.Series(dtype=float)
        erow = el.loc[dt] if dt in el.index else pd.Series(dtype=float)
        marow = ma.loc[dt] if ma is not None else None
        for tk in list(pos.keys()):
            e = pos[tk]
            if (dt - e["entry_date"]).days < minhold_days:
                continue
            cp = e["last_px"]
            underwater = e["val"] < e["cost_basis"]
            hit_trail = (cp / e["peak_px"] - 1) <= trail
            hit_trend = (marow is not None and np.isfinite(marow.get(tk, np.nan)) and cp < marow[tk])
            if winner_safe:                       # only cut trend-break if the position is a loser
                hit_trend = hit_trend and underwater
            if hit_trail or hit_trend:
                cash += e["val"] * (1 - cost)
                why = "trail" if hit_trail else "trend"
                trades.append((dt, tk, why, e["val"], e["val"]/e["cost_basis"]-1)); pos.pop(tk)
        cash += 1000.0; contributed += 1000.0
        if len(srow):
            cand = srow[erow.reindex(srow.index).fillna(False).astype(bool)].dropna()
            cand = cand[~cand.index.isin(pos.keys())].sort_values(ascending=False)
            need = N - len(pos)
            if need > 0 and cash > 1e-9 and len(cand):
                picks = list(cand.index[:need]); amt = cash / len(picks)
                for tk in picks:
                    v = amt * (1 - cost)
                    pos[tk] = {"val": v, "last_px": prow[tk], "peak_px": prow[tk],
                               "entry_date": dt, "cost_basis": v}
                cash = 0.0
            elif cash > 1e-9 and cash_policy == "add_top_held" and len(pos):
                held_sc = {tk: srow.get(tk, np.nan) for tk in pos}
                tops = sorted(held_sc, key=lambda t: -(held_sc[t] if np.isfinite(held_sc[t]) else -1))[:3]
                amt = cash / len(tops)
                for tk in tops:
                    v = amt * (1 - cost); pos[tk]["val"] += v; pos[tk]["cost_basis"] += v
                cash = 0.0
        rows.append((dt, cash + sum(e["val"] for e in pos.values()), contributed))
    eq = pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date")
    return {"equity": eq}

ERAS = [("2000-02", "2000-01", "2002-12", "price-only"),
        ("2003-09", "2003-01", "2009-12", "price-only"),
        ("2010-14", "2010-01", "2014-12", "mixed"),
        ("2015-19", "2015-01", "2019-12", "full"),
        ("2020-26", "2020-01", "2026-06", "full"),
        ("2000-26", "2000-01", "2026-06", "mixed"),
        ("2015-26", "2015-01", "2026-06", "full")]

def era_dates(st, en):
    return M[(M >= pd.Timestamp(st + "-01")) & (M <= pd.Timestamp(en + "-01"))]

def one(score, el, dates, N, trail, winner_safe):
    r = run(px_aug, score, el, dates, N=N, trail=trail, ma=ma_aug,
            minhold_days=30, winner_safe=winner_safe)
    return r["equity"]

def null_ratios(dates, N, trail, winner_safe, qf, nseed=5):
    rr = []
    for sd in range(nseed):
        g = np.random.default_rng(1000 + sd)
        noise = pd.DataFrame(g.random(me.shape), index=M, columns=cols).where(elig)
        sc = make_defended_score(noise)      # SAME regime defense, random selection
        eq = one(sc, elig_aug, dates, N, trail, winner_safe)
        rr.append(eq["V"].iloc[-1] / qf)
    return np.mean(rr), np.max(rr)

VARIANTS = [("A_cutlosers", -0.30, False), ("B_ridewinners", -0.999, True)]

# ---- summary: min-era ratio for every (variant,N,defended?) ----
p(f"\n{'='*90}\nMIN-ERA RATIO SUMMARY (investable eras 2000-02..2020-26)")
summ = {}
for vname, trail, wsafe in VARIANTS:
    for N in (10, 20):
        for deftag, scf, elf in [("defended", make_defended_score, elig_aug),
                                  ("plain",    make_plain_score,    elig_aug)]:
            sc = scf(comp_merged)
            mins = []
            for lab, st, en, _ in ERAS[:5]:
                dts = era_dates(st, en)
                eq = one(sc, elf, dts, N, trail, wsafe)
                qf = dca_benchmark(qqq, dts)["V"].iloc[-1]
                mins.append(eq["V"].iloc[-1] / qf)
            summ[(vname, N, deftag)] = mins
            p(f"  {vname:14} N={N:<2} {deftag:9} min={min(mins):.2f}  eras={[round(m,2) for m in mins]}")

# ---- full era tables with nulls for the DEFENDED variants (both A & B, N=10 & 20) ----
def full_table(vname, N, trail, wsafe):
    sc = make_defended_score(comp_merged)
    p(f"\n  DEFENDED {vname}  N={N}")
    p(f"    era        lbl         ratio  IRR    Sh     DD     | null_mean null_max")
    out = {}
    for lab, st, en, lbl in ERAS:
        dts = era_dates(st, en)
        eq = one(sc, elig_aug, dts, N, trail, wsafe); s = stats(eq)
        qf = dca_benchmark(qqq, dts)["V"].iloc[-1]
        ratio = s["final"] / qf
        nmu, nmx = null_ratios(dts, N, trail, wsafe, qf) if lab not in ("2015-26",) else (np.nan, np.nan)
        out[lab] = (ratio, s, nmu, nmx, lbl)
        p(f"    {lab:8}  {lbl:11} {ratio:5.2f}  {s['irr']*100:4.0f}%  {s['sharpe']:5.2f}  "
          f"{s['maxdd']*100:5.0f}%  |   {nmu:5.2f}    {nmx:5.2f}")
    return out

TABLES = {}
for vname, trail, wsafe in VARIANTS:
    for N in (10, 20):
        TABLES[(vname, N)] = full_table(vname, N, trail, wsafe)

pd.to_pickle({"comp_full": comp_full.astype(np.float32),
              "comp_po": comp_po.astype(np.float32),
              "comp_merged": comp_merged.astype(np.float32),
              "regime_on": regime_on, "fund_avail": fund_avail,
              "summary_min_era": summ, "tables": TABLES,
              "note": "Defended composite = rank-avg(coiled-quality, twin-agreement) among uptrend "
                      "names, Faber QQQ<10moMA -> route contribution to QQQ. Price-only fallback "
                      "(coiled=low-vol+near-high, momentum=twin price leg) for pre-2012. Variants: "
                      "A cut losers (trail-30%+trend), B ride winners (trend-break only if underwater). "
                      "Null = random selection under SAME regime defense (5 seeds)."},
             f"{HERE}/loop2_composite.pkl")
p(f"\nsaved loop2_composite.pkl  DONE t={time.time()-t0:.0f}s")
