"""Invention loop iteration 1 — IS ONLY (2010-2018). Four idea families."""
import sys
sys.path.insert(0, "/home/user/bonds/alt")
import numpy as np
import pandas as pd
from sleeve_engine import backtest_weights, sharpe, cagr, max_dd, ann_vol

ETF = "/home/user/bonds/data/etfs/"
FRED = "/home/user/bonds/data/fred/"
IS_END = pd.Timestamp("2018-12-31")

def load(t):
    df = pd.read_csv(ETF + f"{t}.csv", parse_dates=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"]).set_index("Date")
    return df[["Open", "Close"]].apply(pd.to_numeric, errors="coerce")

def fred(s):
    d = pd.read_csv(FRED + f"{s}.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    return pd.to_numeric(d.iloc[:, 0], errors="coerce")

SPY = load("SPY")
DATES = SPY["Open"].dropna().index
DATES = DATES[DATES >= "2010-03-11"]

def rep(name, r):
    ri = r.loc[:IS_END]
    print(f"{name:42s} IS: SR {sharpe(ri):5.2f}  CAGR {cagr(ri)*100:6.1f}%  "
          f"Vol {ann_vol(ri)*100:5.1f}%  MDD {max_dd(ri)*100:6.1f}%")

# ============ IDEA 1: leverage-tier rotation (vol-drag harvesting) ============
# Same trend signal (QQQ > 200dma & 63d mom > 0), but the VEHICLE depends on
# realized vol tier: low vol -> TQQQ, mid -> QLD, high -> QQQ, gate-off -> cash.
def tier_rotation(v1=0.18, v2=0.28, sig="QQQ", tiers=("TQQQ","QLD","QQQ"), dow=2):
    q = load(sig)["Close"].reindex(DATES).ffill(limit=3)
    vol = (q.pct_change().rolling(21).std() * np.sqrt(252)).shift(1)
    trend = ((q > q.rolling(200).mean()) & (q.pct_change(63) > 0)).shift(1).fillna(False)
    cols = list(tiers)
    W = pd.DataFrame(0.0, index=DATES, columns=cols)
    W.loc[trend & (vol < v1), tiers[0]] = 1.0
    W.loc[trend & (vol >= v1) & (vol < v2), tiers[1]] = 1.0
    W.loc[trend & (vol >= v2), tiers[2]] = 1.0
    reb = (pd.Series(DATES.dayofweek, index=DATES) == dow); reb.iloc[0] = True
    mask = pd.DataFrame(np.broadcast_to(reb.values[:, None], W.shape), index=W.index, columns=W.columns)
    W = W.where(mask, np.nan).ffill().fillna(0.0)
    opens = pd.DataFrame({c: load(c)["Open"].reindex(DATES) for c in cols})
    return backtest_weights(W, opens, 10.0)["ret"]

print("== IDEA 1: leverage-tier rotation ==")
for v1, v2 in [(0.15, 0.25), (0.18, 0.28), (0.20, 0.35)]:
    rep(f"tier QQQ v1={v1} v2={v2}", tier_rotation(v1, v2))
# daily switching instead of weekly
def tier_daily(v1=0.18, v2=0.28):
    q = load("QQQ")["Close"].reindex(DATES).ffill(limit=3)
    vol = (q.pct_change().rolling(21).std() * np.sqrt(252)).shift(1)
    trend = ((q > q.rolling(200).mean()) & (q.pct_change(63) > 0)).shift(1).fillna(False)
    cols = ["TQQQ", "QLD", "QQQ"]
    W = pd.DataFrame(0.0, index=DATES, columns=cols)
    W.loc[trend & (vol < v1), "TQQQ"] = 1.0
    W.loc[trend & (vol >= v1) & (vol < v2), "QLD"] = 1.0
    W.loc[trend & (vol >= v2), "QQQ"] = 1.0
    opens = pd.DataFrame({c: load(c)["Open"].reindex(DATES) for c in cols})
    return backtest_weights(W, opens, 10.0)["ret"]
rep("tier QQQ daily v1=.18 v2=.28", tier_daily())

# ============ IDEA 2: breadth/credit confirmation on the trend book ============
# ORION-style top-4 momentum book, but gated by sector breadth (fraction of
# XL* sectors above 200dma) and credit (HYG vs IEF 63d relative momentum).
def breadth_book(bth=0.5, use_credit=True, dow=2):
    sectors = ["XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLU","XLB"]
    sc = pd.DataFrame({s: load(s)["Close"].reindex(DATES).ffill(limit=3) for s in sectors})
    breadth = (sc > sc.rolling(200).mean()).mean(axis=1).shift(1)
    hyg = load("HYG")["Close"].reindex(DATES).ffill(limit=3)
    ief = load("IEF")["Close"].reindex(DATES).ffill(limit=3)
    credit_ok = (hyg.pct_change(63) - ief.pct_change(63)).shift(1) > -0.02
    risk = ["TQQQ","UPRO","SOXL","TECL","FAS","ERX","EDC","YINN","DRN","UCO","QLD","SSO"]
    closes = pd.DataFrame({t: load(t)["Close"].reindex(DATES).ffill(limit=3) for t in risk})
    mom = closes.pct_change(252).shift(1)
    above = (closes > closes.rolling(200).mean()).shift(1).fillna(False)
    score = mom.where(above)
    ranks = score.rank(axis=1, ascending=False, method="first")
    Wr = (ranks <= 4).astype(float) / 4.0
    gate = (breadth > bth)
    if use_credit:
        gate = gate & credit_ok
    W = Wr.mul(gate.astype(float), axis=0)
    reb = (pd.Series(DATES.dayofweek, index=DATES) == dow); reb.iloc[0] = True
    mask = pd.DataFrame(np.broadcast_to(reb.values[:, None], W.shape), index=W.index, columns=W.columns)
    W = W.where(mask, np.nan).ffill().fillna(0.0)
    opens = pd.DataFrame({t: load(t)["Open"].reindex(DATES) for t in risk})
    return backtest_weights(W, opens, 7.0)["ret"]

print("\n== IDEA 2: breadth/credit-gated momentum book ==")
for bth in [0.4, 0.5, 0.6]:
    rep(f"breadth>{bth}+credit", breadth_book(bth, True))
rep("breadth>0.5 no credit", breadth_book(0.5, False))
rep("VIX-gate reference (ORION-ish)", breadth_book(0.0, False))

# ============ IDEA 3: vol-scaled reversal entries ============
def rev_scaled(z_th=-1.0, hold=5, scale_by_z=True):
    pairs = {"QQQ": "TQQQ", "SPY": "UPRO", "SMH": "SOXL"}
    closes = pd.DataFrame({u: load(u)["Close"].reindex(DATES).ffill(limit=3) for u in pairs})
    r5 = closes.pct_change(5)
    z = (r5 - r5.rolling(60).mean()) / r5.rolling(60).std()
    above = closes > closes.rolling(200).mean()
    trig = ((z < z_th) & above).shift(1).fillna(False)
    depth = (-z).clip(lower=0).shift(1).fillna(0)  # entry size grows with dip depth
    sig = trig.astype(float)
    if scale_by_z:
        sig = sig * (depth / 2.0).clip(upper=1.5)
    held = sig.rolling(hold, min_periods=1).max()
    tot = held.sum(axis=1)
    W = held.div(tot.where(tot > 0), axis=0).mul(tot.clip(upper=1.0), axis=0).fillna(0.0)
    W = W.rename(columns=pairs)
    opens = pd.DataFrame({l: load(l)["Open"].reindex(DATES) for l in pairs.values()})
    return backtest_weights(W, opens, 10.0)["ret"]

print("\n== IDEA 3: vol-scaled reversal ==")
rep("rev base (v3)", rev_scaled(scale_by_z=False))
rep("rev z-scaled", rev_scaled(scale_by_z=True))

# ============ IDEA 4: staggered weekly tranches on ORION-style book ============
def stagger_book(n_tranches=5):
    risk = ["TQQQ","UPRO","SOXL","TECL","FAS","ERX","EDC","YINN","DRN","UCO","QLD","SSO"]
    closes = pd.DataFrame({t: load(t)["Close"].reindex(DATES).ffill(limit=3) for t in risk})
    vix = fred("VIXCLS").reindex(DATES).ffill()
    gate = (vix < 30).shift(1).fillna(False)
    mom = closes.pct_change(252).shift(1)
    above = (closes > closes.rolling(200).mean()).shift(1).fillna(False)
    score = mom.where(above)
    ranks = score.rank(axis=1, ascending=False, method="first")
    Wr = ((ranks <= 4).astype(float) / 4.0).mul(gate.astype(float), axis=0)
    tranches = []
    for dow in range(n_tranches):
        reb = (pd.Series(DATES.dayofweek, index=DATES) == dow); reb.iloc[0] = True
        mask = pd.DataFrame(np.broadcast_to(reb.values[:, None], Wr.shape), index=Wr.index, columns=Wr.columns)
        tranches.append(Wr.where(mask, np.nan).ffill().fillna(0.0))
    W = sum(tranches) / n_tranches
    opens = pd.DataFrame({t: load(t)["Open"].reindex(DATES) for t in risk})
    return backtest_weights(W, opens, 7.0)["ret"]

print("\n== IDEA 4: staggered tranches ==")
rep("single Wednesday tranche", stagger_book(1) if False else breadth_book(0.0, False))
rep("5 staggered tranches", stagger_book(5))
print("\ndone (IS only)")
