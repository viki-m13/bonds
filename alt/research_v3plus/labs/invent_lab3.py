"""Invention loop iteration 3 — new uncorrelated streams. IS ONLY."""
import sys
sys.path.insert(0, "/home/user/bonds/alt")
import numpy as np
import pandas as pd
from sleeve_engine import backtest_weights, sharpe, cagr, max_dd, ann_vol

ETF = "/home/user/bonds/data/etfs/"
IS_END = pd.Timestamp("2018-12-31")

def load(t):
    df = pd.read_csv(ETF + f"{t}.csv", parse_dates=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"]).set_index("Date")
    return df[["Open", "Close"]].apply(pd.to_numeric, errors="coerce")

SPY = load("SPY")
DATES = SPY["Open"].dropna().index
DATES = DATES[DATES >= "2010-03-11"]

def rep(name, r):
    ri = r.loc[:IS_END]
    print(f"{name:40s} IS: SR {sharpe(ri):5.2f}  CAGR {cagr(ri)*100:6.1f}%  "
          f"Vol {ann_vol(ri)*100:5.1f}%  MDD {max_dd(ri)*100:6.1f}%")

def wfreeze(W, dow=2):
    reb = (pd.Series(W.index.dayofweek, index=W.index) == dow); reb.iloc[0] = True
    mask = pd.DataFrame(np.broadcast_to(reb.values[:, None], W.shape), index=W.index, columns=W.columns)
    return W.where(mask, np.nan).ffill().fillna(0.0)

# ===== IDEA A: NDX-vs-SPX spread momentum via LETF + inverse pair =====
def spread_mom(lb=63, dow=2, lev3=True):
    q = load("QQQ")["Close"].reindex(DATES).ffill(limit=3)
    s = SPY["Close"].reindex(DATES).ffill(limit=3)
    sig = (q.pct_change(lb) - s.pct_change(lb)).shift(1)
    longq, shorts = ("TQQQ", "SPXU") if lev3 else ("QLD", "SDS")
    longs, shortq = ("UPRO", "SQQQ") if lev3 else ("SSO", "QID")
    cols = list({longq, shorts, longs, shortq})
    W = pd.DataFrame(0.0, index=DATES, columns=cols)
    W.loc[sig > 0, longq] = 0.5
    W.loc[sig > 0, shorts] = 0.5
    W.loc[sig <= 0, longs] = 0.5
    W.loc[sig <= 0, shortq] = 0.5
    W = wfreeze(W, dow)
    opens = pd.DataFrame({c: load(c)["Open"].reindex(DATES) for c in cols})
    return backtest_weights(W, opens, 10.0)["ret"]

print("== IDEA A: NDX-SPX spread momentum (pair, gross 1.0) ==")
for lb in [21, 63, 126]:
    rep(f"spread mom lb={lb} 3x", spread_mom(lb))
# long-leader-only variant (no inverse leg; half gross in leader, half cash)
def spread_long_only(lb=63, dow=2):
    q = load("QQQ")["Close"].reindex(DATES).ffill(limit=3)
    s = SPY["Close"].reindex(DATES).ffill(limit=3)
    sig = (q.pct_change(lb) - s.pct_change(lb)).shift(1)
    up = (s > s.rolling(200).mean()).shift(1).fillna(False)
    W = pd.DataFrame(0.0, index=DATES, columns=["TQQQ", "UPRO"])
    W.loc[(sig > 0) & up, "TQQQ"] = 1.0
    W.loc[(sig <= 0) & up, "UPRO"] = 1.0
    W = wfreeze(W, dow)
    opens = pd.DataFrame({c: load(c)["Open"].reindex(DATES) for c in ["TQQQ", "UPRO"]})
    return backtest_weights(W, opens, 10.0)["ret"]
rep("leader-only (uptrend gate)", spread_long_only())

# ===== IDEA B: bond turn-of-month (duration rallies at month-end) =====
def bond_tom(vehicle="TMF", n_before=3, n_after=2):
    ym = DATES.to_period("M")
    pos = pd.Series(0.0, index=DATES)
    for p in ym.unique():
        m = DATES[ym == p]
        pos.loc[m[-n_before:]] = 1.0
        pos.loc[m[:n_after]] = 1.0
    W = pd.DataFrame({vehicle: pos})
    opens = pd.DataFrame({vehicle: load(vehicle)["Open"].reindex(DATES)})
    return backtest_weights(W, opens, 10.0)["ret"]

print("\n== IDEA B: bond turn-of-month ==")
for v in ["TMF", "TYD", "UBT"]:
    rep(f"bond TOM {v} (-3,+2)", bond_tom(v))
rep("bond TOM TMF (-4,+3)", bond_tom("TMF", 4, 3))

# ===== IDEA C: defensive risk-off basket (replaces cash when gate off) =====
def defensive_riskoff(dow=2):
    """When SPY < 200dma (risk-off): hold best-momentum of {TMF, UGL, XLP, XLU, IEF};
    else cash. This is the mirror-image of the risk book — a stream that only
    exists in risk-off periods."""
    s = SPY["Close"].reindex(DATES).ffill(limit=3)
    risk_off = (s < s.rolling(200).mean()).shift(1).fillna(False)
    basket = ["TMF", "UGL", "XLP", "XLU", "IEF"]
    closes = pd.DataFrame({t: load(t)["Close"].reindex(DATES).ffill(limit=3) for t in basket})
    mom = closes.pct_change(63).shift(1)
    ranks = mom.rank(axis=1, ascending=False, method="first")
    Wb = (ranks <= 2).astype(float) / 2.0
    W = Wb.mul(risk_off.astype(float), axis=0)
    W = wfreeze(W, dow)
    opens = pd.DataFrame({t: load(t)["Open"].reindex(DATES) for t in basket})
    return backtest_weights(W, opens, 7.0)["ret"]

print("\n== IDEA C: defensive risk-off basket ==")
rep("risk-off top2 of TMF/UGL/XLP/XLU/IEF", defensive_riskoff())

# ===== IDEA D: commodity cross-sectional momentum =====
def commodity_mom(K=1, dow=2):
    basket = {"GLD": "UGL", "USO": "UCO"}
    unlev = ["GLD", "USO", "SLV", "DBC", "CPER"]
    closes = pd.DataFrame({t: load(t)["Close"].reindex(DATES).ffill(limit=3) for t in unlev})
    mom = closes.pct_change(126).shift(1)
    above = (closes > closes.rolling(200).mean()).shift(1).fillna(False)
    score = mom.where(above & (mom > 0))
    ranks = score.rank(axis=1, ascending=False, method="first")
    pick = (ranks <= K).astype(float) / K
    # express GLD/USO via 2x; others unlevered
    cols = ["UGL", "UCO", "SLV", "DBC", "CPER"]
    W = pd.DataFrame(0.0, index=DATES, columns=cols)
    W["UGL"] = pick["GLD"]; W["UCO"] = pick["USO"]
    for t in ["SLV", "DBC", "CPER"]:
        W[t] = pick[t]
    W = wfreeze(W, dow)
    opens = pd.DataFrame({t: load(t)["Open"].reindex(DATES) for t in cols})
    return backtest_weights(W, opens, 10.0)["ret"]

print("\n== IDEA D: commodity x-sec momentum ==")
for K in [1, 2]:
    rep(f"commodity top-{K}", commodity_mom(K))

# ===== correlation of survivors vs existing blend raw =====
print("\n== corr with v3 raw blend (IS) ==")
prod_ret = pd.read_csv("/home/user/bonds/data/results/phoenix_production_returns.csv",
                       parse_dates=["Date"]).set_index("Date")["raw_ret"]
cands = {"SPREAD63": spread_mom(63), "BONDTOM": bond_tom("TMF"),
         "DEFRO": defensive_riskoff(), "CMDTY": commodity_mom(1)}
for n, r in cands.items():
    c = r.loc[:IS_END].corr(prod_ret.loc[:IS_END])
    print(f"  {n:10s} corr={c:+.2f}")
print("done (IS only)")
