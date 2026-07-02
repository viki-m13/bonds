"""PHOENIX v3 research lab — IS-ONLY design (2010-2018; crypto active 2015+).

Protocol: every design decision (candidate inclusion, parameter variant,
blend weights, overlay params) is made from IS metrics printed here.
OOS (2019+) is NOT printed by this script. A separate one-shot script
evaluates the final chosen configuration OOS once.
"""
import sys
sys.path.insert(0, "/home/user/bonds/alt")
import numpy as np
import pandas as pd
from sleeve_engine import backtest_weights, sharpe, cagr, max_dd, ann_vol

ROOT = "/home/user/bonds/"
ETF = ROOT + "data/etfs/"
FRED = ROOT + "data/fred/"
R = ROOT + "data/results/"

IS_START, IS_END = "2010-03-11", "2018-12-31"
ERC_START = "2015-01-02"   # window where all sleeves incl crypto are active

def load_etf(t):
    df = pd.read_csv(ETF + f"{t}.csv", parse_dates=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"]).set_index("Date")
    return df[["Open", "Close"]].apply(pd.to_numeric, errors="coerce")

def load_fred(s):
    d = pd.read_csv(FRED + f"{s}.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    d = d[~d.index.duplicated(keep="first")]
    return pd.to_numeric(d.iloc[:, 0], errors="coerce")

SPY = load_etf("SPY")
DATES = SPY["Open"].dropna().index
DATES = DATES[DATES >= IS_START]

def is_only(r):
    return r.loc[:IS_END]

def rep(name, r):
    ri = is_only(r)
    print(f"  {name:28s} IS: SR {sharpe(ri):5.2f}  CAGR {cagr(ri)*100:6.1f}%  "
          f"Vol {ann_vol(ri)*100:5.1f}%  MDD {max_dd(ri)*100:6.1f}%  n={ri.dropna().shape[0]}")

# ---------------------------------------------------------------- fixed sleeves
van = pd.read_csv(R + "vanguard_returns.csv", parse_dates=[0], index_col=0)["net_ret"]
ori = pd.read_csv(R + "orion_returns.csv", parse_dates=["Date"]).set_index("Date")["orion"]
hel = pd.read_csv(R + "helios_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
cry = pd.read_csv(R + "crypto_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]

print("== Fixed sleeves (already corrected, unified convention) ==")
for n, s in [("VANGUARD", van), ("ORION", ori), ("HELIOS", hel), ("CRYPTO", cry)]:
    rep(n, s)

# ---------------------------------------------------------------- helpers
def weekly_freeze(W, dow=2):
    idx = W.index
    reb = (pd.Series(idx.dayofweek, index=idx) == dow)
    reb.iloc[0] = True
    mask = pd.DataFrame(np.broadcast_to(reb.values[:, None], W.shape),
                        index=W.index, columns=W.columns)
    return W.where(mask, np.nan).ffill().fillna(0.0)

# ---------------------------------------------------------------- C1 BONDLS
def bondls(mom_lb=126, short_vehicle="TMV", dow=2):
    """Duration trend long/short: TLT momentum sign -> TMF or TMV (or cash)."""
    tlt = load_etf("TLT")["Close"].reindex(DATES).ffill(limit=3)
    tmf_o = load_etf("TMF")["Open"].reindex(DATES)
    shv_o = load_etf(short_vehicle)["Open"].reindex(DATES) if short_vehicle else None
    mom = tlt.pct_change(mom_lb).shift(1)
    cols = ["TMF"] + ([short_vehicle] if short_vehicle else [])
    W = pd.DataFrame(0.0, index=DATES, columns=cols)
    W.loc[mom > 0, "TMF"] = 1.0
    if short_vehicle:
        W.loc[mom < 0, short_vehicle] = 1.0
    W = weekly_freeze(W, dow)
    opens = pd.DataFrame({"TMF": tmf_o})
    if short_vehicle:
        opens[short_vehicle] = shv_o
    return backtest_weights(W, opens, 10.0)["ret"]

print("\n== C1 BONDLS variants ==")
for lb in [63, 126, 252]:
    for shv in ["TMV", "TBT", None]:
        rep(f"bondls lb={lb} short={shv}", bondls(lb, shv))

# ---------------------------------------------------------------- C2 REVERSAL
def reversal(z_th=-1.2, hold=5, zwin=60, K=2, sig_lb=5):
    """Dip-buy in uptrend: underlying 5d-return z < z_th while above 200dma
    -> hold matched LETF `hold` days. Equal weight across active, max K."""
    pairs = {"QQQ": "TQQQ", "SPY": "UPRO", "SMH": "SOXL"}
    closes = pd.DataFrame({u: load_etf(u)["Close"].reindex(DATES).ffill(limit=3) for u in pairs})
    opens = pd.DataFrame({l: load_etf(l)["Open"].reindex(DATES) for l in pairs.values()})
    r5 = closes.pct_change(sig_lb)
    z = (r5 - r5.rolling(zwin).mean()) / r5.rolling(zwin).std()
    above = closes > closes.rolling(200).mean()
    trigger = ((z < z_th) & above).shift(1).fillna(False)   # decision t uses close t-1
    # hold for `hold` days after trigger
    sig = trigger.astype(float)
    held = sig.rolling(hold, min_periods=1).max()
    # cap K names, equal weight
    n_active = held.sum(axis=1).clip(upper=K).replace(0, np.nan)
    Wu = held.div(held.sum(axis=1).replace(0, np.nan), axis=0).mul(n_active / n_active, axis=0).fillna(0.0)
    scale = (held.sum(axis=1).clip(upper=K) / held.sum(axis=1).replace(0, np.nan)).fillna(0.0)
    Wu = held.mul(scale, axis=0)  # each active name gets <=1/1, total <= K... normalize:
    tot = Wu.sum(axis=1).replace(0, np.nan)
    Wu = Wu.div(tot, axis=0).mul(tot.clip(upper=1.0), axis=0).fillna(0.0)
    W = Wu.rename(columns=pairs)
    return backtest_weights(W, opens, 10.0)["ret"]

print("\n== C2 REVERSAL variants ==")
for z_th in [-1.0, -1.5]:
    for hold in [3, 5, 10]:
        rep(f"reversal z<{z_th} hold={hold}", reversal(z_th, hold))

# ---------------------------------------------------------------- C3 TOM
def tom(vehicle="QLD", n_before=4, n_after=3, trend_filter=True):
    """Turn-of-month: long `vehicle` last n_before + first n_after trading
    days of each month; optional SPY 200dma filter; else cash."""
    opens = pd.DataFrame({vehicle: load_etf(vehicle)["Open"].reindex(DATES)})
    idx = DATES
    ym = idx.to_period("M")
    pos = pd.Series(0.0, index=idx)
    for p in ym.unique():
        m = idx[ym == p]
        pos.loc[m[-n_before:]] = 1.0
        pos.loc[m[:n_after]] = 1.0
    # decision-dated: the "day t in window" decision is known from the calendar
    # (no market data), but weights must still be decided at close[t-1]:
    # calendar is deterministic -> W[t] usable as is.
    if trend_filter:
        spyc = SPY["Close"].reindex(idx).ffill(limit=3)
        ok = (spyc > spyc.rolling(200).mean()).shift(1).fillna(False)
        pos = pos.where(ok, 0.0)
    W = pd.DataFrame({vehicle: pos})
    return backtest_weights(W, opens, 10.0)["ret"]

print("\n== C3 TOM variants ==")
for v in ["QLD", "TQQQ", "SSO"]:
    for tf in [True, False]:
        rep(f"tom {v} filt={tf}", tom(v, trend_filter=tf))

# ---------------------------------------------------------------- C4 CONVEX
def convex(hy_th=0.3, basket=(("TMF", 0.5), ("SQQQ", 0.5))):
    """Crisis basket when SPY<200dma & 200dma falling & HY widening; else cash."""
    spyc = SPY["Close"].reindex(DATES).ffill(limit=3)
    ma = spyc.rolling(200).mean()
    hy = load_fred("BAMLH0A0HYM2").reindex(DATES).ffill()
    stress = ((spyc < ma) & (ma.diff(20) < 0) & ((hy - hy.shift(20)) > hy_th)).shift(1).fillna(False)
    cols = [b[0] for b in basket]
    opens = pd.DataFrame({c: load_etf(c)["Open"].reindex(DATES) for c in cols})
    W = pd.DataFrame(0.0, index=DATES, columns=cols)
    for c, w in basket:
        W.loc[stress, c] = w
    W = weekly_freeze(W, dow=None) if False else W  # daily gate, no freeze
    return backtest_weights(W, opens, 10.0)["ret"]

print("\n== C4 CONVEX variants ==")
rep("convex TMF/SQQQ", convex())
rep("convex TMF only", convex(basket=(("TMF", 1.0),)))
rep("convex SQQQ only", convex(basket=(("SQQQ", 1.0),)))

# ---------------------------------------------------------------- C5 SVXY carry
def svxy_carry(vix_th=20.0):
    v = load_fred("VIXCLS").reindex(DATES).ffill()
    spyc = SPY["Close"].reindex(DATES).ffill(limit=3)
    ok = ((spyc > spyc.rolling(200).mean()) & (v < vix_th) & (v.diff(5) < 0)).shift(1).fillna(False)
    opens = pd.DataFrame({"SVXY": load_etf("SVXY")["Open"].reindex(DATES)})
    W = pd.DataFrame({"SVXY": ok.astype(float)})
    return backtest_weights(W, opens, 10.0)["ret"]

print("\n== C5 SVXY carry (2011-10+) ==")
for th in [18, 20, 25]:
    rep(f"svxy vix<{th}", svxy_carry(th))

print("\ndone (IS only — no OOS numbers printed)")
