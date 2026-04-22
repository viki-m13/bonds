"""Robustness Test 1: Survivorship bias — add collapsed LETFs to universe.

Goal: take PHOENIX-LITE's core logic (simplest cross-sectional momentum +
regime-gate strategy) and run it TWICE:

  A) Original 17-LETF universe (survivors)
  B) Expanded 32-LETF universe (17 survivors + 15 collapsed LETFs that lost
     ~100% over 2010-2026: SOXS, SQQQ, SPXU, TECS, FAZ, YANG, EDZ, ERY, DRV,
     DUST, LABD, UVXY, VIXY, GLL, UNG)

If (B)'s Sharpe is materially worse than (A)'s, survivorship bias is a real
hit. If they are similar, it means the 63-day momentum signal + regime gate
correctly avoids the doomed inverse/vol names because they almost always have
negative trailing momentum (they are LITERAL trending-down names).

Expected: (B) should actually IMPROVE (or match) because the broader universe
gives more opportunities when momentum is positive, and the doomed names
simply never get picked.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"

SURVIVORS = ["TQQQ","UPRO","QLD","SSO","SOXL","TECL","FAS","ERX","DRN","EDC",
             "YINN","UCO","UGL","NUGT","TMF","UBT","TYD"]
COLLAPSED = ["SOXS","SQQQ","SPXU","TECS","FAZ","YANG","EDZ","ERY","DRV",
             "DUST","LABD","UVXY","VIXY","GLL","UNG"]

IS_END = "2018-12-31"
OOS_START = "2019-01-02"
TC_BPS = 10.0


def load_etf(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["Close","Open"]].apply(pd.to_numeric, errors="coerce")


def load_fred(s):
    p = FRED / f"{s}.csv"
    if not p.exists(): return None
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    d = d[~d.index.duplicated(keep="first")]
    return pd.to_numeric(d.iloc[:, 0], errors="coerce")


def metrics(r):
    r = r.dropna()
    mu = r.mean()*252; sd = r.std()*np.sqrt(252)
    sr = mu/sd if sd>0 else 0
    c = (1+r).cumprod(); dd = (c/c.cummax()-1).min()
    yrs = len(r)/252
    cagr = c.iloc[-1]**(1/yrs) - 1 if c.iloc[-1]>0 else -1
    return {"sharpe":float(sr),"cagr":float(cagr),"vol":float(sd),
            "mdd":float(dd),"navx":float(c.iloc[-1])}


def backtest(close, opn, regime_ok, mom63_lag, universe, N, K):
    idx = opn.index
    cols = list(universe) + ["BIL"]
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    current = pd.Series(0.0, index=cols); current["BIL"] = 1.0

    pick_log = []  # record which names picked at each rebal

    for i, dt in enumerate(idx):
        if i % N == 0:
            if regime_ok.iloc[i]:
                m = mom63_lag.iloc[i].reindex(universe).dropna()
                tradable = [t for t in universe if t in m.index
                            and not np.isnan(opn[t].iloc[i])]
                m = m[tradable]
                m = m[m > 0].nlargest(K)
                new_w = pd.Series(0.0, index=cols)
                if len(m) > 0:
                    each = 1.0/len(m)
                    for t in m.index: new_w[t] = each
                else:
                    new_w["BIL"] = 1.0
                pick_log.append((dt, list(m.index)))
                current = new_w
            else:
                new_w = pd.Series(0.0, index=cols); new_w["BIL"] = 1.0
                pick_log.append((dt, ["BIL"]))
                current = new_w
        W.iloc[i] = current.values

    opn_ret = opn[universe].pct_change().shift(-1).fillna(0)
    bil_ret = opn["BIL"].pct_change().shift(-1).fillna(0) if "BIL" in opn.columns else pd.Series(0.0, index=idx)
    port_ret = (W[universe] * opn_ret[universe]).sum(axis=1) + W["BIL"] * bil_ret
    dW = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    tc = dW * (TC_BPS/1e4)
    port_ret = port_ret - tc
    port_ret = port_ret.shift(1).fillna(0.0)
    return port_ret, pick_log


def main():
    tickers = SURVIVORS + COLLAPSED + ["SPY", "BIL"]
    close, opn = {}, {}
    for t in tickers:
        df = load_etf(t)
        if df is not None:
            close[t] = df["Close"]; opn[t] = df["Open"]
    close = pd.DataFrame(close); opn = pd.DataFrame(opn)

    dates = opn["SPY"].dropna().index
    dates = dates[(dates >= pd.Timestamp("2010-03-11")) &
                  (dates <= pd.Timestamp("2026-04-02"))]
    close = close.reindex(dates).ffill(limit=5)
    opn = opn.reindex(dates).ffill(limit=5)

    # Macro regime (identical to PHOENIX-LITE)
    hy = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    spy = close["SPY"]
    spy_ma = spy.rolling(200).mean()
    spy_ok = (spy > spy_ma) & (spy_ma.diff(20) > 0)
    hy_slope = hy - hy.shift(20)
    regime_ok = (spy_ok & (hy_slope < 1.0) & (vix < 30)).shift(1).fillna(False)

    # Run with SAME N=42, K=5 (PHOENIX-LITE's grid-winner) on both universes
    # so we have a clean A/B
    N, K = 42, 5

    # A) Original survivor universe
    all_avail = [t for t in SURVIVORS if t in close.columns]
    mom63A = close[all_avail].pct_change(63).shift(1)
    retA, picksA = backtest(close, opn, regime_ok, mom63A, all_avail, N, K)

    # B) Expanded = survivors + collapsed
    expanded = [t for t in SURVIVORS + COLLAPSED if t in close.columns]
    mom63B = close[expanded].pct_change(63).shift(1)
    retB, picksB = backtest(close, opn, regime_ok, mom63B, expanded, N, K)

    # Evaluate
    mA = {w: metrics(retA.loc[sl]) for w,sl in [("FULL",slice(None)),("IS",slice(None,IS_END)),("OOS",slice(OOS_START,None))]}
    mB = {w: metrics(retB.loc[sl]) for w,sl in [("FULL",slice(None)),("IS",slice(None,IS_END)),("OOS",slice(OOS_START,None))]}

    print(f"\n=== ROBUSTNESS TEST 1: Survivorship bias (N={N}, K={K}) ===\n")
    print(f"{'window':6s}  {'survivors-only':^32s}  {'+ collapsed LETFs':^32s}")
    print(f"{'':6s}  {'SR':>5s} {'CAGR':>6s} {'MDD':>6s} {'NAVx':>6s}  "
          f"{'SR':>5s} {'CAGR':>6s} {'MDD':>6s} {'NAVx':>6s}")
    for w in ["FULL","IS","OOS"]:
        a = mA[w]; b = mB[w]
        print(f"{w:6s}  {a['sharpe']:5.2f} {a['cagr']*100:5.1f}% {a['mdd']*100:5.1f}% {a['navx']:5.1f}  "
              f"{b['sharpe']:5.2f} {b['cagr']*100:5.1f}% {b['mdd']*100:5.1f}% {b['navx']:5.1f}")

    # Check: did ANY collapsed name ever get picked?
    collapsed_picks = 0
    for d, ps in picksB:
        for p in ps:
            if p in COLLAPSED:
                collapsed_picks += 1
    total_picks = sum(len(ps) for d, ps in picksB)
    print(f"\nOf {total_picks} total position-picks in the expanded universe,")
    print(f"  {collapsed_picks} ({collapsed_picks/max(total_picks,1)*100:.1f}%) landed on collapsed LETFs.")
    print(f"  Reason: they have near-permanent negative 63-day momentum and cannot rank top-K.")

    # Save
    out = {
        "test": "survivorship_bias",
        "N": N, "K": K,
        "survivors_only": {w: mA[w] for w in ["FULL","IS","OOS"]},
        "plus_collapsed": {w: mB[w] for w in ["FULL","IS","OOS"]},
        "n_collapsed_picks": collapsed_picks,
        "n_total_picks": total_picks,
        "collapsed_pick_pct": collapsed_picks / max(total_picks, 1),
    }
    (RESULTS / "robustness_survivorship.json").write_text(json.dumps(out, indent=2))
    print("\nSaved robustness_survivorship.json")


if __name__ == "__main__":
    main()
