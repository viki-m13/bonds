"""Rule-based QUANTUM replacement — no ML, pure rules.

Replaces the XGBoost ML sleeve with a transparent rule-based composite-score
strategy that captures the same information (momentum, vol, macro). Goal:
match or beat QUANTUM's OOS Sharpe 0.87 without training any model.

Composite score per LETF at close[t-1]:
    score = z(21d return) + z(63d return) + z(252d return)
          + z(21d return / 21d vol)      ← short-horizon Sharpe
          + z(63d return / 63d vol)      ← medium-horizon Sharpe

Macro filters (same as other sleeves):
    SPY > 200dma AND HY OAS 20d slope < 1.0 AND VIX < 30

At rebalance (every 21 trading days):
    If macro OK: pick top-3 by composite score; equal weight.
    Else: 100% BIL.

Universe: same 17 LETFs as QUANTUM.
Execution: next-open fill, 10 bps/side TC.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
R = ROOT / "data/results"

UNIVERSE = ["TQQQ","UPRO","QLD","SSO","SOXL","TECL","FAS","ERX","DRN","EDC",
            "YINN","UCO","UGL","NUGT","TMF","UBT","TYD"]
IS_END = pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-02")
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


def zscore(df, axis=1):
    """Cross-sectional z-score within each row."""
    mu = df.mean(axis=axis)
    sd = df.std(axis=axis)
    sd = sd.replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0).fillna(0)


def metrics(r):
    r = r.dropna()
    if len(r) == 0: return {"sharpe":0,"cagr":0,"mdd":0,"vol":0,"sortino":0}
    mu = r.mean()*252; sd = r.std()*np.sqrt(252)
    sr = mu/sd if sd>0 else 0
    c = (1+r).cumprod(); dd = (c/c.cummax()-1).min()
    yrs = len(r)/252
    cagr = c.iloc[-1]**(1/yrs) - 1 if c.iloc[-1]>0 else -1
    neg = r[r<0]
    sortino = mu/(neg.std()*np.sqrt(252)) if len(neg)>0 and neg.std()>0 else 0
    return {"sharpe":float(sr),"cagr":float(cagr),"mdd":float(dd),"vol":float(sd),"sortino":float(sortino)}


def main():
    close, opn = {}, {}
    for t in UNIVERSE + ["SPY", "BIL"]:
        df = load_etf(t)
        if df is not None:
            close[t] = df["Close"]; opn[t] = df["Open"]
    close = pd.DataFrame(close); opn = pd.DataFrame(opn)

    dates = opn["SPY"].dropna().index
    dates = dates[(dates >= pd.Timestamp("2010-03-11")) & (dates <= pd.Timestamp("2026-04-02"))]
    close = close.reindex(dates).ffill(limit=5)
    opn = opn.reindex(dates).ffill(limit=5)

    # Macro regime
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    hy = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    spy = close["SPY"]
    spy_ma = spy.rolling(200).mean()
    spy_ok = (spy > spy_ma) & (spy_ma.diff(20) > 0)
    hy_slope = hy - hy.shift(20)
    regime = (spy_ok & (hy_slope < 1.0) & (vix < 30)).shift(1).fillna(False)

    # Features
    mom21  = close[UNIVERSE].pct_change(21).shift(1)
    mom63  = close[UNIVERSE].pct_change(63).shift(1)
    mom252 = close[UNIVERSE].pct_change(252).shift(1)
    rets = close[UNIVERSE].pct_change().fillna(0)
    vol21  = rets.rolling(21).std().shift(1) * np.sqrt(252)
    vol63  = rets.rolling(63).std().shift(1) * np.sqrt(252)
    sh21 = (mom21  * (252/21))  / vol21.replace(0, np.nan)
    sh63 = (mom63  * (252/63))  / vol63.replace(0, np.nan)

    # Cross-sectional z-scores
    z_m21  = zscore(mom21,  axis=1)
    z_m63  = zscore(mom63,  axis=1)
    z_m252 = zscore(mom252, axis=1)
    z_sh21 = zscore(sh21,   axis=1)
    z_sh63 = zscore(sh63,   axis=1)
    score = z_m21 + z_m63 + z_m252 + z_sh21 + z_sh63

    # Strategy: rebalance every 21 days, pick top-3, macro gate, 21d cadence
    N = 21; K = 3
    cols = UNIVERSE + ["BIL"]
    W = pd.DataFrame(0.0, index=dates, columns=cols)
    current = pd.Series(0.0, index=cols); current["BIL"] = 1.0

    for i, dt in enumerate(dates):
        if i % N == 0:
            if regime.iloc[i]:
                s = score.iloc[i].dropna()
                # Only tradable
                tradable = [t for t in UNIVERSE if t in s.index and not np.isnan(opn[t].iloc[i])]
                s = s[tradable]
                # Require positive momentum at all 3 horizons to avoid doomed names
                pos = (mom21.iloc[i].reindex(tradable) > 0) & \
                      (mom63.iloc[i].reindex(tradable) > 0) & \
                      (mom252.iloc[i].reindex(tradable) > 0)
                s = s[pos]
                top = s.nlargest(K)
                new_w = pd.Series(0.0, index=cols)
                if len(top) > 0:
                    each = 1.0/len(top)
                    for t in top.index: new_w[t] = each
                else:
                    new_w["BIL"] = 1.0
                current = new_w
            else:
                current = pd.Series(0.0, index=cols); current["BIL"] = 1.0
        W.iloc[i] = current.values

    # Returns
    opn_ret = opn[UNIVERSE].pct_change().shift(-1).fillna(0)
    bil_ret = opn["BIL"].pct_change().shift(-1).fillna(0) if "BIL" in opn.columns else pd.Series(0.0, index=dates)
    port_ret = (W[UNIVERSE] * opn_ret[UNIVERSE]).sum(axis=1) + W["BIL"] * bil_ret
    dW = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    tc = dW * (TC_BPS/1e4)
    port_ret = (port_ret - tc).shift(1).fillna(0.0)

    m_full = metrics(port_ret)
    m_is   = metrics(port_ret.loc[:IS_END])
    m_oos  = metrics(port_ret.loc[OOS_START:])

    # Compare vs QUANTUM original
    qua = pd.read_csv(R/"quantum_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    mQ_full = metrics(qua); mQ_is = metrics(qua.loc[:IS_END]); mQ_oos = metrics(qua.loc[OOS_START:])

    print("Rule-based QUANTUM analog (no ML):")
    print(f"  FULL: SR {m_full['sharpe']:5.2f}  CAGR {m_full['cagr']*100:5.1f}%  MDD {m_full['mdd']*100:5.1f}%  Sortino {m_full['sortino']:4.2f}")
    print(f"  IS  : SR {m_is['sharpe']:5.2f}  CAGR {m_is['cagr']*100:5.1f}%  MDD {m_is['mdd']*100:5.1f}%")
    print(f"  OOS : SR {m_oos['sharpe']:5.2f}  CAGR {m_oos['cagr']*100:5.1f}%  MDD {m_oos['mdd']*100:5.1f}%")
    print()
    print("Original QUANTUM (XGBoost ML):")
    print(f"  FULL: SR {mQ_full['sharpe']:5.2f}  CAGR {mQ_full['cagr']*100:5.1f}%  MDD {mQ_full['mdd']*100:5.1f}%")
    print(f"  IS  : SR {mQ_is['sharpe']:5.2f}  CAGR {mQ_is['cagr']*100:5.1f}%  MDD {mQ_is['mdd']*100:5.1f}%")
    print(f"  OOS : SR {mQ_oos['sharpe']:5.2f}  CAGR {mQ_oos['cagr']*100:5.1f}%  MDD {mQ_oos['mdd']*100:5.1f}%")

    # Now blend rule-based QUANTUM with VAN/ORI/HEL
    van = pd.read_csv(R/"vanguard_returns.csv", parse_dates=[0], index_col=0)["net_ret"]
    ori = pd.read_csv(R/"orion_returns.csv", parse_dates=["Date"]).set_index("Date")["orion"]
    hel = pd.read_csv(R/"helios_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]

    df = pd.concat({"V":van,"O":ori,"H":hel,"R":port_ret}, axis=1).reindex(dates).fillna(0)

    # IS inv-vol
    iv = 1.0 / df.loc[:IS_END].std()
    iv = iv.replace([np.inf, -np.inf], np.nan).fillna(0)
    w = (iv/iv.sum())
    raw = pd.Series(df.values @ w.values, index=df.index)

    def apply_overlay(r, dd_win=252, dd_floor=-0.10, vol_win=60, vol_pct=0.99):
        cum = (1+r).cumprod()
        hwm = cum.rolling(dd_win, min_periods=30).max()
        dd = cum/hwm - 1
        dd_mult = (1.0 + dd/dd_floor).clip(lower=0.0, upper=1.0).shift(1).fillna(1.0)
        rv = r.rolling(vol_win).std()
        rv_thr = rv.rolling(252, min_periods=60).quantile(vol_pct)
        vol_ok = (rv <= rv_thr).shift(1).fillna(True).astype(float)
        vol_mult = vol_ok + (1-vol_ok)*0.5
        mult = (dd_mult * vol_mult).clip(lower=0.0, upper=1.0)
        return r*mult, mult

    ret_ov, mult = apply_overlay(raw)

    print()
    print(f"4-sleeve VAN+ORI+HEL+RULE weights: V={w.V:.3f} O={w.O:.3f} H={w.H:.3f} R={w.R:.3f}")
    print()
    print("4-sleeve blend (replacing ML QUANTUM with rule-based):")
    for name, r in [("FULL", ret_ov), ("IS", ret_ov.loc[:IS_END]), ("OOS", ret_ov.loc[OOS_START:])]:
        m = metrics(r)
        print(f"  {name:4s}: SR {m['sharpe']:5.2f}  CAGR {m['cagr']*100:5.1f}%  MDD {m['mdd']*100:5.1f}%  Sortino {m['sortino']:4.2f}")

    out = {
        "standalone": {"full":m_full,"is":m_is,"oos":m_oos},
        "vs_quantum_ml": {"full":mQ_full,"is":mQ_is,"oos":mQ_oos},
        "blend_weights": w.to_dict(),
        "blend_overlayed": {
            "full": metrics(ret_ov),
            "is": metrics(ret_ov.loc[:IS_END]),
            "oos": metrics(ret_ov.loc[OOS_START:]),
        },
    }
    (R/"rulebased_quantum.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"Date": port_ret.index, "ret": port_ret.values}).to_csv(
        R/"rulebased_quantum_returns.csv", index=False)
    print(f"\nSaved rulebased_quantum.json and rulebased_quantum_returns.csv")


if __name__ == "__main__":
    main()
