"""PHOENIX v2 — with and without a crypto sleeve.

Build a simple CRYPTO sleeve (TSMOM on {GBTC, ETHE} with macro gate; weekly
rebal; hold BIL when off), then blend it as a 5th sleeve into PHOENIX v2.

CRYPTO sleeve design:
  - Universe: GBTC (Grayscale BTC, start 2015-05), ETHE (Grayscale ETH, 2019-06)
  - Signal: 63-day return. Hold if positive, else cash.
  - Macro gate: SPY > 200dma & HY-OAS slope < 1.0 (same as other sleeves)
  - Execute at open, 10 bps/side TC
  - Before GBTC exists (2010-2015): 100% cash

Compare three variants:
  A) No crypto — existing 4-sleeve PHOENIX v2 (our benchmark)
  B) + Crypto @ IS inv-vol weight (re-fit on IS with 5 sleeves)
  C) + Crypto @ 10% fixed cap (cap crypto at 10% to avoid dominance)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
R = ROOT / "data/results"

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


def metrics(r):
    r = r.dropna()
    if len(r) == 0: return {"sharpe":0,"cagr":0,"mdd":0,"vol":0}
    mu = r.mean()*252; sd = r.std()*np.sqrt(252)
    sr = mu/sd if sd>0 else 0
    c = (1+r).cumprod(); dd = (c/c.cummax()-1).min()
    yrs = len(r)/252
    cagr = c.iloc[-1]**(1/yrs) - 1 if c.iloc[-1]>0 else -1
    return {"sharpe":float(sr),"cagr":float(cagr),"mdd":float(dd),"vol":float(sd)}


def build_crypto_weights(dates, close, opn, regime_ok):
    """Construct daily target-weight DataFrame for the CRYPTO sleeve.

    Universe = {GBTC, ETHE} + BIL. Weights sum to 1.0. Weekly Friday rebal
    of 63d-momentum, gated by regime; falls back to BIL when no positive
    mom or gate is off.
    """
    universe = [t for t in ["GBTC", "ETHE"] if t in close.columns]
    if not universe:
        cols = ["BIL"]
        W = pd.DataFrame(1.0, index=dates, columns=cols)
        return W

    mom63 = close[universe].pct_change(63).shift(1)
    cols = universe + ["BIL"]
    W = pd.DataFrame(0.0, index=dates, columns=cols)
    current = pd.Series(0.0, index=cols); current["BIL"] = 1.0
    for i, dt in enumerate(dates):
        is_fri = dt.dayofweek == 4
        if is_fri or i == 0:
            if regime_ok.iloc[i]:
                m = mom63.iloc[i].dropna()
                tradable = [t for t in universe
                            if t in m.index and not np.isnan(opn[t].iloc[i])]
                m = m[tradable]
                m = m[m > 0]
                new_w = pd.Series(0.0, index=cols)
                if len(m) > 0:
                    each = 1.0 / len(m)
                    for t in m.index:
                        new_w[t] = each
                else:
                    new_w["BIL"] = 1.0
                current = new_w
            else:
                current = pd.Series(0.0, index=cols); current["BIL"] = 1.0
        W.iloc[i] = current.values
    return W


def build_crypto_sleeve(dates, close, opn, regime_ok):
    """Backtest-compatible: returns daily port_ret (open-to-open, lagged)."""
    W = build_crypto_weights(dates, close, opn, regime_ok)
    universe = [c for c in W.columns if c != "BIL"]
    if not universe:
        return pd.Series(0.0, index=dates)
    opn_ret = pd.DataFrame(0.0, index=dates, columns=universe)
    for u in universe:
        if u in opn.columns:
            opn_ret[u] = opn[u].pct_change().shift(-1).fillna(0)
    bil_ret = opn["BIL"].pct_change().shift(-1).fillna(0) if "BIL" in opn.columns else pd.Series(0.0, index=dates)
    port_ret = (W[universe] * opn_ret[universe]).sum(axis=1) + W["BIL"] * bil_ret
    dW = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    tc = dW * (TC_BPS / 1e4)
    port_ret = port_ret - tc
    return port_ret.shift(1).fillna(0.0)


def build_weights(use_live_proxy: bool = True,
                   live_extend: bool = False) -> pd.DataFrame:
    """Compute the canonical CRYPTO daily target-weight DataFrame.

    Index: trading dates from 2010-03-11.
    Columns: leveraged BTC/ETH proxies + 'BIL'. Weights sum to 1.0.

    If use_live_proxy=True, GBTC weight is shifted to IBIT for dates where
    IBIT is listed (Jan 2024 onwards) — IBIT is the live spot BTC ETF;
    GBTC was the historical proxy (a closed-end Grayscale trust until
    its 2024 ETF conversion).

    live_extend: If True, extend the date index by one BDay forward
        (ffilled) so the last row is W[t+1] using close[t] info.
    """
    close, opn = {}, {}
    keys = ["GBTC", "ETHE", "SPY", "BIL"]
    if use_live_proxy:
        keys = list(set(keys + ["IBIT"]))
    for t in keys:
        df = load_etf(t)
        if df is not None:
            close[t] = df["Close"]; opn[t] = df["Open"]
    close = pd.DataFrame(close); opn = pd.DataFrame(opn)
    dates = opn["SPY"].dropna().index
    dates = dates[dates >= pd.Timestamp("2010-03-11")]
    if live_extend and len(dates) > 0:
        next_day = dates[-1] + pd.tseries.offsets.BDay()
        dates = dates.append(pd.DatetimeIndex([next_day]))
    close = close.reindex(dates).ffill(limit=5)
    opn = opn.reindex(dates).ffill(limit=5)

    hy = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    spy = close["SPY"]
    spy_ma = spy.rolling(200).mean()
    spy_ok = (spy > spy_ma) & (spy_ma.diff(20) > 0)
    hy_slope = hy - hy.shift(20)
    regime_ok = (spy_ok & (hy_slope < 1.0) & (vix < 30)).shift(1).fillna(False)

    W = build_crypto_weights(dates, close, opn, regime_ok)

    if use_live_proxy and "GBTC" in W.columns and "IBIT" in opn.columns:
        ibit_first = opn["IBIT"].dropna().index.min()
        if ibit_first is not None:
            W = W.copy()
            if "IBIT" not in W.columns:
                W["IBIT"] = 0.0
            mask = W.index >= ibit_first
            W.loc[mask, "IBIT"] = W.loc[mask, "IBIT"] + W.loc[mask, "GBTC"]
            W.loc[mask, "GBTC"] = 0.0
    return W


def main():
    # Load existing 4 sleeves
    van = pd.read_csv(R/"vanguard_returns.csv", parse_dates=[0], index_col=0)["net_ret"]
    ori = pd.read_csv(R/"orion_returns.csv", parse_dates=["Date"]).set_index("Date")["orion"]
    hel = pd.read_csv(R/"helios_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    qua = pd.read_csv(R/"quantum_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]

    # Build CRYPTO sleeve
    close, opn = {}, {}
    for t in ["GBTC","ETHE","SPY","BIL"]:
        df = load_etf(t)
        if df is not None:
            close[t] = df["Close"]; opn[t] = df["Open"]
    close = pd.DataFrame(close); opn = pd.DataFrame(opn)

    dates = opn["SPY"].dropna().index
    dates = dates[(dates >= pd.Timestamp("2010-03-11"))]
    close = close.reindex(dates).ffill(limit=5)
    opn = opn.reindex(dates).ffill(limit=5)

    hy = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    spy = close["SPY"]
    spy_ma = spy.rolling(200).mean()
    spy_ok = (spy > spy_ma) & (spy_ma.diff(20) > 0)
    hy_slope = hy - hy.shift(20)
    regime_ok = (spy_ok & (hy_slope < 1.0) & (vix < 30)).shift(1).fillna(False)

    crypto_ret = build_crypto_sleeve(dates, close, opn, regime_ok)

    # Align
    df = pd.concat({"V":van,"O":ori,"H":hel,"Q":qua,"C":crypto_ret}, axis=1).reindex(dates).fillna(0)

    print(f"Crypto sleeve standalone metrics:")
    mc_full = metrics(crypto_ret.reindex(dates).fillna(0))
    mc_is = metrics(crypto_ret.loc[crypto_ret.index <= IS_END])
    mc_oos = metrics(crypto_ret.loc[crypto_ret.index >= OOS_START])
    print(f"  FULL: SR {mc_full['sharpe']:5.2f}  CAGR {mc_full['cagr']*100:5.1f}%  MDD {mc_full['mdd']*100:5.1f}%")
    print(f"  IS  : SR {mc_is['sharpe']:5.2f}  CAGR {mc_is['cagr']*100:5.1f}%  MDD {mc_is['mdd']*100:5.1f}%")
    print(f"  OOS : SR {mc_oos['sharpe']:5.2f}  CAGR {mc_oos['cagr']*100:5.1f}%  MDD {mc_oos['mdd']*100:5.1f}%")
    print()

    # Correlation of crypto vs other sleeves (OOS, where GBTC data is full)
    oos_df = df.loc[OOS_START:]
    print("Crypto correlations with other sleeves (OOS only):")
    for other in ["V","O","H","Q"]:
        c = oos_df["C"].corr(oos_df[other])
        print(f"  C -- {other}: {c:.3f}")
    print()

    # Variant A: no crypto (4 sleeves)
    is_df4 = df[["V","O","H","Q"]].loc[:IS_END]
    iv4 = 1.0 / is_df4.std()
    w4 = (iv4 / iv4.sum())
    bA = df[["V","O","H","Q"]].values @ w4.values
    bA = pd.Series(bA, index=df.index)
    mA_full = metrics(bA); mA_is = metrics(bA.loc[:IS_END]); mA_oos = metrics(bA.loc[OOS_START:])

    # Variant B: 5 sleeves, IS inv-vol
    is_df5 = df[["V","O","H","Q","C"]].loc[:IS_END]
    iv5 = 1.0 / is_df5.std()
    # Guard against zero std on C during IS (no crypto before 2015)
    iv5 = iv5.replace([np.inf, -np.inf], np.nan).fillna(0)
    w5 = (iv5 / iv5.sum())
    bB = df[["V","O","H","Q","C"]].values @ w5.values
    bB = pd.Series(bB, index=df.index)
    mB_full = metrics(bB); mB_is = metrics(bB.loc[:IS_END]); mB_oos = metrics(bB.loc[OOS_START:])

    # Variant C: 4 sleeves + 10% crypto (fixed cap)
    # Cap crypto at 10%, renormalize the other 4 to 90% by inv-vol
    w_cap = pd.Series({"V": w4.V*0.9, "O": w4.O*0.9, "H": w4.H*0.9, "Q": w4.Q*0.9, "C": 0.10})
    bC = df[["V","O","H","Q","C"]].values @ w_cap.values
    bC = pd.Series(bC, index=df.index)
    mC_full = metrics(bC); mC_is = metrics(bC.loc[:IS_END]); mC_oos = metrics(bC.loc[OOS_START:])

    print(f"{'Variant':>32s}   {'FULL SR':>8s}  {'FULL CAGR':>10s}  {'FULL MDD':>9s}  {'OOS SR':>7s}  {'OOS CAGR':>9s}")
    for name, m_full, m_oos in [
        ("A: No crypto (baseline v2)", mA_full, mA_oos),
        ("B: + Crypto (IS inv-vol 5-sleeve)", mB_full, mB_oos),
        ("C: + Crypto capped @ 10%", mC_full, mC_oos),
    ]:
        print(f"{name:>32s}   {m_full['sharpe']:>8.2f}  {m_full['cagr']*100:>9.1f}%  "
              f"{m_full['mdd']*100:>8.1f}%  {m_oos['sharpe']:>7.2f}  {m_oos['cagr']*100:>8.1f}%")

    print()
    print(f"5-sleeve IS inv-vol weights: V={w5.V:.3f} O={w5.O:.3f} H={w5.H:.3f} Q={w5.Q:.3f} C={w5.C:.3f}")
    print(f"Capped 10% crypto weights:   V={w_cap.V:.3f} O={w_cap.O:.3f} H={w_cap.H:.3f} Q={w_cap.Q:.3f} C={w_cap.C:.3f}")

    out = {
        "crypto_sleeve_standalone": {"full": mc_full, "is": mc_is, "oos": mc_oos},
        "variant_A_no_crypto": {"full": mA_full, "is": mA_is, "oos": mA_oos, "weights": w4.to_dict()},
        "variant_B_crypto_invvol": {"full": mB_full, "is": mB_is, "oos": mB_oos, "weights": w5.to_dict()},
        "variant_C_crypto_capped10pct": {"full": mC_full, "is": mC_is, "oos": mC_oos, "weights": w_cap.to_dict()},
        "oos_corr_crypto_vs_others": {
            "V": float(oos_df["C"].corr(oos_df["V"])),
            "O": float(oos_df["C"].corr(oos_df["O"])),
            "H": float(oos_df["C"].corr(oos_df["H"])),
            "Q": float(oos_df["C"].corr(oos_df["Q"])),
        },
    }
    (R / "phoenix_v2_crypto.json").write_text(json.dumps(out, indent=2))

    # Save crypto sleeve returns for future use
    pd.DataFrame({"Date": dates, "ret": crypto_ret.values}).to_csv(R / "crypto_returns.csv", index=False)
    print(f"\nSaved phoenix_v2_crypto.json and crypto_returns.csv")


if __name__ == "__main__":
    main()
