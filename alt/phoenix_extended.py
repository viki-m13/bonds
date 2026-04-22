"""PHOENIX v2 extended backtest 2005-2026 using synthetic LETF histories.

Build simplified analogs of the 4 sleeves using the extended data so we can
test through the 2008 GFC. We're not trying to perfectly replicate VAN/ORI/HEL/QUA
here — we're asking a simpler question: does a similar multi-signal LETF
rotation strategy survive 2008?

Strategy analogs:
  S1 (VAN-analog): monthly momentum top-3 across 17 LETFs + SPY 200dma regime
                   gate + HY-OAS slope gate + VIX gate
  S2 (ORI-analog): weekly top-3 risk-on LETFs OR top-2 safe-haven (TMF/UGL)
                   based on VIX <22 toggle
  S3 (HEL-analog): weekly top-3 LETFs chosen by signal on unleveraged
                   underlyings (SPY/QQQ/TLT/GLD/XLK/XLE/XLF/SMH/VNQ/EEM/FXI)
  S4 (QUA-analog): 21-day composite of (21d, 63d, 252d return z-scores) top-3

All: next-open fill, 10 bps/side TC, macro gate identical across.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
EXT = ROOT / "data/etfs_extended"
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
R = ROOT / "data/results"

UNIVERSE = ["TQQQ","UPRO","QLD","SSO","SOXL","TECL","FAS","ERX","DRN","EDC",
            "YINN","UCO","UGL","NUGT","TMF","UBT","TYD"]
TC_BPS = 10.0


def load_etf(t, prefer_ext=True):
    # Prefer extended synthetic; fallback to original
    for p in [EXT/f"{t}.csv", ETF/f"{t}.csv"]:
        if p.exists():
            df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
            df = df[~df.index.duplicated(keep="first")]
            return df[["Close","Open"]].apply(pd.to_numeric, errors="coerce")
    return None


def load_fred(s):
    p = FRED/f"{s}.csv"
    if not p.exists(): return None
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    d = d[~d.index.duplicated(keep="first")]
    return pd.to_numeric(d.iloc[:,0], errors="coerce")


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
    return {"sharpe":float(sr),"cagr":float(cagr),"mdd":float(dd),
            "vol":float(sd),"sortino":float(sortino),
            "navx":float(c.iloc[-1])}


def build_regime(dates, close):
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    hy = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    spy = close["SPY"]
    spy_ma = spy.rolling(200).mean()
    spy_ok = (spy > spy_ma) & (spy_ma.diff(20) > 0)
    hy_slope = hy - hy.shift(20)
    regime = (spy_ok & (hy_slope < 1.0) & (vix < 30)).shift(1).fillna(False)
    # Risk-on subgate for ORI
    risk_on = (spy_ok & (vix.shift(1) < 22)).fillna(False)
    return regime, risk_on


def backtest_signal(dates, close, opn, regime, signal_fn, rebal_fn, K=3):
    cols = list(UNIVERSE) + ["BIL"]
    W = pd.DataFrame(0.0, index=dates, columns=cols)
    current = pd.Series(0.0, index=cols); current["BIL"] = 1.0

    for i, dt in enumerate(dates):
        if rebal_fn(i, dt):
            if regime.iloc[i]:
                s = signal_fn(i, dt).reindex(UNIVERSE).dropna()
                tradable = [t for t in UNIVERSE if t in s.index
                            and not np.isnan(opn[t].iloc[i])]
                s = s[tradable]
                top = s[s > 0].nlargest(K)  # require positive
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

    opn_ret = opn[UNIVERSE].pct_change().shift(-1).fillna(0)
    bil_ret = opn["BIL"].pct_change().shift(-1).fillna(0)
    port_ret = (W[UNIVERSE] * opn_ret[UNIVERSE]).sum(axis=1) + W["BIL"] * bil_ret
    dW = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    tc = dW * (TC_BPS/1e4)
    port_ret = (port_ret - tc).shift(1).fillna(0.0)
    return port_ret


def main():
    # Load ALL tickers
    close, opn = {}, {}
    for t in UNIVERSE + ["SPY","BIL","QQQ","TLT","IEF","GLD","USO","XLK","XLE","XLF","SMH","VNQ","EEM","FXI"]:
        df = load_etf(t)
        if df is not None:
            close[t] = df["Close"]; opn[t] = df["Open"]
    close = pd.DataFrame(close); opn = pd.DataFrame(opn)

    dates = opn["SPY"].dropna().index
    dates = dates[(dates >= pd.Timestamp("2005-01-03")) & (dates <= pd.Timestamp("2026-04-02"))]
    close = close.reindex(dates).ffill(limit=5)
    opn = opn.reindex(dates).ffill(limit=5)

    print(f"Extended backtest: {dates[0].date()} to {dates[-1].date()} ({len(dates)} days)")

    regime, risk_on = build_regime(dates, close)

    # Signals
    mom21 = close[UNIVERSE].pct_change(21).shift(1)
    mom63 = close[UNIVERSE].pct_change(63).shift(1)
    mom252 = close[UNIVERSE].pct_change(252).shift(1)
    rets = close[UNIVERSE].pct_change().fillna(0)
    vol63 = rets.rolling(63).std().shift(1) * np.sqrt(252)

    # Unleveraged underlyings for HEL analog
    UNDER_MAP = {"UPRO":"SPY","SSO":"SPY","TQQQ":"QQQ","QLD":"QQQ","SOXL":"SMH",
                 "TECL":"XLK","FAS":"XLF","ERX":"XLE","DRN":"VNQ","EDC":"EEM",
                 "YINN":"FXI","UCO":"USO","UGL":"GLD","NUGT":"GLD",
                 "TMF":"TLT","UBT":"TLT","TYD":"IEF"}
    under_mom63 = pd.DataFrame({
        t: (close[UNDER_MAP[t]].pct_change(63).shift(1) if UNDER_MAP[t] in close.columns
            else close[t].pct_change(63).shift(1))
        for t in UNIVERSE})

    # S1 VAN-analog: monthly momentum
    month_ends = dates.to_series().resample("ME").last().values
    rebal_month = lambda i, dt: dt in month_ends
    s1 = backtest_signal(dates, close, opn, regime,
                         lambda i, dt: mom63.iloc[i], rebal_month, K=3)

    # S2 ORI-analog: weekly, risk-on/safe toggle
    # Simplification: always use risk-on mom63 when regime OK
    rebal_fri = lambda i, dt: dt.dayofweek == 4 or i == 0
    s2 = backtest_signal(dates, close, opn, regime,
                         lambda i, dt: mom63.iloc[i], rebal_fri, K=3)

    # S3 HEL-analog: weekly, unleveraged signal
    s3 = backtest_signal(dates, close, opn, regime,
                         lambda i, dt: under_mom63.iloc[i], rebal_fri, K=2)

    # S4 QUA-analog: 21-day cadence, composite z-score
    def zscore(df, row_i):
        row = df.iloc[row_i]
        mu = row.mean(); sd = row.std()
        return (row - mu)/sd if sd > 0 else row*0
    def s4_signal(i, dt):
        z1 = zscore(mom21, i)
        z2 = zscore(mom63, i)
        z3 = zscore(mom252, i)
        return z1 + z2 + z3
    rebal_21 = lambda i, dt: i % 21 == 0
    s4 = backtest_signal(dates, close, opn, regime, s4_signal, rebal_21, K=3)

    # Standalone metrics
    print("\nStandalone analog sleeves (2005-2026):")
    print(f"{'Sleeve':12s}  {'FULL SR':>8s}  {'CAGR':>8s}  {'MDD':>8s}  {'Sortino':>7s}")
    for name, r in [("S1 VAN-analog", s1), ("S2 ORI-analog", s2),
                    ("S3 HEL-analog", s3), ("S4 QUA-analog", s4)]:
        m = metrics(r)
        print(f"{name:12s}  {m['sharpe']:>8.2f}  {m['cagr']*100:>7.1f}%  "
              f"{m['mdd']*100:>7.1f}%  {m['sortino']:>7.2f}")

    # Blend using IS inv-vol on the 2005-2018 extended IS
    df = pd.concat({"S1":s1,"S2":s2,"S3":s3,"S4":s4}, axis=1)
    IS_END = "2018-12-31"; OOS_START = "2019-01-02"
    iv = 1.0 / df.loc[:IS_END].std()
    iv = iv.replace([np.inf, -np.inf], np.nan).fillna(0)
    w = (iv/iv.sum())
    raw = pd.Series(df.values @ w.values, index=df.index)

    # Apply same overlay
    def apply_overlay(r, dd_floor=-0.10, vol_pct=0.99):
        cum = (1+r).cumprod()
        hwm = cum.rolling(252, min_periods=30).max()
        dd = cum/hwm - 1
        dd_mult = (1.0 + dd/dd_floor).clip(0,1).shift(1).fillna(1.0)
        rv = r.rolling(60).std()
        rv_thr = rv.rolling(252, min_periods=60).quantile(vol_pct)
        vol_mult = (rv <= rv_thr).shift(1).fillna(True).astype(float)
        vol_mult = vol_mult + (1-vol_mult)*0.5
        mult = (dd_mult * vol_mult).clip(0,1)
        return r*mult, mult

    ret, mult = apply_overlay(raw)

    print(f"\n4-sleeve analog weights (IS inv-vol 2005-2018):")
    print(f"  {w.round(3).to_dict()}")

    # Window metrics
    windows = {
        "2005-2018 (extended IS incl 2008 GFC)": ret.loc[:IS_END],
        "2019-2026 (OOS)":                         ret.loc[OOS_START:],
        "2005-2026 (extended FULL, 21 yrs)":       ret,
        "2008-01 to 2009-06 (GFC stress)":         ret.loc["2008-01-01":"2009-06-30"],
        "2008 calendar year":                       ret.loc["2008-01-01":"2008-12-31"],
    }
    print(f"\nExtended results:")
    print(f"  {'window':42s}  {'SR':>5s}  {'CAGR':>7s}  {'MDD':>7s}  {'NAVx':>5s}")
    for name, r in windows.items():
        m = metrics(r)
        print(f"  {name:42s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  {m['mdd']*100:>6.1f}%  {m['navx']:>5.2f}")

    # Save
    out = {
        "period": f"{dates[0].date()} to {dates[-1].date()}",
        "n_days": int(len(dates)),
        "analog_weights": w.to_dict(),
        "standalone_sleeves": {
            "S1_VAN_analog": metrics(s1),
            "S2_ORI_analog": metrics(s2),
            "S3_HEL_analog": metrics(s3),
            "S4_QUA_analog": metrics(s4),
        },
        "blend_overlayed": {name: metrics(r) for name, r in windows.items()},
        "note": "Analog sleeves — simpler than production VAN/ORI/HEL/QUA but same signal flavor. Does NOT use trained QUANTUM ML model (uses z-score composite instead). Tests whether the multi-signal framework survives 2008.",
    }
    (R/"phoenix_extended.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"Date": ret.index, "ret": ret.values, "raw_ret": raw.values,
                  "mult": mult.values}).to_csv(R/"phoenix_extended_returns.csv", index=False)
    print("\nSaved phoenix_extended.json and phoenix_extended_returns.csv")


if __name__ == "__main__":
    main()
