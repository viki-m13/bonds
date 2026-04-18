"""Build AURORA strategy returns CSV and save to data/results/.
AURORA = Covered-Call Income 40% + Weekly Momentum 40% + Managed Futures 20%.

Config picked from aurora_step3_combo.py grid search — the best
Sharpe ratio configuration with annual return >= 20%.
  SR=1.22  Ret=20.3%  Vol=16.7%  MDD=-15.4%
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"

# AURORA weights
W_COVCALL = 0.40      # JEPI/JEPQ/SPYI/DIVO equal-weight blend
W_MOMO = 0.40         # Weekly top-3 of 3x leveraged ETFs
W_MF = 0.20           # DBMF/CTA/KMLM equal-weight blend

MOMO_UNIVERSE = ["TQQQ", "UPRO", "SOXL", "TECL", "FAS", "TMF", "UGL"]
COVCALL_TICKERS = ["JEPI", "JEPQ", "SPYI", "DIVO"]
MF_TICKERS = ["DBMF", "CTA", "KMLM"]


def load_etf(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()


def load_fred(s):
    p = FRED / f"{s}.csv"
    if not p.exists(): return None
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    return pd.to_numeric(d, errors="coerce").sort_index()


def weekly_momo(universe, lookback, top_n, dates, rebal_days=5,
                tc_bps=15.0, regime=None, cash_ticker="BIL"):
    prices = pd.DataFrame({t: load_etf(t) for t in universe}).dropna()
    prices = prices.reindex(dates).ffill().dropna()
    rets = prices.pct_change().fillna(0)
    d2 = rets.index
    cash = load_etf(cash_ticker).reindex(d2).ffill().pct_change().fillna(0)
    current = pd.Series(0.0, index=prices.columns)
    port = pd.Series(0.0, index=d2)
    last_idx = -rebal_days
    rebal_events = []
    for i in range(len(d2)):
        if i >= lookback and i - last_idx >= rebal_days:
            momo = prices.iloc[i] / prices.iloc[i - lookback] - 1
            ranked = momo.sort_values(ascending=False)
            positive = [t for t in ranked.index if momo[t] > 0]
            top = positive[:top_n]
            new_target = pd.Series(0.0, index=prices.columns)
            if top:
                w = 1.0 / len(top)
                for t in top:
                    new_target[t] = w
            tc = (new_target - current).abs().sum() * (tc_bps / 1e4)
            port.iloc[i] -= tc
            rebal_events.append({"date": d2[i], "holdings": list(top),
                                 "weight_each": 1.0/len(top) if top else 0.0})
            current = new_target
            last_idx = i
        r = (rets.iloc[i] * current).sum()
        g = float(regime.iloc[i]) if regime is not None else 1.0
        r = g * r + (1 - g) * cash.iloc[i]
        port.iloc[i] += r
    return port, rebal_events


def covered_call(dates):
    pieces = {}
    for t in COVCALL_TICKERS:
        p = load_etf(t)
        if p is None: continue
        pieces[t] = p.reindex(dates).ffill().pct_change().fillna(0)
    df = pd.DataFrame(pieces)
    avail = (df != 0).astype(int).sum(axis=1).replace(0, np.nan)
    ret = df.sum(axis=1) / avail.fillna(1)
    bil = load_etf("BIL").reindex(dates).ffill().pct_change().fillna(0)
    return ret.where(avail.notna(), bil).fillna(0)


def managed_futures(dates):
    pieces = {}
    for t in MF_TICKERS:
        p = load_etf(t)
        if p is None: continue
        pieces[t] = p.reindex(dates).ffill().pct_change().fillna(0)
    df = pd.DataFrame(pieces)
    avail = (df != 0).astype(int).sum(axis=1).replace(0, np.nan)
    ret = df.sum(axis=1) / avail.fillna(1)
    bil = load_etf("BIL").reindex(dates).ffill().pct_change().fillna(0)
    return ret.where(avail.notna(), bil).fillna(0)


def spy_regime(dates, vix_cap=30, ma_len=200):
    spy = load_etf("SPY").reindex(dates).ffill()
    vix = load_fred("VIXCLS")
    ok = (spy > spy.rolling(ma_len).mean())
    if vix is not None:
        v = vix.reindex(dates).ffill()
        ok = ok & (v < vix_cap)
    return ok.shift(1).fillna(False).astype(float)


def main():
    print("Building AURORA...")
    dates = load_etf("SPY").loc["2014-01-01":].index
    regime = spy_regime(dates)

    sleeve_C = covered_call(dates)
    sleeve_M, rebal_events = weekly_momo(MOMO_UNIVERSE, 20, 3, dates,
                                          rebal_days=5, regime=regime)
    sleeve_F = managed_futures(dates)

    port = W_COVCALL * sleeve_C + W_MOMO * sleeve_M + W_MF * sleeve_F
    # Trim to non-zero start (when covered-call and MF are actually trading)
    first_nz = port.ne(0).idxmax() if (port != 0).any() else port.index[0]
    port = port.loc[first_nz:]
    rebal_events = [e for e in rebal_events if e["date"] >= first_nz]

    # Save main returns CSV (matches zephyr_returns.csv format)
    out = port.to_frame("Close")
    out.index.name = "Date"
    out.to_csv(RESULTS / "aurora_returns.csv")

    # Save sleeve returns
    sleeves_df = pd.DataFrame({
        "CoveredCall": sleeve_C.loc[first_nz:],
        "Momo": sleeve_M.loc[first_nz:],
        "MF": sleeve_F.loc[first_nz:],
    })
    sleeves_df.index.name = "Date"
    sleeves_df.to_csv(RESULTS / "aurora_sleeves.csv")

    # Save rebalance events (weekly momo picks)
    rebal_df = pd.DataFrame([
        {"date": e["date"].strftime("%Y-%m-%d"),
         "pick_1": e["holdings"][0] if len(e["holdings"]) > 0 else "",
         "pick_2": e["holdings"][1] if len(e["holdings"]) > 1 else "",
         "pick_3": e["holdings"][2] if len(e["holdings"]) > 2 else "",
         "n_holdings": len(e["holdings"])}
        for e in rebal_events
    ])
    rebal_df.to_csv(RESULTS / "aurora_momo_rebalances.csv", index=False)

    # Regime series
    regime.loc[first_nz:].to_frame("gate").to_csv(
        RESULTS / "aurora_regime.csv", index_label="Date")

    # Report metrics
    ar = port.mean() * 252
    av = port.std() * np.sqrt(252)
    sr = ar / av if av > 0 else 0
    cum = (1 + port).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    print(f"AURORA from {first_nz.date()} to {port.index[-1].date()} ({len(port)/252:.1f}y)")
    print(f"  Sharpe={sr:.2f} Ret={ar*100:.1f}% Vol={av*100:.1f}% MDD={mdd*100:.1f}%")
    print(f"  NAV multiple: {cum.iloc[-1]:.3f}x  ({len(rebal_events)} momo rebalances)")


if __name__ == "__main__":
    main()
