"""Priority 2c — Crash-protected momentum (Barroso & Santa-Clara 2015).

BSc (JFE 2015) showed that momentum strategies can have their Sharpe roughly
DOUBLED by adding a vol-gate: scale exposure inversely to realised strategy
vol. Specifically:
  - Run momentum as usual (top-N cross-sectional, 12-1 month lookback)
  - Compute trailing realised vol of the strategy over 6 months
  - Scale gross exposure so ex-ante vol = target (12% is their canonical)

Pre-registered form:
  - Momentum: top 3 of 17 LETFs, lookback 126d, skip-1-month (skip-21d),
    monthly rebal
  - Target ex-ante vol: 15% (between BSc canonical 12% and our 20% bootstrap floor)
  - Vol window: 6 months (126 days)
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import (common_window_returns, run_backtest, summarise)
from letf_crypto_universe import load_with_crypto
from letf_universe import LETF_LONG_2011


OUT = Path("/home/user/bonds/data/results")


def mom_skip1_fn(tickers, lookback=252, skip=21, top_n=3):
    """Classic momentum: past lookback-skip to lookback return; skip most recent."""
    def fn(d, hist):
        if len(hist) < lookback + 10: return None
        past = hist[tickers].iloc[-lookback:-skip] if skip > 0 else hist[tickers].iloc[-lookback:]
        past = past.dropna(axis=1, how="any")
        if past.shape[1] == 0: return None
        cum = (1 + past).prod() - 1
        picks = cum.sort_values(ascending=False).head(top_n).index.tolist()
        out = pd.Series(0.0, index=hist.columns)
        if picks:
            out.loc[picks] = 1.0 / len(picks)
        return out
    return fn


def apply_crash_protection(base_ret, vol_window=126, target_vol=0.15,
                            cap=3.0, tc_bps=15):
    sigma = base_ret.rolling(vol_window).std() * np.sqrt(252)
    sigma = sigma.shift(1)
    w = (target_vol / sigma).replace([np.inf, -np.inf], np.nan).clip(0, cap).fillna(0)
    turnover = w.diff().abs().fillna(0)
    tc = turnover * (tc_bps / 1e4)
    return w * base_ret - tc, w


def main():
    px = load_with_crypto([], start="2011-01-01")
    rets = common_window_returns(px)

    rows = []
    for lb in (126, 252):
        for sk in (0, 21):
            for n in (3, 5):
                fn = mom_skip1_fn(LETF_LONG_2011, lookback=lb, skip=sk, top_n=n)
                r_raw, _ = run_backtest(rets, fn, rebal_days=21, exec_lag=1)
                s = summarise(r_raw, f"mom all17 lb={lb} skip={sk} top{n} RAW")
                rows.append(s)
                for tv in (0.12, 0.15, 0.20):
                    for vw in (63, 126, 252):
                        r_vm, _ = apply_crash_protection(r_raw, vol_window=vw,
                                                         target_vol=tv)
                        s = summarise(r_vm,
                          f"mom all17 lb={lb} skip={sk} top{n} CP tv={int(tv*100)}% vw={vw}d")
                        rows.append(s)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_crashmom.csv", index=False)

    # Pre-registered canonical pick: lb=252, skip=21, top3, tv=15%, vw=126d
    canonical = df[df.label == "mom all17 lb=252 skip=21 top3 CP tv=15% vw=126d"]
    print("Pre-registered canonical BSc momentum (lb=252 skip=21 top3, CP tv=15% vw=126d):")
    print(canonical.to_string(index=False))
    print()
    print("Top 20 by Sharpe:")
    print(df.sort_values("sharpe", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
