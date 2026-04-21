"""Priority 2b — Volatility-managed portfolios (Moreira & Muir 2017).

Pre-registered form:
  - Hold a FIXED long portfolio (the LETF exposure)
  - Scale gross exposure inversely to realised vol:
        w_t = c / sigma_t   where sigma_t = realised stdev of past 21-day returns
    c is chosen so that long-run average weight = 1 (no net tilt)
  - Moreira-Muir (JF 2017) find this raises Sharpe by ~0.15 in US equities,
    robust OOS, robust to the vol window.

Applied to:
  - 100% TQQQ (most volatile) — potentially huge vol-drag reduction
  - 60/40 UPRO/TMF (HFEA)
  - Static equal-weight core6
  - invvol clean4 lb=21 (our surviving contender)

We test vol windows 21d / 63d / 126d (no sweep beyond the Moreira-Muir canonical
set).
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import (common_window_returns, run_backtest, summarise,
                         w_fixed)
from letf_crypto_universe import load_with_crypto
from hydra_core import load_etf


OUT = Path("/home/user/bonds/data/results")


def vol_managed_backtest(base_ret, vol_window=21, target_vol=None, tc_bps=15):
    """Scale a daily return series inversely to realised vol.

    Given base_ret (raw unscaled strategy return series), compute:
        w_t = target_vol / sigma_{t-vol_window..t-1}
    capped at 3x.  Then scaled_ret = w_t * base_ret - tc(|Δw|)
    If target_vol is None, normalise so that mean(w) = 1 (Moreira-Muir form).
    """
    sigma = base_ret.rolling(vol_window).std() * np.sqrt(252)
    sigma = sigma.shift(1)  # use PAST vol, not contemporaneous
    w = 1.0 / sigma.replace(0, np.nan)
    if target_vol is None:
        # Scale so mean w = 1 (pure reweighting, zero-cost)
        mean_w = w.mean()
        w = w / mean_w
    else:
        w = w * target_vol
    w = w.clip(0.0, 3.0).fillna(0.0)
    # Costs on weight changes
    turnover = w.diff().abs().fillna(0)
    tc = turnover * (tc_bps / 1e4)
    return w * base_ret - tc, w


def run_strategy_get_rets(rets, fn, rebal_days=21):
    r, _ = run_backtest(rets, fn, rebal_days=rebal_days, exec_lag=1)
    return r


def main():
    px = load_with_crypto([], start="2011-01-01")
    rets = common_window_returns(px)

    core6 = ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"]
    clean4 = ["UPRO","TQQQ","TMF","UGL"]

    def invvol_fn(tickers, lookback):
        def fn(d, hist):
            if len(hist) < lookback + 5: return None
            r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
            if r.shape[1] == 0: return None
            inv = 1 / r.std().replace(0, np.nan).fillna(0)
            w = inv / inv.sum()
            out = pd.Series(0.0, index=hist.columns)
            out.loc[w.index] = w
            return out
        return fn

    bases = [
        ("100% TQQQ", w_fixed({"TQQQ": 1.0})),
        ("HFEA 55/45 UPRO/TMF", w_fixed({"UPRO":0.55,"TMF":0.45})),
        ("HFEA-Tech 60/40 TQQQ/TMF", w_fixed({"TQQQ":0.6,"TMF":0.4})),
        ("EW core6", w_fixed({t: 1/6 for t in core6})),
        ("invvol clean4 lb=21", invvol_fn(clean4, 21)),
        ("invvol core6 lb=63", invvol_fn(core6, 63)),
    ]

    rows = []
    for name, fn in bases:
        base_r = run_strategy_get_rets(rets, fn)
        s = summarise(base_r, f"{name} [unscaled]")
        rows.append(s)
        # Vol-managed: no explicit target (mean-w=1)
        for vw in (21, 63, 126):
            scaled_r, _ = vol_managed_backtest(base_r, vol_window=vw)
            s = summarise(scaled_r, f"{name} VM vw={vw}d")
            rows.append(s)
        # Vol-targeted form (canonical target 20% ann)
        for tv in (0.20, 0.30):
            scaled_r, _ = vol_managed_backtest(base_r, vol_window=63, target_vol=tv)
            s = summarise(scaled_r, f"{name} VM tv={int(tv*100)}% vw=63d")
            rows.append(s)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_volmanaged.csv", index=False)

    print(df.sort_values("sharpe", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
