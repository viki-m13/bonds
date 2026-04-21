"""Invention 1 — drawdown-throttle overlay ("Nova" risk budgeting).

Rationale: every LETF strategy has 55-80% MDD and 2-3 yr recoveries.
The user experience problem is PATH, not Sharpe. A proven CTA technique:
de-leverage when underwater vs a trailing peak.

Mechanics (pre-registered):
  1. Track running 252d peak of base-strategy NAV
  2. Current DD = NAV / 252d-peak - 1
  3. Leverage multiplier:
       DD in [  0%, -10%]   -> 100% exposure
       DD in [-10%, -20%]   -> linear 100% -> 50%
       DD in [-20%, -30%]   -> linear 50%  -> 25%
       DD <  -30%           -> 25% exposure (floor)
  4. Shift(1) — throttle decision made at close T-1 is effective day T open
  5. Turnover cost on |Δw|

Applied to: TSMOM K=3m tv=15%, invvol core6 lb=63, invvol clean4 lb=21,
and the stability pick invvol core6 + VM.

Expected: MDD cut roughly in half, Sharpe approximately preserved.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import (common_window_returns, run_backtest, summarise)
from letf_crypto_universe import load_with_crypto
from letf_tsmom import tsmom_with_vol_target, tsmom_backtest, prep as tsmom_prep
from letf_volmanaged import vol_managed_backtest


OUT = Path("/home/user/bonds/data/results")


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


def apply_dd_throttle(base_ret, peak_window=252,
                      dd_start=-0.10, dd_mid=-0.20, dd_floor=-0.30,
                      w_start=1.0, w_mid=0.5, w_floor=0.25,
                      smooth_days=5, tc_bps=15):
    """DD-aware leverage multiplier."""
    nav = (1 + base_ret).cumprod()
    peak = nav.rolling(peak_window, min_periods=1).max()
    dd = nav / peak - 1

    def mult(d):
        if d >= dd_start:
            return w_start
        if d >= dd_mid:
            t = (d - dd_start) / (dd_mid - dd_start)  # 0..1
            return w_start + (w_mid - w_start) * t
        if d >= dd_floor:
            t = (d - dd_mid) / (dd_floor - dd_mid)
            return w_mid + (w_floor - w_mid) * t
        return w_floor

    m = dd.apply(mult)
    if smooth_days and smooth_days > 1:
        m = m.rolling(smooth_days, min_periods=1).mean()
    m = m.shift(1).fillna(w_start)  # decision at close T-1 effective day T
    turnover = m.diff().abs().fillna(0)
    tc = turnover * (tc_bps / 1e4)
    return m * base_ret - tc, m


def main():
    tsmom_px = tsmom_prep()
    px = load_with_crypto([], start="2011-01-01")
    rets = common_window_returns(px)

    core6 = ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"]
    clean4 = ["UPRO","TQQQ","TMF","UGL"]

    bases = {}
    r, _ = run_backtest(rets, invvol_fn(core6, 63), rebal_days=21)
    bases["invvol core6 lb=63"] = r
    r, _ = run_backtest(rets, invvol_fn(clean4, 21), rebal_days=21)
    bases["invvol clean4 lb=21"] = r
    r, _ = tsmom_with_vol_target(tsmom_px, K_months=3, target_vol=0.15)
    bases["TSMOM K=3m tv=15%"] = r
    r, _ = tsmom_with_vol_target(tsmom_px, K_months=3, target_vol=0.20)
    bases["TSMOM K=3m tv=20%"] = r
    vm, _ = vol_managed_backtest(bases["invvol core6 lb=63"], vol_window=126)
    bases["invvol core6 + VM"] = vm

    rows = []
    for name, r in bases.items():
        s = summarise(r.reindex(rets.index).fillna(0), f"{name} [raw]")
        rows.append(s)
        # Default DD-throttle: 10/20/30 breakpoints, 100/50/25 weights
        tr, _ = apply_dd_throttle(r)
        s2 = summarise(tr.reindex(rets.index).fillna(0), f"{name} + DD")
        rows.append(s2)
        # Tighter DD-throttle: 5/10/20 breakpoints (more reactive)
        tr2, _ = apply_dd_throttle(r, dd_start=-0.05, dd_mid=-0.10, dd_floor=-0.20)
        s3 = summarise(tr2.reindex(rets.index).fillna(0), f"{name} + DD-tight")
        rows.append(s3)
        # Asymmetric: weaker de-risk (keep more exposure)
        tr3, _ = apply_dd_throttle(r, w_mid=0.70, w_floor=0.50)
        s4 = summarise(tr3.reindex(rets.index).fillna(0), f"{name} + DD-soft")
        rows.append(s4)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_dd_throttle.csv", index=False)

    print(df.sort_values("sharpe", ascending=False).to_string(index=False))

    # Focused side-by-side
    print("\n=== Key comparison: DD-throttle effect per base strategy ===")
    for name in bases:
        sub = df[df.label.str.startswith(name)]
        print(f"\n{name}:")
        print(sub[["label","cagr","vol","mdd","sharpe","cagr_mdd"]].to_string(index=False))


if __name__ == "__main__":
    main()
