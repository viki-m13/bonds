"""Invention 2 — VIX-regime-gated leverage.

Rationale: high VIX = widespread de-risking. Our LETF portfolios take the
biggest losses in high-VIX regimes (2020 COVID, 2022 inflation). A VIX-based
gate can cut exposure before realised drawdown shows up.

Pre-registered rule (no sweep):
  VIX < 20   -> 100% exposure
  VIX 20-30  -> linear 100% -> 50%
  VIX 30-40  -> linear 50%  -> 25%
  VIX > 40   -> 25% (floor)

Signal uses prior-day VIX close (no look-ahead). Shift(1) for execution lag.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import (common_window_returns, run_backtest, summarise)
from letf_crypto_universe import load_with_crypto
from letf_tsmom import tsmom_with_vol_target, prep as tsmom_prep
from letf_volmanaged import vol_managed_backtest
from hydra_core import load_etf


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


def load_vix():
    df = pd.read_csv("/home/user/bonds/data/fred/VIXCLS.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    s = df.set_index("Date")["VIXCLS"].astype(float)
    s = s.replace(".", np.nan).dropna()
    return s


def vix_gate_multiplier(vix_series, idx,
                        v_low=20, v_mid=30, v_hi=40,
                        w_low=1.0, w_mid=0.5, w_hi=0.25,
                        smooth_days=5):
    v = vix_series.reindex(idx).ffill()

    def mult(x):
        if pd.isna(x) or x <= v_low:
            return w_low
        if x <= v_mid:
            t = (x - v_low) / (v_mid - v_low)
            return w_low + (w_mid - w_low) * t
        if x <= v_hi:
            t = (x - v_mid) / (v_hi - v_mid)
            return w_mid + (w_hi - w_mid) * t
        return w_hi

    m = v.apply(mult)
    if smooth_days and smooth_days > 1:
        m = m.rolling(smooth_days, min_periods=1).mean()
    m = m.shift(1).fillna(1.0)
    return m


def apply_vix_gate(base_ret, vix, tc_bps=15, **kw):
    m = vix_gate_multiplier(vix, base_ret.index, **kw)
    turnover = m.diff().abs().fillna(0)
    tc = turnover * (tc_bps / 1e4)
    return m * base_ret - tc, m


def main():
    vix = load_vix()
    if vix is None:
        print("VIX data not available; aborting.")
        return
    print(f"VIX loaded: {vix.index[0].date()} .. {vix.index[-1].date()}")

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
    r, _ = tsmom_with_vol_target(tsmom_px, K_months=3, target_vol=0.20)
    bases["TSMOM K=3m tv=20%"] = r
    vm, _ = vol_managed_backtest(bases["invvol core6 lb=63"], vol_window=126)
    bases["invvol core6 + VM"] = vm

    rows = []
    for name, r in bases.items():
        r = r.reindex(rets.index).fillna(0)
        s = summarise(r, f"{name} [raw]")
        rows.append(s)
        # Default VIX gate: 20/30/40 breakpoints
        g, _ = apply_vix_gate(r, vix)
        rows.append(summarise(g, f"{name} + VIX"))
        # Aggressive gate: 18/25/35
        g2, _ = apply_vix_gate(r, vix, v_low=18, v_mid=25, v_hi=35)
        rows.append(summarise(g2, f"{name} + VIX-tight"))
        # Softer gate: 22/35/50
        g3, _ = apply_vix_gate(r, vix, v_low=22, v_mid=35, v_hi=50)
        rows.append(summarise(g3, f"{name} + VIX-soft"))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_vix_gate.csv", index=False)
    print(df.sort_values("sharpe", ascending=False).to_string(index=False))

    print("\n=== Effect per base ===")
    for name in bases:
        sub = df[df.label.str.startswith(name)]
        print(f"\n{name}:")
        print(sub[["label","cagr","vol","mdd","sharpe","cagr_mdd"]].to_string(index=False))


if __name__ == "__main__":
    main()
