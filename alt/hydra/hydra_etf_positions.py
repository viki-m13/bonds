"""Resolve HYDRA per-ETF notional exposure by replaying each sleeve's
internal weight toggle and per-sleeve vol scalar.

ETF_exposure[date, etf] = sum_over_sleeves (
    ensemble_inv_vol_weight[date, sleeve]      # risk_parity_ensemble
  * portfolio_scalar[date]                      # target 20% vol, cap 5x
  * sleeve_internal_weight[date, sleeve, etf]   # 2-way toggle, rebal'd monthly
  * per_sleeve_vol_scalar[date, sleeve]         # 10% target, cap 1.5x
)

Captured by monkey-patching `apply_tc` (grabs w) and `vol_target`
(grabs per-sleeve scalar) on hydra_core, then calling each sleeve.

Outputs:
  data/results/hydra_etf_positions.csv  — date x ETF, NAV fraction
"""
from pathlib import Path
import numpy as np
import pandas as pd

import hydra_core
from hydra_core import load_etf
from hydra_sleeves_v3 import SLEEVES


RESULTS = Path("/home/user/bonds/data/results")


def build():
    # --- capture apparatus ---
    captured_w = {}          # sleeve_name -> DataFrame (date x etf weight)
    captured_vs = {}         # sleeve_name -> Series (per-sleeve vol scalar)
    _hold = {"w": None, "vs": None}

    orig_apply_tc = hydra_core.apply_tc
    orig_vol_target = hydra_core.vol_target

    def patched_apply_tc(weight_ts, raw_ret, ret_index):
        _hold["w"] = weight_ts
        return orig_apply_tc(weight_ts, raw_ret, ret_index)

    def patched_vol_target(ret, target=hydra_core.VOL_TARGET,
                           window=hydra_core.VOL_LOOKBACK, cap=1.5):
        out, scale = orig_vol_target(ret, target=target, window=window, cap=cap)
        _hold["vs"] = scale
        return out, scale

    hydra_core.apply_tc = patched_apply_tc
    hydra_core.vol_target = patched_vol_target

    # Also patch the names imported at top of hydra_sleeves_v3
    import hydra_sleeves_v3 as hsv
    hsv.apply_tc = patched_apply_tc
    hsv.vol_target = patched_vol_target

    # --- run sleeves ---
    spy = load_etf("SPY")
    dates = spy.index

    for fn in SLEEVES:
        _hold["w"] = None
        _hold["vs"] = None
        r = fn(dates)
        name = r.name
        captured_w[name] = _hold["w"].copy() if _hold["w"] is not None else pd.DataFrame(index=dates)
        captured_vs[name] = _hold["vs"].copy() if _hold["vs"] is not None else pd.Series(0.0, index=dates)

    # restore
    hydra_core.apply_tc = orig_apply_tc
    hydra_core.vol_target = orig_vol_target

    # --- load ensemble-level weights and scalar ---
    sl = pd.read_csv(RESULTS / "hydra_sleeves.csv", parse_dates=["Date"]).set_index("Date")
    vols = sl.rolling(63).std().shift(1) * np.sqrt(252)
    vols = vols.where(vols > 0.001)
    inv = (1 / vols).where(vols.notna(), 0)
    w_port = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    raw = (w_port * sl).sum(axis=1)
    pv = raw.rolling(63).std().shift(1) * np.sqrt(252)
    scalar = (0.20 / pv).clip(upper=5.0).fillna(0)

    # sleeve gross = inv_vol_weight × portfolio_scalar
    sleeve_gross = w_port.multiply(scalar, axis=0)   # date x sleeve

    # --- aggregate to ETF exposure ---
    etf_exposure = pd.DataFrame(0.0, index=dates, columns=[], dtype=float)
    for name in sl.columns:
        if name not in captured_w:
            continue
        w_inner = captured_w[name].reindex(dates).fillna(0)   # date x etf
        vs = captured_vs[name].reindex(dates).fillna(0)       # date
        if name not in sleeve_gross.columns:
            continue
        g = sleeve_gross[name].reindex(dates).fillna(0)       # date
        # Each sleeve contributes g × w_inner × vs (signed, s.t. s26/s27/s28 have signed weights)
        contrib = w_inner.multiply(g * vs, axis=0)            # date x etf (this sleeve)
        for col in contrib.columns:
            if col not in etf_exposure.columns:
                etf_exposure[col] = 0.0
            etf_exposure[col] = etf_exposure[col].add(contrib[col], fill_value=0)

    etf_exposure = etf_exposure.fillna(0)
    etf_exposure.index.name = "Date"
    out = RESULTS / "hydra_etf_positions.csv"
    etf_exposure.to_csv(out)
    print(f"Wrote {out} ({len(etf_exposure)} rows x {etf_exposure.shape[1]} ETFs)")
    last = etf_exposure.iloc[-1].sort_values(key=abs, ascending=False)
    print(f"\nTop 10 ETF positions on {etf_exposure.index[-1].date()}:")
    for etf, pct in last.head(10).items():
        print(f"  {etf:6s}  {pct*100:+8.2f}%")
    trades = etf_exposure.diff().iloc[-1].sort_values(key=abs, ascending=False)
    print(f"\nTop 10 ETF trades on {etf_exposure.index[-1].date()}:")
    for etf, pct in trades.head(10).items():
        if abs(pct) < 1e-6:
            continue
        action = "BUY " if pct > 0 else "SELL"
        print(f"  {action} {etf:6s}  {pct*100:+8.3f}%")
    return etf_exposure


if __name__ == "__main__":
    build()
