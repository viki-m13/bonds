"""ML sleeve + aggressive overlays to see if this alone can hit higher Sharpe."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import util

OUT = Path("/home/user/bonds/data/apex")
FRED = Path("/home/user/bonds/data/fred")


def _fred(name, idx):
    fp = FRED / f"{name}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[df.columns[0]].astype(float).reindex(idx).ffill()


def main():
    op, cp = util.load_prices()
    # Load ML weights
    W = pd.read_csv(OUT / "ml_v2_weights.csv", parse_dates=["Date"], index_col="Date")
    W = W.reindex(cp.index).fillna(0.0)
    full = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for c in W.columns:
        if c in full.columns:
            full[c] = W[c]

    # Strong overlays
    # 1. Dual-bear gate
    spy = cp["SPY"]
    tlt = cp["TLT"] if "TLT" in cp.columns else spy
    # Use 20d windows for faster reaction
    spy_r20 = spy.pct_change(20)
    tlt_r20 = tlt.pct_change(20)
    dual_bear = ((spy_r20 < -0.02) & (tlt_r20 < -0.01)).astype(float)
    dual_m = (1 - 0.75 * dual_bear).shift(1).fillna(1.0)

    # 2. VIX gate
    vix = _fred("VIXCLS", cp.index)
    vix_m = pd.Series(1.0, index=cp.index)
    vix_m[vix > 30] = 0.5
    vix_m[vix > 40] = 0.25
    vix_m = vix_m.shift(1).fillna(1.0)

    # 3. Apply to weights
    full = full.mul(dual_m * vix_m, axis=0)

    # Raw return
    rets = cp.pct_change()
    raw_r = (full.shift(1).fillna(0.0) * rets.reindex_like(full).fillna(0.0)).sum(axis=1)

    # DD throttle
    c = (1 + raw_r).cumprod()
    hwm = c.rolling(252, min_periods=30).max()
    dd = c / hwm - 1
    dd_mult = (1 + dd / -0.10).clip(0, 1).shift(1).fillna(1.0)

    # Vol target
    rv = raw_r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    target_vol = 0.20
    vm_raw = (target_vol / rv.replace(0, np.nan)).clip(lower=0.1, upper=3.0)
    gross = full.sum(axis=1).replace(0, np.nan)
    max_up = (1.0 / gross).clip(lower=1.0)
    vol_mult = np.minimum(vm_raw, max_up).shift(1).fillna(1.0)

    total_mult = dd_mult * vol_mult
    w_eff = full.mul(total_mult, axis=0)
    rs = w_eff.sum(axis=1)
    final_scale = np.minimum(1.0, 1.0 / rs.replace(0, np.nan)).fillna(1.0)
    w_eff = w_eff.mul(final_scale, axis=0)

    # Net return
    gross_r = (w_eff.shift(1).fillna(0.0) * rets.reindex_like(w_eff).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w_eff.diff().abs().fillna(w_eff.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w_eff.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    net = gross_r - drag

    print("=== ML + overlays ===")
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", "2018-12-31")),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC", ("2007-01-01", "2009-12-31")),
                        ("COVID", ("2020-01-01", "2020-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31")),
                        ("2023-24", ("2023-01-01", "2024-12-31"))]:
        util.summarize(util.regime_slice(net, s, e), f"  {lbl}")


if __name__ == "__main__":
    main()
