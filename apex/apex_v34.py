"""APEX v34 — same as v33 but caps daily positions at MAX 5 (Phoenix-style).

User insight: Phoenix runs ~5 positions at a time. APEX was running 10-14
which dilutes signal and adds turnover. Cap to top-5 per day and renormalize.

Also adds a Phoenix-comparable regime period set to the factsheet:
  - 2010-2026 FULL (Phoenix native window)
  - 2010-2018 IS
  - 2019-2026 OOS
  - 2008 calendar year
  - Jan 2008 - Jun 2009 (full GFC stress)
  - 2020 COVID
  - 2022 rate-hike
  - 2023-2024 recovery
  - 2025+ live
  - 1999-2026 FULL (APEX-only deep history)
  - Pre-2008 (2000-2008)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
import numpy as np
import pandas as pd
import util
import sleeves_phoenix_exact as PX
import sleeves_v15 as SV15
import sleeves_v18 as SV18
import sleeves_v26 as SV26
import sleeves_v29 as SV29
import sleeves_v12 as SV12
import crypto_v2 as CV2
from apex_v13 import extend_cp

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"

LETF_BUILDERS = {
    "PX_HELIOS":    lambda cp: PX.sleeve_helios_exact(cp),
    "HMM_REGIME":   lambda cp: SV18.sleeve_hmm(cp),
    "DIVERGENCE":   lambda cp: SV18.sleeve_divergence(cp),
    "ACCEL_MOM":    lambda cp: SV26.sleeve_accel_mom(cp),
    "SKEW_MOM":     lambda cp: SV29.sleeve_skew_mom(cp),
    "HURST":        lambda cp: SV26.sleeve_hurst(cp),
}


def build_sleeves(cp):
    sw = {}
    for name, fn in LETF_BUILDERS.items():
        W = fn(cp)
        r = PX._weights_to_ret(W, cp)
        rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
        m = (0.15 / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
        sw[name] = W.mul(m, axis=0)
    return sw


def cap_positions(W: pd.DataFrame, max_pos: int = 5) -> pd.DataFrame:
    """Per-row, keep only top `max_pos` positions by weight; redistribute mass.

    Each row's gross is preserved by scaling the surviving positions up.
    """
    if max_pos >= len(W.columns):
        return W
    arr = W.to_numpy(copy=True)
    n_rows, n_cols = arr.shape
    for i in range(n_rows):
        row = arr[i]
        if not np.any(row > 0):
            continue
        gross = row.sum()
        if gross <= 0:
            continue
        idx = np.argsort(-row)[:max_pos]
        new = np.zeros_like(row)
        new[idx] = row[idx]
        new_gross = new.sum()
        if new_gross > 0:
            new = new * (gross / new_gross)
        arr[i] = new
    return pd.DataFrame(arr, index=W.index, columns=W.columns)


def run_v34(cp, sw, base_crypto_w=0.40, swing=0.15,
            target_vol=0.18, dd_floor=-0.10, max_pos: int = 5):
    """Same as v33 dynamic-crypto run but caps each day to max_pos LETFs."""
    strength = CV2.btc_regime_strength(cp.index)
    crypto_w_series = base_crypto_w + swing * (2 * strength - 1)
    crypto_w_series = crypto_w_series.clip(lower=0.20, upper=0.80)
    crypto_w_series = crypto_w_series.shift(1).fillna(base_crypto_w)
    letf_cap_series = 1.0 - crypto_w_series

    first = next(iter(sw.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    n = len(sw)
    for name, W in sw.items():
        w_sleeve = letf_cap_series / n
        P = P + W.fillna(0.0).mul(w_sleeve, axis=0)

    # ====== POSITION CAP (Phoenix-style) ======
    P = cap_positions(P, max_pos=max_pos)

    rets = cp.pct_change()
    spy_rv60 = cp["SPY"].pct_change().rolling(60).std() * np.sqrt(util.DPY)
    thr = spy_rv60.rolling(504, min_periods=60).quantile(0.99)
    regime_ok = (spy_rv60 <= thr).astype(float).fillna(1.0)
    regime_mult = (regime_ok + (1 - regime_ok) * 0.5).shift(1).fillna(1.0)
    P = P.mul(regime_mult, axis=0)

    dbs = SV12.dual_bear_score(cp)
    dbs_mult = pd.Series(1.0, index=cp.index)
    dbs_mult[dbs >= 3] = 0.5
    dbs_mult[dbs >= 4] = 0.25
    dbs_mult = dbs_mult.shift(1).fillna(1.0)
    P = P.mul(dbs_mult, axis=0)

    raw_r = (P.shift(1).fillna(0.0) * rets.reindex_like(P).fillna(0.0)).sum(axis=1)
    c = (1 + raw_r).cumprod()
    hwm = c.rolling(252, min_periods=30).max()
    dd = c / hwm - 1
    dd_mult = (1 + dd / dd_floor).clip(0, 1).shift(1).fillna(1.0)

    rv = raw_r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    vm_raw = (target_vol / rv.replace(0, np.nan)).clip(lower=0.2, upper=3.0)
    gross_now = P.sum(axis=1).replace(0, np.nan)
    max_up = (letf_cap_series / gross_now).clip(lower=1.0)
    vol_mult = np.minimum(vm_raw, max_up).shift(1).fillna(1.0)

    total_mult = dd_mult * vol_mult
    w_eff = P.mul(total_mult, axis=0)
    rs = w_eff.sum(axis=1)
    fs = np.minimum(1.0, letf_cap_series / rs.replace(0, np.nan)).fillna(1.0)
    w_eff = w_eff.mul(fs, axis=0)

    gross_ret = (w_eff.shift(1).fillna(0.0) * rets.reindex_like(w_eff).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w_eff.diff().abs().fillna(w_eff.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w_eff.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    letf_net = gross_ret - drag

    crypto_r = SV15.multi_crypto_returns(cp.index, target_vol=0.20)
    net = letf_net + crypto_w_series * crypto_r
    return net, w_eff


def main():
    op, cp = util.load_prices()
    cp = extend_cp(cp)
    for t in ["SH", "PSQ", "SDS", "TBF", "UUP", "DBC", "HYG"]:
        if t not in cp.columns:
            s = SV15._etf_close(t, cp.index)
            if not s.isna().all():
                cp[t] = s

    sw = build_sleeves(cp)
    net, w_eff = run_v34(cp, sw, base_crypto_w=0.40, swing=0.15,
                          target_vol=0.18, dd_floor=-0.10, max_pos=5)

    # === Headline ===
    print("=== APEX v34 (max 5 positions) ===")
    for lbl, (s, e) in [
        ("FULL 1999-2026", ("1999-01-01", "2027-12-31")),
        ("Phoenix-window 2010-2026", ("2010-01-01", "2027-12-31")),
        ("IS 2005-2018", ("2005-01-01", IS_END)),
        ("OOS 2019+", (OOS_START, "2027-12-31")),
        ("Pre-2008 (2000-2008)", ("2000-01-01", "2008-12-31")),
        ("GFC Jan-08 to Jun-09", ("2008-01-01", "2009-06-30")),
        ("2008 cal", ("2008-01-01", "2008-12-31")),
        ("COVID 2020", ("2020-01-01", "2020-12-31")),
        ("2022 rate-hike", ("2022-01-01", "2022-12-31")),
        ("2023-2024 recovery", ("2023-01-01", "2024-12-31")),
        ("2025+ live", ("2025-01-01", "2027-12-31")),
    ]:
        util.summarize(util.regime_slice(net, s, e), lbl)

    # === Position count diagnostics ===
    pos_counts = (w_eff > 0.001).sum(axis=1)
    print(f"\n=== Position count (where >0.1% weight) ===")
    print(f"  Mean:   {pos_counts.mean():.2f}")
    print(f"  Median: {pos_counts.median():.0f}")
    print(f"  Max:    {pos_counts.max()}")
    print(f"  P95:    {pos_counts.quantile(0.95):.0f}")
    print(f"  P99:    {pos_counts.quantile(0.99):.0f}")

    # === Save ===
    net.to_frame("apex_v34_ret").to_csv(OUT / "apex_v34_returns.csv")
    w_eff.to_csv(OUT / "apex_v34_weights.csv")
    (OUT / "apex_v34_meta.json").write_text(json.dumps({
        "version": "v34_max5pos",
        "sleeves": list(LETF_BUILDERS.keys()),
        "max_positions": 5,
        "base_crypto_w": 0.40,
        "swing": 0.15,
        "target_vol": 0.18,
        "dd_floor": -0.10,
    }, indent=2))
    print(f"\nSaved v34 returns and weights to {OUT}")


if __name__ == "__main__":
    main()
