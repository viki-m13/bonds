"""APEX — final strategy candidate.

Architecture:
  Primary: ML5 sleeve (5-day horizon XGBoost, top-3, weekly rebal) — high CAGR
  Diversifiers: TREND, RPAR_CF, PAA, ORION — robust rule-based
  Overlays: dual-bear gate, VIX gate, DD throttle, vol target

Blend: 40% ML5 + 60% split among 4 rule-based sleeves (15% each).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import pandas as pd
import util
import sleeves_v3 as S3

OUT = Path("/home/user/bonds/data/apex")
FRED = Path("/home/user/bonds/data/fred")


def _fred(name, idx):
    fp = FRED / f"{name}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[df.columns[0]].astype(float).reindex(idx).ffill()


def load_ml5(cp, target_vol=0.15):
    """Load ML5 weights from disk, scale to target vol."""
    fp = Path("/home/user/bonds/data/apex/ml5_weights.csv")
    if not fp.exists():
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    W = pd.read_csv(fp, parse_dates=["Date"], index_col="Date")
    W = W.reindex(cp.index).fillna(0.0)
    full = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for c in W.columns:
        if c in full.columns:
            full[c] = W[c]
    # Scale to target vol (down only)
    r = S3._weights_to_returns(full, cp)
    rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return full.mul(m, axis=0)


SLEEVE_FNS = {
    "ML5":      load_ml5,
    "TREND":    S3.sleeve_trend_vol,
    "RPAR_CF":  S3.sleeve_rpar_cf,
    "PAA":      S3.sleeve_paa,
    "ORION":    S3.sleeve_orion,
    "HELIOS":   S3.sleeve_helios,
}

# ML5 is the alpha engine; others provide diversification
BLEND_WEIGHTS = {
    "ML5":     0.50,
    "TREND":   0.12,
    "RPAR_CF": 0.15,
    "PAA":     0.08,
    "ORION":   0.08,
    "HELIOS":  0.07,
}

PORT_VOL_TARGET = 0.25
DD_FLOOR = -0.10


def dual_bear_gate(cp, spy_thr=-0.02, tlt_thr=-0.01, win=20):
    spy = cp["SPY"]
    tlt = cp["TLT"] if "TLT" in cp.columns else spy
    dual_bear = ((spy.pct_change(win) < spy_thr) & (tlt.pct_change(win) < tlt_thr)).astype(float)
    return (1 - 0.75 * dual_bear).shift(1).fillna(1.0)


def vix_gate(cp):
    vix = _fred("VIXCLS", cp.index)
    m = pd.Series(1.0, index=cp.index)
    m[vix > 30] = 0.5
    m[vix > 40] = 0.25
    return m.shift(1).fillna(1.0)


def corr_flip_gate(cp):
    """When corr(SPY, TLT, 20d) > 0 for 10 of last 10 days → halve TMF/UBT/TYD"""
    spy_r = cp["SPY"].pct_change()
    tlt_r = cp["TLT"].pct_change() if "TLT" in cp.columns else spy_r
    corr = spy_r.rolling(20).corr(tlt_r)
    pos_streak = (corr > 0).rolling(10).sum()
    return (pos_streak < 8).astype(float).fillna(1.0).shift(1).fillna(1.0)


def build(cp):
    sw = {}
    for name, fn in SLEEVE_FNS.items():
        try:
            sw[name] = fn(cp, target_vol=0.15)
        except TypeError:
            sw[name] = fn(cp)
    return sw


def run(cp, sleeve_weights=None, blend=None, target_vol=PORT_VOL_TARGET, dd_floor=DD_FLOOR):
    if sleeve_weights is None:
        sleeve_weights = build(cp)
    sw = sleeve_weights
    if blend is None:
        blend = BLEND_WEIGHTS
    first = next(iter(sleeve_weights.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    for name, W in sleeve_weights.items():
        if name in blend:
            P = P + W.fillna(0.0) * blend[name]
    P = P.clip(upper=1.0, lower=0.0)

    # Apply bond-correlation-flip gate to bond LETFs only
    bond_assets = [c for c in ["TMF", "UBT", "TYD"] if c in P.columns]
    if bond_assets:
        cfm = corr_flip_gate(cp)
        for c in bond_assets:
            P[c] = P[c] * cfm

    # Apply dual-bear and VIX gates to all
    vm = vix_gate(cp)
    dm = dual_bear_gate(cp)
    P = P.mul(vm * dm, axis=0)

    rets = cp.pct_change()
    raw_r = (P.shift(1).fillna(0.0) * rets.reindex_like(P).fillna(0.0)).sum(axis=1)

    c = (1 + raw_r).cumprod()
    hwm = c.rolling(252, min_periods=30).max()
    dd = c / hwm - 1
    dd_mult = (1 + dd / dd_floor).clip(0, 1).shift(1).fillna(1.0)

    rv = raw_r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    vm_raw = (target_vol / rv.replace(0, np.nan)).clip(lower=0.2, upper=3.0)
    gross_now = P.sum(axis=1).replace(0, np.nan)
    max_up = (1.0 / gross_now).clip(lower=1.0)
    vol_mult = np.minimum(vm_raw, max_up).shift(1).fillna(1.0)

    total_mult = dd_mult * vol_mult
    w_eff = P.mul(total_mult, axis=0)
    rs = w_eff.sum(axis=1)
    fs = np.minimum(1.0, 1.0 / rs.replace(0, np.nan)).fillna(1.0)
    w_eff = w_eff.mul(fs, axis=0)

    gross_ret = (w_eff.shift(1).fillna(0.0) * rets.reindex_like(w_eff).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w_eff.diff().abs().fillna(w_eff.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w_eff.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    net = gross_ret - drag
    return net, w_eff, sw


def main():
    op, cp = util.load_prices()
    print("Running APEX final strategy...")
    net, w_eff, sw = run(cp)

    print(f"\nSleeve breakdown:")
    for name, W in sw.items():
        r = S3._weights_to_returns(W, cp)
        m = util.metrics(r)
        print(f"  {name:10s}  SR={m['sharpe']:>5.2f}  CAGR={m['cagr']*100:>5.1f}%  "
              f"Vol={m['vol']*100:>5.1f}%  MDD={m['mdd']*100:>6.1f}%")

    print("\n=== APEX FINAL ===")
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", "2018-12-31")),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("RateHike 22", ("2022-01-01", "2022-12-31")),
                        ("Recovery 23-24", ("2023-01-01", "2024-12-31"))]:
        util.summarize(util.regime_slice(net, s, e), f"  {lbl}")

    net.to_frame("apex_net_ret").to_csv(OUT / "apex_final_returns.csv")
    w_eff.to_csv(OUT / "apex_final_weights.csv")
    # Metadata
    meta = {
        "blend_weights": BLEND_WEIGHTS,
        "target_vol": PORT_VOL_TARGET,
        "dd_floor": DD_FLOOR,
        "sleeve_vol": 0.15,
    }
    (OUT / "apex_final_meta.json").write_text(json.dumps(meta, indent=2))

    # Full metrics to JSON
    metrics = {
        "full": util.metrics(net),
        "is": util.metrics(util.regime_slice(net, "2005-01-01", "2018-12-31")),
        "oos": util.metrics(util.regime_slice(net, "2019-01-02", "2027-12-31")),
        "pre08": util.metrics(util.regime_slice(net, "2000-01-01", "2008-12-31")),
        "gfc": util.metrics(util.regime_slice(net, "2007-01-01", "2009-12-31")),
        "covid": util.metrics(util.regime_slice(net, "2020-01-01", "2020-12-31")),
        "ratehike22": util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31")),
        "rec2324": util.metrics(util.regime_slice(net, "2023-01-01", "2024-12-31")),
    }
    (OUT / "apex_final_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
