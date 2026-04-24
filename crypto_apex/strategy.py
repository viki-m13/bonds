"""CRYPTO-APEX strategy — APEX methodology on broad crypto universe.

Six sleeves + portfolio overlays + survivorship-bias analysis.

Usage:
  python strategy.py          # full run + save returns
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
import numpy as np
import pandas as pd
import util
from util import DPY, load_prices, load_macro, metrics, summarize, regime_slice, SURVIVORS, DEAD, ALL_COINS, OUT, _weights_to_ret
import sleeves as SV

IS_END = "2022-06-30"  # First ~70% of BTC history
OOS_START = "2022-07-01"


def build_portfolio(cp: pd.DataFrame, sleeve_weights: dict,
                    target_vol: float = 0.30, dd_floor: float = -0.25,
                    gross_cap: float = 1.0, tc_bps: float = 30.0,
                    macro: dict = None) -> pd.Series:
    """Aggregate sleeves with equal weight, apply overlays, return net daily series.

    target_vol: 30% annualized (crypto is 3-5x equity vol; still aggressive but controlled)
    dd_floor: -25% (crypto drawdowns routinely exceed 50%)
    """
    # Vol-target each sleeve to 25% first
    sw_adj = {}
    for name, W in sleeve_weights.items():
        r = _weights_to_ret(W, cp, tc_bps=tc_bps)
        rv = r.rolling(60, min_periods=20).std() * np.sqrt(DPY)
        m = (0.25 / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
        sw_adj[name] = W.mul(m, axis=0)

    # Equal-weight aggregation
    first = next(iter(sw_adj.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    n = len(sw_adj)
    for name, W in sw_adj.items():
        P = P + W.fillna(0.0) / n

    gross = P.sum(axis=1)
    scale = np.minimum(1.0, gross_cap / gross.replace(0, np.nan)).fillna(1.0)
    P = P.mul(scale, axis=0)

    # MASTER BTC KILL-SWITCH — crypto winter = full cash.
    btc = cp["BTC"]
    btc_hwm90 = btc.rolling(90, min_periods=30).max()
    btc_dd = btc / btc_hwm90 - 1
    btc_above_200 = (btc > btc.rolling(200).mean()).astype(float)
    btc_63d_mom = btc.pct_change(63)
    # Cash if: BTC below 200MA OR BTC 63d mom < -10% OR DD > 30% from 90d high
    cash = ((btc_above_200 < 0.5) | (btc_63d_mom < -0.10) | (btc_dd < -0.30)).astype(float)
    master_mult = (1 - cash).shift(1).fillna(1.0)
    P = P.mul(master_mult, axis=0)

    rets = util.safe_returns(cp, cap=0.30)

    raw_r = (P.shift(1).fillna(0.0) * rets.reindex_like(P).fillna(0.0)).sum(axis=1)
    c = (1 + raw_r).cumprod()
    hwm = c.rolling(365, min_periods=30).max()
    dd = c / hwm - 1
    dd_mult = (1 + dd / dd_floor).clip(0, 1).shift(1).fillna(1.0)

    rv = raw_r.rolling(60, min_periods=20).std() * np.sqrt(DPY)
    vm = (target_vol / rv.replace(0, np.nan)).clip(lower=0.2, upper=2.0).shift(1).fillna(1.0)

    total_mult = dd_mult * vm
    w_eff = P.mul(total_mult, axis=0)
    rs = w_eff.sum(axis=1)
    fs = np.minimum(1.0, gross_cap / rs.replace(0, np.nan)).fillna(1.0)
    w_eff = w_eff.mul(fs, axis=0)

    gross_ret = (w_eff.shift(1).fillna(0.0) * rets.reindex_like(w_eff).fillna(0.0)).sum(axis=1)
    dw = w_eff.diff().abs().fillna(w_eff.abs())
    drag = dw.sum(axis=1).shift(1).fillna(0.0) * tc_bps / 1e4
    return gross_ret - drag


FINAL_SLEEVES = ["ACCEL", "HURST", "DOMINANCE"]
TARGET_VOL = 0.25
DD_FLOOR = -0.25


def main():
    cp = load_prices()
    print(f"Universe: {len(cp.columns)} coins, {cp.index[0].date()} to {cp.index[-1].date()}")
    print(f"  Survivors: {len(SURVIVORS)}, Dead/delisted: {len(DEAD)}")

    macro = load_macro(cp.index)
    all_sw = SV.build_all(cp, macro)
    sw = {k: all_sw[k] for k in FINAL_SLEEVES}

    net = build_portfolio(cp, sw, target_vol=TARGET_VOL, dd_floor=DD_FLOOR)
    net = net.fillna(0.0)

    print("\n=== HEADLINE ===")
    for lbl, (s, e) in [
        ("FULL 14-26", ("2014-09-17", "2027-12-31")),
        ("IS 14-22-06", ("2014-09-17", IS_END)),
        ("OOS 22-07+", (OOS_START, "2027-12-31")),
        ("2018 bear", ("2018-01-01", "2018-12-31")),
        ("2020 COVID+bull", ("2020-01-01", "2020-12-31")),
        ("2021 mania", ("2021-01-01", "2021-12-31")),
        ("2022 crypto winter", ("2022-01-01", "2022-12-31")),
        ("2023-24 recovery", ("2023-01-01", "2024-12-31")),
        ("2025+ live", ("2025-01-01", "2027-12-31")),
    ]:
        summarize(regime_slice(net, s, e), lbl)

    # Save primary returns
    net.to_frame("crypto_apex_ret").to_csv(OUT / "crypto_apex_returns.csv")

    # === SURVIVORSHIP BIAS ANALYSIS ===
    print("\n=== SURVIVORSHIP BIAS ANALYSIS ===")
    cp_surv = load_prices(coins=SURVIVORS)
    sw_surv_all = SV.build_all(cp_surv, macro)
    sw_surv = {k: sw_surv_all[k] for k in FINAL_SLEEVES}
    net_surv = build_portfolio(cp_surv, sw_surv, target_vol=TARGET_VOL, dd_floor=DD_FLOOR).fillna(0.0)

    m_full = metrics(net)
    m_surv = metrics(net_surv)
    print(f"  FULL universe (survivors+dead):  SR={m_full['sharpe']:.2f}, CAGR={m_full['cagr']*100:.1f}%, MDD={m_full['mdd']*100:.1f}%")
    print(f"  SURVIVORS-ONLY (biased):          SR={m_surv['sharpe']:.2f}, CAGR={m_surv['cagr']*100:.1f}%, MDD={m_surv['mdd']*100:.1f}%")
    print(f"  Survivorship bias impact:         dSR={m_surv['sharpe']-m_full['sharpe']:+.2f}, "
          f"dCAGR={(m_surv['cagr']-m_full['cagr'])*100:+.1f}%")

    # === SLEEVE CONTRIBUTIONS ===
    print("\n=== PER-SLEEVE OOS SR ===")
    sleeve_metrics = {}
    for name, W in sw.items():
        r = _weights_to_ret(W, cp)
        m = metrics(regime_slice(r, OOS_START, "2027-12-31"))
        sleeve_metrics[name] = m
        print(f"  {name:12s} OOS SR={m['sharpe']:>5.2f}  CAGR={m['cagr']*100:>6.1f}%  MDD={m['mdd']*100:>6.1f}%")

    # === BENCHMARKS ===
    print("\n=== BENCHMARKS (buy-and-hold, survivor-only) ===")
    btc_r = cp["BTC"].pct_change().fillna(0.0).clip(-0.30, 0.30)
    eq_r = cp[SURVIVORS].pct_change().mean(axis=1).fillna(0.0).clip(-0.30, 0.30)
    m_btc = metrics(btc_r)
    m_eq = metrics(eq_r)
    m_btc_oos = metrics(regime_slice(btc_r, OOS_START, "2027-12-31"))
    m_eq_oos = metrics(regime_slice(eq_r, OOS_START, "2027-12-31"))
    print(f"  BTC hold (capped):       Full SR={m_btc['sharpe']:.2f}, CAGR={m_btc['cagr']*100:.1f}%, MDD={m_btc['mdd']*100:.1f}%, OOS SR={m_btc_oos['sharpe']:.2f}")
    print(f"  Equal-weight survivors:  Full SR={m_eq['sharpe']:.2f}, CAGR={m_eq['cagr']*100:.1f}%, MDD={m_eq['mdd']*100:.1f}%, OOS SR={m_eq_oos['sharpe']:.2f}")
    print(f"  CRYPTO-APEX:             Full SR={m_full['sharpe']:.2f}, CAGR={m_full['cagr']*100:.1f}%, MDD={m_full['mdd']*100:.1f}%, OOS SR={metrics(regime_slice(net,OOS_START,'2027-12-31'))['sharpe']:.2f}")

    # Save meta
    meta = {
        "version": "crypto_apex_v1",
        "sleeves": list(sw.keys()),
        "universe_full": ALL_COINS,
        "survivors": SURVIVORS,
        "dead": DEAD,
        "target_vol": TARGET_VOL,
        "dd_floor": DD_FLOOR,
        "tc_bps": 30.0,
        "benchmarks": {
            "btc_hold": m_btc,
            "equal_weight_survivors": m_eq,
            "btc_hold_oos": m_btc_oos,
            "eq_weight_oos": m_eq_oos,
        },
        "metrics": {
            "full": metrics(net),
            "is": metrics(regime_slice(net, "2014-09-17", IS_END)),
            "oos": metrics(regime_slice(net, OOS_START, "2027-12-31")),
            "y2018": metrics(regime_slice(net, "2018-01-01", "2018-12-31")),
            "y2020": metrics(regime_slice(net, "2020-01-01", "2020-12-31")),
            "y2021": metrics(regime_slice(net, "2021-01-01", "2021-12-31")),
            "y2022": metrics(regime_slice(net, "2022-01-01", "2022-12-31")),
            "y2324": metrics(regime_slice(net, "2023-01-01", "2024-12-31")),
            "y2025plus": metrics(regime_slice(net, "2025-01-01", "2027-12-31")),
        },
        "survivorship_bias": {
            "full_universe_sr": m_full["sharpe"],
            "survivors_only_sr": m_surv["sharpe"],
            "bias_sr": m_surv["sharpe"] - m_full["sharpe"],
            "bias_cagr": m_surv["cagr"] - m_full["cagr"],
        },
        "sleeve_oos": {name: m["sharpe"] for name, m in sleeve_metrics.items()},
    }
    (OUT / "crypto_apex_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nSaved returns and meta to {OUT}")


if __name__ == "__main__":
    main()
