"""Parameter sweep for the v12 MULTI_SCAN sleeve.

Tests multiple (top_k, min_signals) combinations against the baseline
ensemble. For each config:
  * Run the standalone MULTI_SCAN sleeve to measure its OWN OOS SR
  * Run the full ensemble WITH that config to measure ensemble impact

Honest reporting — print all numbers, don't cherry-pick.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import json

from util import (DPY, OUT, SURVIVORS, DEAD, ALL_COINS, load_prices,
                  load_macro, safe_returns, eligibility, metrics,
                  regime_slice, weights_to_ret)
import sleeves as SV
import strategy as ST

OOS_START = "2022-07-01"


def run_standalone(cp, fn, name, tc_bps=20.0):
    """Run a single sleeve in isolation, return metrics."""
    W = fn(cp)
    r = weights_to_ret(W.shift(1).fillna(0.0), cp, tc_bps=tc_bps)
    m_full = metrics(r.dropna())
    m_oos = metrics(regime_slice(r, OOS_START, "2027-12-31"))
    return name, m_full, m_oos


def run_ensemble_with_scan(cp, macro, top_k, min_signals):
    """Build full ensemble with MULTI_SCAN configured to (top_k, min_signals).

    Returns full + OOS metrics on the net portfolio return.
    """
    sleeves = SV.build_all(cp, macro)
    # Override the MULTI_SCAN sleeve with the configured version
    sleeves["MULTI_SCAN"] = SV.sleeve_multi_scan(cp, macro, top_k=top_k,
                                                  min_signals=min_signals)
    net, _W = ST.build_portfolio(cp, sleeves)
    net = net.fillna(0.0)
    return metrics(net), metrics(regime_slice(net, OOS_START, "2027-12-31"))


def run_ensemble_no_scan(cp, macro):
    """Baseline: ensemble without MULTI_SCAN."""
    sleeves = SV.build_all(cp, macro)
    sleeves.pop("MULTI_SCAN", None)
    net, _W = ST.build_portfolio(cp, sleeves)
    net = net.fillna(0.0)
    return metrics(net), metrics(regime_slice(net, OOS_START, "2027-12-31"))


def main():
    print("=" * 80)
    print("CRYPTO-TITAN v12 — MULTI_SCAN parameter sweep")
    print("=" * 80)

    cp = load_prices()
    macro = load_macro(cp.index)
    print(f"Universe: {len(cp.columns)} coins, "
          f"{cp.index[0].date()} → {cp.index[-1].date()}")

    print("\n=== STANDALONE MULTI_SCAN sweeps ===")
    print(f"{'config':<18} {'full SR':>8} {'OOS SR':>8} {'CAGR':>8} {'MDD':>8} {'days':>6}")
    print("-" * 64)
    standalone_results = {}
    for top_k in [3, 5, 7, 10]:
        for min_sig in [3, 4, 5]:
            cfg = f"k={top_k} s={min_sig}"
            W = SV.sleeve_multi_scan(cp, macro, top_k=top_k,
                                     min_signals=min_sig)
            r = weights_to_ret(W.shift(1).fillna(0.0), cp, tc_bps=20.0).dropna()
            m_full = metrics(r)
            m_oos = metrics(regime_slice(r, OOS_START, "2027-12-31"))
            standalone_results[cfg] = (m_full, m_oos)
            print(f"  {cfg:<16} {m_full['sharpe']:>8.2f} {m_oos['sharpe']:>8.2f} "
                  f"{m_full['cagr']*100:>7.1f}% {m_full['mdd']*100:>7.1f}% {m_full['n']:>6d}")

    print("\n=== BASELINE ensemble (NO MULTI_SCAN) ===")
    m_base_full, m_base_oos = run_ensemble_no_scan(cp, macro)
    print(f"  full SR={m_base_full['sharpe']:.2f}  OOS SR={m_base_oos['sharpe']:.2f}  "
          f"CAGR={m_base_full['cagr']*100:.1f}%  MDD={m_base_full['mdd']*100:.1f}%  "
          f"NAV={m_base_full['nav']:.1f}")

    print("\n=== ENSEMBLE WITH MULTI_SCAN sweeps (vs baseline) ===")
    print(f"{'config':<18} {'full SR':>8} {'ΔSR':>6} {'OOS SR':>8} {'ΔOOS':>6} "
          f"{'CAGR':>8} {'MDD':>8}")
    print("-" * 70)
    ensemble_results = {}
    for top_k in [3, 5, 7]:
        for min_sig in [3, 4, 5]:
            cfg = f"k={top_k} s={min_sig}"
            m_full, m_oos = run_ensemble_with_scan(cp, macro, top_k, min_sig)
            ensemble_results[cfg] = (m_full, m_oos)
            d_full = m_full['sharpe'] - m_base_full['sharpe']
            d_oos = m_oos['sharpe'] - m_base_oos['sharpe']
            mark = " ←" if d_oos > 0.05 else ""
            print(f"  {cfg:<16} {m_full['sharpe']:>8.2f} {d_full:>+6.2f} "
                  f"{m_oos['sharpe']:>8.2f} {d_oos:>+6.2f} "
                  f"{m_full['cagr']*100:>7.1f}% {m_full['mdd']*100:>7.1f}%{mark}")

    print("\n=== RANKING by ENSEMBLE OOS SR ===")
    ranked = sorted(ensemble_results.items(),
                    key=lambda kv: -kv[1][1]['sharpe'])
    for cfg, (mf, mo) in ranked[:5]:
        print(f"  {cfg:<16}  Full SR {mf['sharpe']:.2f}, "
              f"OOS SR {mo['sharpe']:.2f}, CAGR {mf['cagr']*100:.1f}%, "
              f"MDD {mf['mdd']*100:.1f}%, NAV {mf['nav']:.1f}×")
    print(f"  baseline (no scan) Full {m_base_full['sharpe']:.2f}, "
          f"OOS {m_base_oos['sharpe']:.2f}, CAGR {m_base_full['cagr']*100:.1f}%, "
          f"MDD {m_base_full['mdd']*100:.1f}%, NAV {m_base_full['nav']:.1f}×")

    # Save sweep results
    out = {
        "as_of": str(cp.index[-1].date()),
        "universe_size": len(cp.columns),
        "baseline": {"full": m_base_full, "oos": m_base_oos},
        "standalone": {cfg: {"full": mf, "oos": mo}
                       for cfg, (mf, mo) in standalone_results.items()},
        "ensemble": {cfg: {"full": mf, "oos": mo}
                     for cfg, (mf, mo) in ensemble_results.items()},
    }
    fp = OUT / "multi_scan_sweep.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved sweep results to {fp}")


if __name__ == "__main__":
    main()
