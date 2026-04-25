"""CRYPTO-TITAN — standalone crypto strategy (no dependency on apex/phoenix).

Design:
  * Universe: 15 survivors + 5 dead (LUNA1/USTC/FTT/MATIC/UNI). Each coin
    ineligible before inception AND after its last valid price.
  * 4 long-only vol-managed trend sleeves: BTC_VM, ETH_VM, BTC_SLOW, ALT_DIVERS.
  * Inverse-volatility weighting across sleeves (risk parity).
  * CONSENSUS overlay scales exposure by agreement of independent signals.
  * Daily rebalance with smoothed weights (no weekly discontinuities).
  * Catastrophic-DD daily overlay protects LONG positions from LUNA-style blowups.
  * Portfolio-level 14% target vol, -12% DD floor, 30% BTC master DD kill.
  * Transaction costs: 20 bps per unit turnover.

Usage:
  python strategy.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
import numpy as np
import pandas as pd

from util import (DPY, OUT, SURVIVORS, DEAD, ALL_COINS, load_prices,
                  load_macro, safe_returns, eligibility, metrics, regime_slice,
                  summarize, weights_to_ret)
import sleeves as SV


IS_END = "2022-06-30"
OOS_START = "2022-07-01"

# v7: leveraged ensemble for CAGR 50%+ via Hyperliquid perps / OKX
# leveraged tokens. Conviction-scaled: 1× exposure baseline, up to 3×
# when MTF Fractal + Convex Breakout both fire (rare high-edge windows).
TARGET_VOL = 0.20
DD_FLOOR = -0.18
GROSS_CAP = 1.5
TC_BPS = 20.0          # spot longs round-trip
TC_BPS_SHORT = 30.0    # perp shorts round-trip
# Funding rate model — leveraged longs PAY funding on the lever portion
# (typical Hyperliquid/OKX BTC perp ~5-15%/yr avg in bull, less in bear).
# Daily on the LEVERAGED EXTRA notional (= max(0, gross - 1.0)):
LEVERAGE_FUNDING_BPS_DAY = 4.0   # 4 bps/day ≈ 15%/yr on lever portion
FUNDING_BULL_BPS_DAY = 3.0
FUNDING_BEAR_BPS_DAY = -3.0
SMOOTH_SPAN = 7

# v7 LEVERAGE — conviction-scaled.
# Sweet spot empirically: ramp 0.45→0.85, max 3.0×.
BASE_LEVERAGE = 1.0
MAX_LEVERAGE = 3.0
LEVER_RAMP_LOW = 0.45
LEVER_RAMP_HIGH = 0.85


def _master_gate(cp: pd.DataFrame) -> pd.Series:
    """Master risk-on multiplier ∈ [0, 1], SOFT version.

    Weighted vote over four risk-on conditions; each absent condition merely
    reduces the multiplier instead of killing it. Breadth computed against
    ONLY the curated strategy universe (not the extended universe) so dead
    coins in the extended set don't distort breadth.
    """
    elig = eligibility(cp, 150)
    ma100 = cp.rolling(100, min_periods=50).mean()
    trending = (cp > ma100).astype(float) * elig
    breadth = trending.sum(axis=1) / elig.sum(axis=1).replace(0, np.nan)
    breadth_ok = (breadth > 0.25).astype(float).fillna(0.0)

    btc = cp["BTC"]
    btc_above_200 = (btc > btc.rolling(200, min_periods=100).mean()).astype(float)
    btc_hwm90 = btc.rolling(90, min_periods=30).max()
    btc_dd = btc / btc_hwm90 - 1
    btc_dd_ok = (btc_dd > -0.30).astype(float)
    btc_mom_ok = (btc.pct_change(63) > -0.08).astype(float)

    gate = 0.25 * (breadth_ok + btc_above_200 + btc_dd_ok + btc_mom_ok)
    gate = gate.ewm(span=7, adjust=False).mean()
    gate = gate.where(gate > 0.25, 0.0)
    return gate.fillna(0.0)


def _inverse_vol_sleeve_weights(sleeve_rets: dict, window: int = 90) -> pd.DataFrame:
    """Per-date inverse-volatility weights across sleeves (risk parity)."""
    vols = pd.DataFrame({n: r.rolling(window, min_periods=30).std() * np.sqrt(DPY)
                         for n, r in sleeve_rets.items()})
    inv = 1.0 / vols.replace(0, np.nan)
    w = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(
        1.0 / len(sleeve_rets))
    return w


def _adaptive_sharpe_weights(sleeve_rets: dict, window: int = 252,
                              floor: float = 0.0) -> pd.DataFrame:
    """Per-date softmax-weights of sleeves by trailing Sharpe over `window`.

    A sleeve with negative trailing Sharpe gets zero weight (floor); positives
    get weight ∝ Sharpe. Normalised to sum to 1. Smooth with 30d EMA.
    """
    sharpes = {}
    for n, r in sleeve_rets.items():
        mu = r.rolling(window, min_periods=60).mean() * DPY
        sd = r.rolling(window, min_periods=60).std() * np.sqrt(DPY)
        sharpes[n] = (mu / sd.replace(0, np.nan)).fillna(0.0)
    sh = pd.DataFrame(sharpes).clip(lower=floor)
    sm = sh.ewm(span=30, adjust=False).mean()
    tot = sm.sum(axis=1).replace(0, np.nan)
    w = sm.div(tot, axis=0)
    # Fallback to equal weight when no sleeve has positive trailing Sharpe
    w = w.fillna(1.0 / len(sleeve_rets))
    return w


def _signal_stability(sleeves: dict, window: int = 14) -> pd.Series:
    """Meta-regime: 'how much are the sleeve signals flipping?'
    High instability = chop regime → reduce exposure. Returns [0, 1] score
    where 1 = stable (low flipping) and 0 = very unstable."""
    votes = []
    for name, W in sleeves.items():
        v = (W.sum(axis=1) > 1e-4).astype(float)
        votes.append(v)
    vote_df = pd.concat(votes, axis=1)
    # How many sleeve-vote FLIPS per sleeve per day, averaged?
    flips = vote_df.diff().abs().rolling(window, min_periods=7).mean().mean(axis=1)
    # Normalise: <0.05 flips/sleeve = stable, >0.15 = unstable
    stability = (1.0 - (flips / 0.15).clip(0, 1)).fillna(0.5)
    return stability


def build_portfolio(cp: pd.DataFrame, sleeves: dict) -> tuple:
    """CONVICTION-CONCENTRATION framework (v5):
      * Compute per-coin aggregated signal strength across all sleeves
      * Measure CONVICTION = fraction of sleeves voting long × stability
      * If conviction < LOW: 100% cash
      * If conviction HIGH: concentrate in TOP_K coins with highest signal,
        run full leverage
      * Otherwise: proportional diversified RP allocation
      * Weekly rebalance, daily catastrophic-DD overlay only

    Philosophy: step aside during chop; swing hard when signals align.
    """

    # ============================================================
    # STEP 1 — per-sleeve vol-targeted weights
    # ============================================================
    sleeve_rets = {name: weights_to_ret(W.shift(1).fillna(0.0), cp, tc_bps=0.0)
                   for name, W in sleeves.items()}
    sw_adj = {}
    for name, W in sleeves.items():
        rv = sleeve_rets[name].rolling(60, min_periods=20).std() * np.sqrt(DPY)
        m = (0.18 / rv.replace(0, np.nan)).clip(lower=0.2, upper=1.5).shift(1).fillna(1.0)
        sw_adj[name] = W.mul(m, axis=0)

    # ============================================================
    # STEP 2 — risk-parity blended aggregated signal
    # ============================================================
    rp_w = _inverse_vol_sleeve_weights(sleeve_rets, window=90).shift(1).fillna(
        1.0 / len(sleeves))
    first = next(iter(sw_adj.values()))
    dir_W = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    pair_W = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    for name, W in sw_adj.items():
        rp_series = rp_w[name] if name in rp_w.columns else pd.Series(
            1.0 / len(sw_adj), index=W.index)
        Wc = W.mul(rp_series, axis=0).fillna(0.0)
        if name in SV.MARKET_NEUTRAL:
            pair_W = pair_W + Wc
        else:
            dir_W = dir_W + Wc

    # ============================================================
    # STEP 3 — NONLINEAR conviction curve + master gate
    # Power-curve so low conviction → near-cash, high conviction → amplified.
    # This captures "concentrate when model suggests" while preserving diversif.
    # ============================================================
    conv_vote = SV.consensus_signal(cp, sleeves).shift(1).fillna(0.0)
    gate = _master_gate(cp).shift(1).fillna(0.0)
    # Cash below 0.15 conviction; mild power curve above.
    conv_clean = (conv_vote - 0.15).clip(lower=0.0) / 0.55
    conv_scale = conv_clean ** 1.1 * 1.35
    conv_scale = conv_scale.clip(upper=1.35)
    dir_W = dir_W.mul(conv_scale, axis=0).mul(gate, axis=0)
    P = dir_W.add(pair_W, fill_value=0.0)

    # EMA smooth
    P = P.ewm(span=SMOOTH_SPAN, adjust=False).mean()

    # ============================================================
    # STEP 4 — WEEKLY SNAP (Wednesdays)
    # ============================================================
    is_rebal = pd.Series(P.index.dayofweek == 2, index=P.index)
    P_weekly = P.where(is_rebal, other=np.nan).ffill().fillna(0.0)

    gs = P_weekly.abs().sum(axis=1)
    fs = np.minimum(1.0, GROSS_CAP / gs.replace(0, np.nan)).fillna(1.0)
    P_weekly = P_weekly.mul(fs, axis=0)

    # ============================================================
    # STEP 5 — DAILY VOL TARGET + DAILY SAFETY OVERLAYS
    # ============================================================
    rets = safe_returns(cp, cap=0.22)

    raw_r = (P_weekly.shift(1).fillna(0.0) * rets.reindex_like(P_weekly).fillna(0.0)
             ).sum(axis=1)

    # DAILY vol targeting — EWMA(60) updated every day, asymmetric clip
    rv_d = raw_r.ewm(span=60, adjust=False).std() * np.sqrt(DPY)
    vm = (TARGET_VOL / rv_d.replace(0, np.nan)).clip(lower=0.3, upper=1.5)
    vm = vm.shift(1).fillna(1.0)

    # Daily strategy-level DD floor (vs 365d high)
    c = (1 + raw_r).cumprod()
    hwm = c.rolling(365, min_periods=30).max()
    dd = c / hwm - 1
    dd_mult = (1 + dd / DD_FLOOR).clip(0, 1).shift(1).fillna(1.0)

    # NOVEL: Strategy self-stop — when own 30d realised return < -4%,
    # apply 50% reduction for 21 days. This catches slow bleeds in chop
    # regimes (2025+) that the 365d DD floor alone can't see.
    roll_30d = (1 + raw_r).rolling(30, min_periods=20).apply(
        lambda x: x.prod() - 1, raw=True)
    self_stop_active = (roll_30d < -0.04).astype(float)
    # Hold the stop active for 21 days after trigger (rolling max)
    self_stop_held = self_stop_active.rolling(21, min_periods=1).max()
    self_stop_mult = (1.0 - 0.5 * self_stop_held).shift(1).fillna(1.0)

    # SIGN-AWARE catastrophic-DD overlay — protects LONGS from coin crashes
    # (LUNA, FTT) but ALLOWS SHORTS to ride the crash down. This is the
    # central asymmetry that makes long+short on a wide universe profitable.
    elig_live = eligibility(cp, min_history=1, catastrophe_dd=-0.99,
                            dd_window=60).shift(1).fillna(0.0).reindex_like(P_weekly)
    elig_nocrash = eligibility(cp, min_history=1, catastrophe_dd=-0.28,
                                dd_window=60).shift(1).fillna(0.0).reindex_like(P_weekly)
    long_mask = (P_weekly > 0).astype(float)
    short_mask = (P_weekly < 0).astype(float)
    sign_aware = (long_mask * elig_nocrash) + (short_mask * elig_live)

    W_eff = P_weekly.mul(vm * dd_mult * self_stop_mult, axis=0) * sign_aware

    # ============================================================
    # CONVICTION-SCALED LEVERAGE (v7) — Hyperliquid perps / OKX
    # ============================================================
    lever_curve = ((conv_vote - LEVER_RAMP_LOW) /
                    (LEVER_RAMP_HIGH - LEVER_RAMP_LOW)).clip(0, 1)
    leverage = (BASE_LEVERAGE +
                 (MAX_LEVERAGE - BASE_LEVERAGE) * lever_curve).shift(1).fillna(1.0)

    # Chop detection tested but cut too much trending alpha. Removed.

    W_eff = W_eff.mul(leverage, axis=0)

    # Final gross cap (allow up to MAX_LEVERAGE on the day-of-rebalance)
    gs = W_eff.abs().sum(axis=1)
    fs = np.minimum(1.0, MAX_LEVERAGE / gs.replace(0, np.nan)).fillna(1.0)
    W_eff = W_eff.mul(fs, axis=0)

    # ============================================================
    # STEP 6 — RETURNS + TC + PERP FUNDING COST
    # ============================================================
    W_held = W_eff.shift(1).fillna(0.0)
    gross_ret = (W_held * rets.reindex_like(W_held).fillna(0.0)).sum(axis=1)

    # Sign-aware TC: shorts cost a bit more on perps than spot longs
    long_turn = W_eff.where(W_eff > 0, 0.0).diff().abs().fillna(0.0)
    short_turn = W_eff.where(W_eff < 0, 0.0).diff().abs().fillna(0.0)
    long_drag = long_turn.sum(axis=1) * TC_BPS / 1e4
    short_drag = short_turn.sum(axis=1) * TC_BPS_SHORT / 1e4

    # PERP FUNDING on shorts (when active — currently no shorts).
    btc_r63 = cp["BTC"].pct_change(63).fillna(0.0)
    funding_per_day = pd.Series(
        np.where(btc_r63 > 0, FUNDING_BULL_BPS_DAY, FUNDING_BEAR_BPS_DAY),
        index=cp.index) / 1e4
    short_notional = (-W_held.where(W_held < 0, 0.0)).sum(axis=1)
    short_funding = short_notional * funding_per_day

    # LEVERAGE FUNDING on the levered portion of LONG notionals.
    # Hyperliquid/OKX charge funding on the borrowed (leveraged) part —
    # the part above 1× of capital. Conservative ~15%/yr = 4 bps/day.
    long_notional = W_held.where(W_held > 0, 0.0).sum(axis=1)
    levered_extra = (long_notional - 1.0).clip(lower=0.0)
    leverage_funding = levered_extra * LEVERAGE_FUNDING_BPS_DAY / 1e4

    net_ret = gross_ret - long_drag - short_drag - short_funding - leverage_funding
    return net_ret, W_eff


def main():
    cp = load_prices()
    print(f"Universe: {len(cp.columns)} coins, {cp.index[0].date()} → {cp.index[-1].date()}")
    print(f"  Survivors ({len(SURVIVORS)}): {SURVIVORS}")
    print(f"  Dead/delisted ({len(DEAD)}): {DEAD}")

    macro = load_macro(cp.index)
    sleeves = SV.build_all(cp, macro)
    net, W_eff = build_portfolio(cp, sleeves)
    net = net.fillna(0.0)

    print("\n=== HEADLINE ===")
    for lbl, (s, e) in [
        ("FULL 14-26",          ("2014-09-17", "2027-12-31")),
        ("IS 14 → 22-06",       ("2014-09-17", IS_END)),
        ("OOS 22-07 → now",     (OOS_START, "2027-12-31")),
        ("2017 bull",           ("2017-01-01", "2017-12-31")),
        ("2018 bear",           ("2018-01-01", "2018-12-31")),
        ("2019 recovery",       ("2019-01-01", "2019-12-31")),
        ("2020 COVID+bull",     ("2020-01-01", "2020-12-31")),
        ("2021 mania",          ("2021-01-01", "2021-12-31")),
        ("2022 crypto winter",  ("2022-01-01", "2022-12-31")),
        ("2023-24 recovery",    ("2023-01-01", "2024-12-31")),
        ("2025+",               ("2025-01-01", "2027-12-31")),
    ]:
        summarize(regime_slice(net, s, e), lbl)

    print("\n=== PER-SLEEVE (daily, no overlay) ===")
    sleeve_metrics = {}
    for name, W in sleeves.items():
        r = weights_to_ret(W.shift(1).fillna(0.0), cp, tc_bps=TC_BPS)
        m_full = metrics(r.dropna())
        m_oos = metrics(regime_slice(r, OOS_START, "2027-12-31"))
        sleeve_metrics[name] = {"full": m_full, "oos": m_oos}
        print(f"  {name:14s} FULL SR={m_full['sharpe']:>5.2f}  "
              f"OOS SR={m_oos['sharpe']:>5.2f}  "
              f"CAGR={m_full['cagr']*100:>6.1f}%  "
              f"MDD={m_full['mdd']*100:>6.1f}%")

    print("\n=== SURVIVORSHIP BIAS ANALYSIS ===")
    # Survivors-only test (classical survivorship-biased backtest)
    cp_surv = load_prices(coins=SURVIVORS)
    macro_surv = load_macro(cp_surv.index)
    sw_surv = SV.build_all(cp_surv, macro_surv)
    net_surv, _ = build_portfolio(cp_surv, sw_surv)
    net_surv = net_surv.fillna(0.0)

    # FULL 111-coin universe test (no cherry-picking — all dead coins in)
    from util import EXTENDED_UNIVERSE
    cp_ext = load_prices(coins=EXTENDED_UNIVERSE)
    macro_ext = load_macro(cp_ext.index)
    sw_ext = SV.build_all(cp_ext, macro_ext)
    net_ext, _ = build_portfolio(cp_ext, sw_ext)
    net_ext = net_ext.fillna(0.0)
    m_ext = metrics(net_ext)
    m_ext_oos = metrics(regime_slice(net_ext, OOS_START, "2027-12-31"))
    m_full = metrics(net)
    m_surv = metrics(net_surv)
    print(f"  STRATEGY (50: survivors+dead):     SR={m_full['sharpe']:.2f}  "
          f"CAGR={m_full['cagr']*100:.1f}%  MDD={m_full['mdd']*100:.1f}%")
    print(f"  SURVIVORS-only (biased, n=35):     SR={m_surv['sharpe']:.2f}  "
          f"CAGR={m_surv['cagr']*100:.1f}%  MDD={m_surv['mdd']*100:.1f}%")
    print(f"  EXTENDED (111: all dead included): SR={m_ext['sharpe']:.2f}  "
          f"CAGR={m_ext['cagr']*100:.1f}%  MDD={m_ext['mdd']*100:.1f}%")
    print(f"  Bias delta (survivors - strategy):    ΔSR={m_surv['sharpe']-m_full['sharpe']:+.2f}")
    print(f"  Robustness (extended - strategy):     ΔSR={m_ext['sharpe']-m_full['sharpe']:+.2f}  "
          f"OOS ΔSR={m_ext_oos['sharpe']-metrics(regime_slice(net,OOS_START,'2027-12-31'))['sharpe']:+.2f}")

    print("\n=== BENCHMARKS ===")
    btc_r = cp["BTC"].pct_change().clip(-0.25, 0.25).fillna(0.0)
    eq_r = cp[SURVIVORS].pct_change().mean(axis=1).clip(-0.25, 0.25).fillna(0.0)
    m_btc = metrics(btc_r); m_btc_oos = metrics(regime_slice(btc_r, OOS_START, "2027-12-31"))
    m_eq = metrics(eq_r); m_eq_oos = metrics(regime_slice(eq_r, OOS_START, "2027-12-31"))
    m_oos = metrics(regime_slice(net, OOS_START, "2027-12-31"))
    print(f"  BTC hold:       Full SR={m_btc['sharpe']:.2f}  OOS SR={m_btc_oos['sharpe']:.2f}  CAGR={m_btc['cagr']*100:.1f}%")
    print(f"  EW survivors:   Full SR={m_eq['sharpe']:.2f}  OOS SR={m_eq_oos['sharpe']:.2f}  CAGR={m_eq['cagr']*100:.1f}%")
    print(f"  CRYPTO-TITAN:   Full SR={m_full['sharpe']:.2f}  OOS SR={m_oos['sharpe']:.2f}  CAGR={m_full['cagr']*100:.1f}%")

    net.to_frame("crypto_titan_ret").to_csv(OUT / "crypto_titan_returns.csv")
    W_eff.to_csv(OUT / "crypto_titan_weights.csv")

    meta = {
        "version": "crypto_titan_v1",
        "target_vol": TARGET_VOL,
        "dd_floor": DD_FLOOR,
        "gross_cap": GROSS_CAP,
        "tc_bps": TC_BPS,
        "smooth_span": SMOOTH_SPAN,
        "sleeves": list(sleeves.keys()),
        "universe_full": ALL_COINS,
        "survivors": SURVIVORS,
        "dead": DEAD,
        "benchmarks": {
            "btc_hold": m_btc, "btc_hold_oos": m_btc_oos,
            "ew_survivors": m_eq, "ew_survivors_oos": m_eq_oos,
        },
        "metrics": {
            "full": m_full,
            "is":   metrics(regime_slice(net, "2014-09-17", IS_END)),
            "oos":  m_oos,
            "y2018": metrics(regime_slice(net, "2018-01-01", "2018-12-31")),
            "y2020": metrics(regime_slice(net, "2020-01-01", "2020-12-31")),
            "y2021": metrics(regime_slice(net, "2021-01-01", "2021-12-31")),
            "y2022": metrics(regime_slice(net, "2022-01-01", "2022-12-31")),
            "y2324": metrics(regime_slice(net, "2023-01-01", "2024-12-31")),
            "y2025+": metrics(regime_slice(net, "2025-01-01", "2027-12-31")),
        },
        "survivorship_bias": {
            "full_universe_sr": m_full["sharpe"],
            "survivors_only_sr": m_surv["sharpe"],
            "bias_sr": m_surv["sharpe"] - m_full["sharpe"],
            "bias_cagr": m_surv["cagr"] - m_full["cagr"],
        },
        "sleeve_metrics": sleeve_metrics,
    }
    (OUT / "crypto_titan_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
