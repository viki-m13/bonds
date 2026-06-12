"""Bear-market behavior for the biweekly DCA picker.

Question set:
  1. What should the biweekly contribution buy when the regime is risk-off?
     (bear "sleeves" paired with the 6m-momentum bull sleeve via a regime
     switch)
  2. Can recovery triggers (breadth thrust, VIX percentile relaxation,
     SPY>50dma after deep drawdown) switch back to the bull sleeve earlier
     than the 200dma?
  3. Do any sell/exit rules actually help? (HY-OAS panic liquidation,
     per-stock 300dma trailing stop, vs no-sell control)
  4. VIX-scaled aggressiveness: in very-high-VIX periods pick higher-beta
     rebound candidates instead of defensives.

Causality: every series is a trailing rolling window, expanding max, or a
non-negative shift; regime features come from regime.build_regime which is
trailing-only. State machines iterate forward in time and use only the
current/previous day's values. Cross-sectional ranks are within row d.

Run:  python research/signals_bear.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data as data_mod
import protocol
import regime

# ---------------------------------------------------------------- shared

_C: dict = {}


def shared():
    if "P" not in _C:
        P = data_mod.build_panel()
        _C["P"] = P
        idx = P["close"].index
        _C["R"] = regime.build_regime(idx)
        spy = data_mod.load_benchmark("SPY")["Close"].reindex(idx).ffill()
        _C["spy"] = spy
        _C["spy_ma50"] = spy.rolling(50).mean()
    return _C


def _rets(P):
    if "rets" not in _C:
        _C["rets"] = P["close"].pct_change(fill_method=None)
    return _C["rets"]


def _vol126(P):
    if "vol126" not in _C:
        _C["vol126"] = _rets(P).rolling(126, min_periods=100).std()
    return _C["vol126"]


def _beta252(P):
    """Trailing 252d beta vs SPY (mean-of-products minus product-of-means)."""
    if "beta252" not in _C:
        r = _rets(P)
        m = shared()["spy"].pct_change()
        W = 252
        rm = r.mul(m, axis=0)
        cov = (rm.rolling(W, min_periods=200).mean()
               - r.rolling(W, min_periods=200).mean()
               * m.rolling(W, min_periods=200).mean().values[:, None])
        var = (m.pow(2).rolling(W, min_periods=200).mean()
               - m.rolling(W, min_periods=200).mean() ** 2)
        _C["beta252"] = cov.div(var, axis=0)
    return _C["beta252"]


def _dd_ath(P):
    """Drawdown from all-time high (expanding max of close; causal)."""
    if "dd_ath" not in _C:
        c = P["close"]
        _C["dd_ath"] = c / c.cummax() - 1.0
    return _C["dd_ath"]


def _rs12(P):
    """12m return minus SPY 12m return."""
    if "rs12" not in _C:
        c = P["close"]
        spy = shared()["spy"]
        _C["rs12"] = (c / c.shift(252) - 1.0).sub(
            spy / spy.shift(252) - 1.0, axis=0)
    return _C["rs12"]


def _quality_uptrend(P):
    """Long-term uptrend intact: above 400d MA OR positive 24m return."""
    if "qual" not in _C:
        c = P["close"]
        ma400 = c.rolling(400, min_periods=300).mean()
        r24 = c / c.shift(504) - 1.0
        _C["qual"] = (c > ma400) | (r24 > 0)
    return _C["qual"]


def mom126(P):
    c = P["close"]
    return c / c.shift(126) - 1.0


# ---------------------------------------------------------------- bear sleeves

def bear_lowvol_rs(P):
    """(a) Lowest vol among names with positive 12m RS vs SPY."""
    return (-_vol126(P)).where(_rs12(P) > 0)


def bear_defquality(P):
    """(b) Defensive quality: rank(low vol) + rank(shallow drawdown-from-ATH)."""
    mem = P["member"]
    rv = (-_vol126(P)).where(mem).rank(axis=1, pct=True)
    rd = _dd_ath(P).where(mem).rank(axis=1, pct=True)   # dd<=0; higher=closer ATH
    return rv + rd


def bear_rebound(P):
    """(c) Max-drawdown rebounders: quality uptrend names 30-60% below ATH,
    scored by discount depth (deeper = better)."""
    dd = _dd_ath(P)
    elig = _quality_uptrend(P) & (dd <= -0.30) & (dd >= -0.60)
    return (-dd).where(elig)


def bear_rebound_fallback(P):
    """(c') Rebounders with low-vol fallback so the sleeve is never empty:
    rebounders score in [2,3); fallback low-vol-RS names rank in [0,1)."""
    reb = bear_rebound(P)
    fb = bear_lowvol_rs(P).where(P["member"]).rank(axis=1, pct=True)
    return (2.0 + (-_dd_ath(P)).clip(0, 0.9)).where(reb.notna()).fillna(fb)


def bear_mom_lowbeta(P):
    """(d) 6m momentum restricted to the low-beta half of the membership."""
    b = _beta252(P).where(P["member"])
    med = b.median(axis=1)
    return mom126(P).where(b.le(med, axis=0))


def bear_vix_aggro(P):
    """(4) High-beta rebound candidates: quality uptrend, >=20% off ATH,
    scored by beta (most aggressive first)."""
    dd = _dd_ath(P)
    elig = _quality_uptrend(P) & (dd <= -0.20)
    return _beta252(P).where(elig)


# ---------------------------------------------------------------- regime logic

def bear_mask_base():
    """Base bear definition: SPY<200dma OR breadth_200_ma10 < 0.4."""
    R = shared()["R"]
    return (R["spy_above_200"] < 1) | (R["breadth_200_ma10"] < 0.40)


def recovery_triggers():
    """Daily booleans, all trailing-only."""
    sh = shared()
    R, spy, ma50 = sh["R"], sh["spy"], sh["spy_ma50"]
    b = R["breadth_200_ma10"]
    thrust = (b > 0.5) & (b.rolling(42).min() < 0.30)          # <0.3 -> >0.5 in ~2m
    vix_rlx = (R["vix_pct3y"] < 0.75) & (R["vix_pct3y"].rolling(63).max() > 0.90)
    spy50 = (spy > ma50) & (R["spy_dd"] < -0.20)               # 50dma reclaim in deep dd
    return pd.DataFrame({"thrust": thrust, "vix_rlx": vix_rlx, "spy50": spy50})


def risk_on_with_recovery(triggers: list[str]):
    """Risk-on = ~base-bear, plus an early-on latch: while base says bear,
    any listed trigger flips risk-on until the next fresh bear entry
    (base on->off transition) resets the latch. Forward-only state machine."""
    bear = bear_mask_base().to_numpy()
    trig = recovery_triggers()[triggers].any(axis=1).to_numpy() if triggers \
        else np.zeros(len(bear), bool)
    on = np.empty(len(bear), bool)
    latch = False
    prev_bear = False
    for i in range(len(bear)):
        if not bear[i]:
            latch = False
            on[i] = True
        else:
            if not prev_bear:
                latch = False          # fresh bear entry resets the latch
            if trig[i]:
                latch = True
            on[i] = latch
        prev_bear = bear[i]
    idx = shared()["P"]["close"].index
    return pd.Series(on, index=idx)


def switch(bull: pd.DataFrame, bear: pd.DataFrame | None,
           on: pd.Series) -> pd.DataFrame:
    """Bull sleeve when risk-on, bear sleeve (or cash=NaN) when risk-off."""
    if bear is None:
        bear_v = np.full(bull.shape, np.nan)
    else:
        bear_v = bear.to_numpy(float)
    out = np.where(on.to_numpy()[:, None], bull.to_numpy(float), bear_v)
    return pd.DataFrame(out, index=bull.index, columns=bull.columns)


def overlay(base: pd.DataFrame, alt: pd.DataFrame,
            mask: pd.Series) -> pd.DataFrame:
    """Use `alt` scores on days where mask is True, else `base`."""
    out = np.where(mask.to_numpy()[:, None], alt.to_numpy(float),
                   base.to_numpy(float))
    return pd.DataFrame(out, index=base.index, columns=base.columns)


# ---------------------------------------------------------------- sell rules

def sell_panic():
    """Sell-all when HY OAS > trailing 95th pct (3y window) AND SPY<200dma;
    stay liquidated (True) until a recovery trigger or base risk-on. The
    paired buy-side must hold cash during the same panic state."""
    sh = shared()
    R = sh["R"]
    oas_pct = R["hy_oas"].rolling(756, min_periods=252).rank(pct=True)
    cond = ((oas_pct > 0.95) & (R["spy_above_200"] < 1)).to_numpy()
    rec = recovery_triggers().any(axis=1).to_numpy()
    base_on = (~bear_mask_base()).to_numpy()
    panic = np.empty(len(cond), bool)
    state = False
    for i in range(len(cond)):
        if base_on[i] or rec[i]:
            state = False
        if cond[i]:
            state = True
        panic[i] = state
    return pd.Series(panic, index=R.index)


def sell_stop300(P):
    """Per-stock trailing stop: close >15% below own 300d MA."""
    c = P["close"]
    ma300 = c.rolling(300, min_periods=250).mean()
    return c < 0.85 * ma300


# ---------------------------------------------------------------- experiments

def main():
    sh = shared()
    P = sh["P"]
    idx = P["close"].index
    cards = {}

    def ev(scores, name, sell=None):
        cards[name] = protocol.evaluate_signal(scores, name, k=3, sell=sell)
        return cards[name]

    bull = mom126(P)

    # ---- 0. reference: bull sleeve everywhere (== naive baseline)
    ev(bull, "bear_ref_mom126")

    # ---- 1. pure bear sleeves run full-time (information only)
    sleeves = {
        "lowvol_rs": bear_lowvol_rs(P),
        "defq": bear_defquality(P),
        "rebound": bear_rebound(P),
        "rebound_fb": bear_rebound_fallback(P),
        "mom_lowbeta": bear_mom_lowbeta(P),
    }
    for nm, s in sleeves.items():
        ev(s, f"bear_pure_{nm}")

    # ---- 2. regime switch: bull sleeve in risk-on, bear sleeve in risk-off
    on_base = ~bear_mask_base()
    print(f"\nbear days (base mask): {(~on_base).mean():.1%} of sample")
    ev(switch(bull, None, on_base), "bear_sw_cash")      # trend-gate control
    for nm, s in sleeves.items():
        ev(switch(bull, s, on_base), f"bear_sw_{nm}")

    # ---- 3. recovery triggers: switch back to bull sleeve early
    best_bear = sleeves["rebound_fb"]   # re-pointed after stage-2 inspection
    combos = {
        "thrust": ["thrust"],
        "vixrlx": ["vix_rlx"],
        "spy50": ["spy50"],
        "all": ["thrust", "vix_rlx", "spy50"],
    }
    for nm, trigs in combos.items():
        on_r = risk_on_with_recovery(trigs)
        extra = (on_r & ~on_base).mean()
        print(f"  recovery '{nm}': +{extra:.1%} of days flipped early to bull")
        for snm in ("rebound_fb", "lowvol_rs", "defq"):
            ev(switch(bull, sleeves[snm], on_r), f"bear_sw_{snm}_rec_{nm}")

    # ---- 4. sell rules (on the no-recovery switch and on plain bull)
    panic = sell_panic()
    print(f"\npanic-liquidation days: {panic.mean():.1%} of sample")
    sell_all = pd.DataFrame(np.broadcast_to(panic.to_numpy()[:, None],
                                            P["close"].shape).copy(),
                            index=idx, columns=P["close"].columns)
    # buy-side must also sit in cash during panic
    for snm in ("rebound_fb", "lowvol_rs"):
        sc = switch(bull, sleeves[snm], on_base)
        sc_panic = sc.mask(panic, np.nan)
        ev(sc_panic, f"bear_sw_{snm}_sellpanic", sell=sell_all)
    ev(bull.mask(panic, np.nan), "bear_mom126_sellpanic", sell=sell_all)

    stop = sell_stop300(P)
    print(f"stop300 firing share (member-days): "
          f"{(stop & P['member']).to_numpy().sum() / P['member'].to_numpy().sum():.1%}")
    ev(bull, "bear_mom126_stop300", sell=stop)
    ev(switch(bull, sleeves["rebound_fb"], on_base),
       "bear_sw_rebound_fb_stop300", sell=stop)

    # ---- 5. VIX-scaled aggressiveness
    R = sh["R"]
    hot = R["vix_pct3y"] > 0.90
    print(f"\nhigh-VIX (pct3y>0.9) days: {hot.mean():.1%} of sample")
    aggro = bear_vix_aggro(P)
    for snm in ("rebound_fb", "lowvol_rs"):
        base_sc = switch(bull, sleeves[snm], on_base)
        ev(overlay(base_sc, aggro, hot & ~on_base), f"bear_sw_{snm}_vixaggro")
    # aggro overlay on plain momentum (no bear sleeve at all)
    ev(overlay(bull, aggro, hot), "bear_mom126_vixaggro")

    # ---- 6. best switched + best recovery + vix aggro stacked
    on_best = risk_on_with_recovery(["thrust", "vix_rlx", "spy50"])
    stacked = overlay(switch(bull, sleeves["rebound_fb"], on_best),
                      aggro, hot & ~on_best)
    ev(stacked, "bear_stack_rebound_rec_all_vixaggro")

    # ---- summary table
    print("\n=== SUMMARY ===")
    cols = ("win_qqq", "win_spy", "med_vs_qqq", "worst_vs_qqq", "full_mult")
    hdr = f"{'name':42s} " + " ".join(f"{c:>11s}" for c in cols) + \
        "   GFC    recov  bear22 vol18"
    print(hdr)
    for nm, c in cards.items():
        rg = c["regimes"]
        line = f"{nm:42s} " + " ".join(
            f"{c[k]:>11.3f}" if c[k] is not None else f"{'na':>11s}"
            for k in cols)
        for r in ("GFC_2007_2009", "recovery_2009_2012", "bear_2022",
                  "vol_2018"):
            v = rg.get(r, {}).get("vs_qqq")
            line += f" {v:+.2f}" if v is not None else "   na"
        print(line)


if __name__ == "__main__":
    main()
