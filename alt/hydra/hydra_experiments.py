"""Answer: (1) no-crypto variant, (2) ETH inclusion, (3) vol scaling confirm,
(4) rebalance-frequency sensitivity."""
from pathlib import Path
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats, PORT_VOL
from hydra_sleeves_v3 import (
    SLEEVES, s12_btc_trend, s21_eth_trend,
)


def risk_parity(df, target_vol=0.20, window=63, lev_cap=5.0):
    vols = df.rolling(window).std().shift(1) * np.sqrt(252)
    vols = vols.where(vols > 0.001)
    inv = 1 / vols
    inv = inv.where(vols.notna(), 0)
    w = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    raw = (w * df).sum(axis=1)
    pv = raw.rolling(window).std().shift(1) * np.sqrt(252)
    scale = (target_vol / pv).clip(upper=lev_cap).fillna(0)
    return raw * scale


def build(sleeve_fns, dates):
    out = {fn(dates).name: fn(dates) for fn in sleeve_fns}
    df = pd.DataFrame(out).reindex(dates).fillna(0)
    return df


def hydra_port(sleeve_fns, dates, **kw):
    df = build(sleeve_fns, dates)
    port = risk_parity(df, **kw).fillna(0)
    nz = (df != 0).any(axis=1)
    return port[nz], df


def summarise(r, label):
    s = stats(r, label)
    IS = pd.Timestamp("2018-01-01")
    si = stats(r.loc[:IS], "IS")
    so = stats(r.loc[IS:], "OOS")
    print(f"  {label:28s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}  "
          f"| IS SR={si['sharpe']:>5.2f}  OOS SR={so['sharpe']:>5.2f}")
    return s


def main():
    spy = load_etf("SPY")
    dates = spy.index
    print(f"Window: {dates[0].date()} .. {dates[-1].date()} ({len(dates)/252:.1f}y)\n")

    # Q1/Q2: crypto variants
    print("=== Q1 / Q2: Crypto variants ===")
    # Baseline = HYDRA as shipped (includes s12_btc)
    base, _ = hydra_port(SLEEVES, dates)
    summarise(base, "HYDRA (base, BTC only)")

    # No crypto
    no_btc = [fn for fn in SLEEVES if fn is not s12_btc_trend]
    r, _ = hydra_port(no_btc, dates)
    summarise(r, "HYDRA (no crypto)")

    # BTC + ETH
    with_eth = list(SLEEVES) + [s21_eth_trend]
    r, _ = hydra_port(with_eth, dates)
    summarise(r, "HYDRA (BTC + ETH)")

    # ETH only (swap s12 for s21)
    eth_only = [fn for fn in SLEEVES if fn is not s12_btc_trend] + [s21_eth_trend]
    r, _ = hydra_port(eth_only, dates)
    summarise(r, "HYDRA (ETH only)")

    # Q4: rebalance frequency — implemented by changing vol-scaling window
    # and freezing the ensemble weights to coarser cadence. The vol-scaling
    # window is already daily; the knob that matters is the sleeve-signal
    # cadence AND the ensemble weight cadence. Sleeves are already monthly
    # inside; we vary the ensemble weight cadence instead.
    print("\n=== Q4: Ensemble weight cadence (sleeve signals stay monthly) ===")
    df = build(SLEEVES, dates)

    def rp_freeze(df, freq, target_vol=0.20, window=63, lev_cap=5.0):
        """Rebuild inverse-vol weights only on `freq` boundary (D, W, M, Q)."""
        vols = df.rolling(window).std().shift(1) * np.sqrt(252)
        vols = vols.where(vols > 0.001)
        inv = 1 / vols
        inv = inv.where(vols.notna(), 0)
        w = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        # Freeze weights at freq boundaries
        if freq != "D":
            mask = pd.Series(False, index=df.index)
            if freq == "W":
                idx = df.resample("W-FRI").last().index.intersection(df.index)
            elif freq == "M":
                idx = df.resample("ME").last().index.intersection(df.index)
            elif freq == "Q":
                idx = df.resample("QE").last().index.intersection(df.index)
            mask.loc[idx] = True
            w = w.where(mask, np.nan).ffill().fillna(0)
        raw = (w * df).sum(axis=1)
        pv = raw.rolling(window).std().shift(1) * np.sqrt(252)
        scale = (target_vol / pv).clip(upper=lev_cap).fillna(0)
        return raw * scale

    nz = (df != 0).any(axis=1)
    for freq, label in [("D", "Daily weights"),
                        ("W", "Weekly weights"),
                        ("M", "Monthly weights"),
                        ("Q", "Quarterly weights")]:
        r = rp_freeze(df, freq).fillna(0)[nz]
        summarise(r, f"HYDRA ({label})")


def scaling_cadence_test():
    """Q5: does daily vol-scaling matter, or can we do it weekly/monthly
    without losing much Sharpe?"""
    print("\n=== Q5: Vol-scale update cadence (weights always daily) ===")
    spy = load_etf("SPY")
    dates = spy.index
    df = build(SLEEVES, dates)

    def rp_scale_freeze(df, scale_freq, target_vol=0.20, window=63, lev_cap=5.0):
        vols = df.rolling(window).std().shift(1) * np.sqrt(252)
        vols = vols.where(vols > 0.001)
        inv = 1 / vols
        inv = inv.where(vols.notna(), 0)
        w = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        raw = (w * df).sum(axis=1)
        pv = raw.rolling(window).std().shift(1) * np.sqrt(252)
        scale = (target_vol / pv).clip(upper=lev_cap).fillna(0)
        if scale_freq != "D":
            if scale_freq == "W":
                idx = scale.resample("W-FRI").last().index.intersection(scale.index)
            elif scale_freq == "M":
                idx = scale.resample("ME").last().index.intersection(scale.index)
            mask = pd.Series(False, index=scale.index)
            mask.loc[idx] = True
            scale = scale.where(mask, np.nan).ffill().fillna(0)
        return raw * scale

    nz = (df != 0).any(axis=1)
    for freq, label in [("D", "Daily scaling"), ("W", "Weekly scaling"), ("M", "Monthly scaling")]:
        r = rp_scale_freeze(df, freq).fillna(0)[nz]
        summarise(r, f"HYDRA ({label})")


def static_leverage_test():
    """Q7 (user ask): strip portfolio vol scaling entirely.  Use equal-weight
    or inverse-vol-at-inception, apply a STATIC leverage to hit ~20% ann
    vol in-sample, and rebalance every N days.  Compare to shipped."""
    print("\n=== Q7: No dynamic vol scaling — static leverage + fixed cadence ===")
    spy = load_etf("SPY")
    dates = spy.index
    df = build(SLEEVES, dates)
    nz = (df != 0).any(axis=1)

    def eq_live(df):
        live = (df != 0).cummax().astype(float)
        return live.div(live.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    def freeze_n(w, n):
        # rebalance every n business days, ffill in between
        mask = pd.Series(False, index=w.index)
        mask.iloc[::n] = True
        return w.where(mask, np.nan).ffill().fillna(0)

    # Pick a static leverage that hits 20% ann vol on the shipped inverse-vol daily
    # variant of equal-weight.
    for label, wfn in [("equal-weight", eq_live)]:
        for cadence, cadence_name in [(1, "daily"), (5, "5-day"), (21, "monthly")]:
            w = freeze_n(wfn(df), cadence)
            raw = (w * df).sum(axis=1)[nz].dropna()
            native_vol = raw.std() * np.sqrt(252)
            # Scale once to hit 20% ann vol (no daily vol targeting)
            lev = 0.20 / native_vol
            scaled = raw * lev
            s = summarise(scaled, f"{label}, {cadence_name} rebal (lev={lev:.2f}x)")

    print("\n--- Same but hit 10% ann vol (like sleeve base) ---")
    for cadence, cadence_name in [(1, "daily"), (5, "5-day"), (21, "monthly")]:
        w = freeze_n(eq_live(df), cadence)
        raw = (w * df).sum(axis=1)[nz].dropna()
        native_vol = raw.std() * np.sqrt(252)
        lev = 0.10 / native_vol
        scaled = raw * lev
        summarise(scaled, f"eq-wt, {cadence_name} rebal, 10% target (lev={lev:.2f}x)")

    print("\n--- Unlevered (accept native vol, no scaling at all) ---")
    for cadence, cadence_name in [(1, "daily"), (5, "5-day"), (21, "monthly")]:
        w = freeze_n(eq_live(df), cadence)
        r = (w * df).sum(axis=1)[nz].dropna()
        summarise(r, f"eq-wt, {cadence_name} rebal, UNLEVERED")


def simplicity_test():
    """Q6: strip out the portfolio-level vol target entirely.  Try:
      - inverse-vol weights, daily drift, no book-level vol target
      - inverse-vol weights, monthly rebal, no book-level vol target
      - equal-weight across live sleeves, monthly rebal
      - equal-weight, daily drift
    Sleeve-level internal vol scaling stays (so sleeves are still ~10% vol).
    """
    print("\n=== Q6: Strip portfolio vol target / try equal-weight ===")
    spy = load_etf("SPY")
    dates = spy.index
    df = build(SLEEVES, dates)
    nz = (df != 0).any(axis=1)

    def invol_weights(df, window=63):
        vols = df.rolling(window).std().shift(1) * np.sqrt(252)
        vols = vols.where(vols > 0.001)
        inv = 1 / vols
        inv = inv.where(vols.notna(), 0)
        return inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    def eq_weights(df):
        live = (df != 0).astype(float)
        # A sleeve is "live" from its first non-zero day
        first_live = (df != 0).cummax()
        live = first_live.astype(float)
        return live.div(live.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    def freeze_monthly(w):
        idx = w.resample("ME").last().index.intersection(w.index)
        mask = pd.Series(False, index=w.index)
        mask.loc[idx] = True
        return w.where(mask, np.nan).ffill().fillna(0)

    # A. Inverse-vol daily, no book vol target
    w = invol_weights(df)
    r = (w * df).sum(axis=1)[nz]
    summarise(r, "Inv-vol daily, no book VT")

    # B. Inverse-vol monthly, no book vol target
    w = freeze_monthly(invol_weights(df))
    r = (w * df).sum(axis=1)[nz]
    summarise(r, "Inv-vol monthly, no book VT")

    # C. Equal-weight daily, no book vol target
    w = eq_weights(df)
    r = (w * df).sum(axis=1)[nz]
    summarise(r, "Eq-wt daily, no book VT")

    # D. Equal-weight monthly, no book vol target
    w = freeze_monthly(eq_weights(df))
    r = (w * df).sum(axis=1)[nz]
    summarise(r, "Eq-wt monthly, no book VT")

    # E. Baseline: everything on (shipped)
    port = risk_parity(df)[nz]
    summarise(port, "HYDRA shipped (all VT)")

    # F. Inv-vol daily + book VT but monthly (weights change daily, scaler monthly)
    w = invol_weights(df)
    raw = (w * df).sum(axis=1)
    pv = raw.rolling(63).std().shift(1) * np.sqrt(252)
    scale = (0.20 / pv).clip(upper=5.0).fillna(0)
    idx = scale.resample("ME").last().index.intersection(scale.index)
    mask = pd.Series(False, index=scale.index); mask.loc[idx] = True
    scale_m = scale.where(mask, np.nan).ffill().fillna(0)
    r = (raw * scale_m)[nz]
    summarise(r, "Inv-vol daily + monthly VT")


if __name__ == "__main__":
    main()
    scaling_cadence_test()
    simplicity_test()
    static_leverage_test()
