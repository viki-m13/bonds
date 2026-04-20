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


if __name__ == "__main__":
    main()
