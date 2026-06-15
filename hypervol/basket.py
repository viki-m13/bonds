"""HYPERVOL — diversified intraday funding-carry basket (the improvement).

Per-coin the carry is now thin (Sharpe ~0.3-0.4) and funding signs are dispersed
(LINK/NEAR ~+9%/yr while ATOM/APT ~-14%/yr). Two ideas combine:

  1. Be on the RECEIVING side of each coin's funding (short perp where funding>0,
     long perp where funding<0) — a regime-adaptive rule that the 2023-24
     "always short everything" version ignored, and that matters now that funding
     goes negative on many alts.
  2. DIVERSIFY across coins. Each delta-neutral leg's residual is mostly
     idiosyncratic basis noise, so a risk-parity basket should lift Sharpe well
     above any single coin.

Risk-parity weights are inverse-basis-vol, which also auto-downweights small caps
whose Binance.US spot is stale (and whose 'basis' is partly data noise, not a
tradeable edge). Everything is hourly, net of HL taker+slippage on flips.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .engine import COST_BPS
from .intraday import load, HPY, hourly_stats

DATA = Path("/home/user/bonds/data/hypervol")


def universe() -> list[str]:
    return sorted(p.stem.replace("intraday_", "")
                  for p in DATA.glob("intraday_*.parquet"))


def coin_panel() -> dict[str, pd.DataFrame]:
    out = {}
    for c in universe():
        df = load(c)
        out[c] = df
    return out


def basket(rule: str = "sign", fund_smooth_h: int = 168,
           deadband: float = 0.05, cost_bps: float = COST_BPS,
           rp: bool = True, wcap: float = 0.30,
           max_basis_std: float = 0.0025) -> pd.DataFrame:
    """Build the basket hourly return.

    rule 'sign'   -> v_i = sign(trailing funding)  (always receive funding)
    rule 'short'  -> v_i = +1 always               (2023-24 'always short' rule)
    fund_smooth_h  trailing window (hours) to denoise funding before taking sign
    deadband       require |smoothed funding ann| > deadband to flip (anti-churn)
    rp             risk-parity (inverse basis-vol) vs equal weight
    wcap           per-coin weight cap
    max_basis_std  drop coins whose hourly basis std exceeds this (bad spot data)
    """
    panel = coin_panel()
    coins = [c for c in panel
             if panel[c]["basis"].std() <= max_basis_std]
    # common hourly index
    idx = None
    for c in coins:
        idx = panel[c].index if idx is None else idx.intersection(panel[c].index)

    rets, weights = {}, {}
    for c in coins:
        df = panel[c].reindex(idx)
        fund_ann = df["funding"] * HPY
        smooth = fund_ann.rolling(fund_smooth_h, min_periods=24).mean()
        if rule == "short":
            v = pd.Series(1.0, index=idx)
        else:
            # sign with deadband + hold last side
            raw = np.where(smooth > deadband, 1.0,
                           np.where(smooth < -deadband, -1.0, np.nan))
            v = pd.Series(raw, index=idx).ffill().fillna(0.0)
        v_eff = v.shift(1).fillna(0.0)
        unit = df["dbasis"] + df["funding"]           # short-perp carry, ex-sign
        gross = v_eff * unit
        turn = v.diff().abs().fillna(v.abs())
        cost = (2 * cost_bps / 1e4) * turn.shift(1).fillna(0.0)
        rets[c] = (gross - cost)
        weights[c] = 1.0 / df["basis"].std() if rp else 1.0

    R = pd.DataFrame(rets)
    w = pd.Series(weights)
    w = (w / w.sum()).clip(upper=wcap)
    w = w / w.sum()
    basket_ret = (R * w).sum(axis=1)
    out = pd.DataFrame({"ret": basket_ret})
    out["equity"] = (1 + out["ret"]).cumprod()
    out.attrs["coins"] = coins
    out.attrs["weights"] = w.to_dict()
    return out


def report() -> None:
    coins = universe()
    print("=" * 92)
    print(f"INTRADAY FUNDING-CARRY BASKET — {len(coins)} coins, hourly, "
          f"{load('BTC').index[0].date()} -> {load('BTC').index[-1].date()}")
    print("=" * 92)

    # liquidity filter: keep only coins whose Binance.US spot is tight enough
    # that the perp/spot basis is a real edge and not stale-quote noise.
    THR = 0.0025
    panel = coin_panel()
    kept = sorted(c for c in coins if panel[c]['basis'].std() <= THR)
    drop = sorted(c for c in coins if panel[c]['basis'].std() > THR)
    print(f"kept (hourly basis std <={THR:.2%}, liquid US spot): {kept}")
    print(f"dropped (stale/illiquid Binance.US spot):            {drop}\n")

    def show(label, bt):
        s = hourly_stats(bt["ret"])
        print(f"  {label:42s} CAGR {s['cagr']:+6.1%}  vol {s['vol']:5.1%}  "
              f"Sharpe {s['sharpe']:+5.2f}  maxDD {s['maxdd']:+6.2%}")
        return s

    print("Per-coin, kept universe (always-receive funding, hourly):")
    for c in kept:
        df = load(c)
        fund_ann = df["funding"] * HPY
        v = np.sign(fund_ann.rolling(168, min_periods=24).mean()).ffill().fillna(0)
        unit = df["dbasis"] + df["funding"]
        s = hourly_stats(v.shift(1).fillna(0) * unit)
        print(f"    {c:5s} fund {fund_ann.mean():+6.1%}  Sharpe {s['sharpe']:+5.2f}  "
              f"vol {s['vol']:5.1%}")

    print("\nBaskets:")
    show("OLD: always-short EW (2023-24 rule)",
         basket(rule="short", rp=False))
    show("always-short risk-parity",
         basket(rule="short", rp=True))
    show("NEW: sign-follow EW",
         basket(rule="sign", rp=False))
    s_best = show("NEW: sign-follow risk-parity [headline]",
                  basket(rule="sign", rp=True))

    # robustness across smoothing windows
    print("\nRobustness (sign-follow RP) across funding-smoothing window:")
    for h in [72, 120, 168, 240, 336]:
        s = hourly_stats(basket(rule="sign", rp=True, fund_smooth_h=h)["ret"])
        print(f"    smooth={h:4d}h  Sharpe {s['sharpe']:+5.2f}  CAGR {s['cagr']:+6.1%}")

    print("\nCost sensitivity (sign-follow RP):")
    for m in [1, 2, 4]:
        s = hourly_stats(basket(rule="sign", rp=True, cost_bps=COST_BPS*m)["ret"])
        print(f"    {COST_BPS*m:4.1f}bps/side  Sharpe {s['sharpe']:+5.2f}  CAGR {s['cagr']:+6.1%}")


if __name__ == "__main__":
    report()
