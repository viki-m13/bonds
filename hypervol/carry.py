"""HYPERVOL — delta-neutral basis-carry sleeve (the faithful 'sell-the-premium'
analog of SVXY) and its regime-gated, Strategy-4 form.

Economic idea
-------------
On an options venue you sell the volatility risk premium by being short VXX /
long SVXY. On a perp venue you cannot sell options, but the *same* premium shows
up as the funding rate: perps trade at a premium to spot ~90% of the time, so a
delta-neutral book of  LONG spot + SHORT perp  simply collects funding. Crucially,
perps never expire, so unlike VIX futures there is NO roll cost — the only costs
are entering, exiting, and flipping.

Strategy-4 mapping for the carry
--------------------------------
  u = +1  : short perp / long spot  -> collect POSITIVE funding (SVXY analog)
  u = -1  : long perp / short spot  -> collect NEGATIVE funding (VXX  analog)
  u =  0  : flat (ambiguous regime)

Per-unit daily carry return:
  s_t = (spot_ret_t - perp_ret_t) + funding_day_t        # for u=+1
  carry_ret_t = u_{t-1} * s_t  - cost(|u_t - u_{t-1}|)

Costs: each leg pays HL taker+slippage on the traded notional; flipping u from
+1 to -1 reverses BOTH legs (turnover 2 per leg).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .engine import (DATA, COST_BPS, DPY, add_signals, load_coin, stats,
                     fmt_stats)

SPOT = Path("/home/user/bonds/data/crypto")


def load_carry_frame(coin: str) -> pd.DataFrame:
    """Aligned perp/spot/funding/DVOL frame for the carry sleeve."""
    perp = pd.read_parquet(DATA / f"perp_{coin}.parquet")
    fund = pd.read_parquet(DATA / f"funding_{coin}.parquet")
    sp = pd.read_csv(SPOT / f"{coin}_USD.csv", parse_dates=["Date"]).set_index("Date")
    sp.index = pd.to_datetime(sp.index).normalize()
    df = pd.DataFrame({
        "perp": perp["close"], "spot": sp["Close"].astype(float),
        "funding_day": fund["funding_day"], "funding_ann": fund["funding_ann"],
    })
    if coin in ("BTC", "ETH"):
        dv = pd.read_parquet(DATA / f"dvol_{coin}.parquet")
        df = df.join(dv, how="left")
    else:
        df["iv"] = np.nan
    df = df.dropna(subset=["perp", "spot", "funding_day"])
    df["perp_ret"] = df["perp"].pct_change()
    df["spot_ret"] = df["spot"].pct_change()
    # Guard the ~5% of days with a spot gap: if either leg jumps >40% in a day vs
    # the other (data gap, not real basis), null the basis that day.
    df["basis"] = df["spot_ret"] - df["perp_ret"]
    df.loc[df["basis"].abs() > 0.05, "basis"] = 0.0   # >5% daily basis == data gap
    # realized-vol / eVRP signals (reuse engine)
    df["ret"] = df["perp_ret"]
    df = add_signals(df, rv_window=10, fund_window=7)
    return df


def _signed_weight(df: pd.DataFrame, gated: bool) -> pd.Series:
    """u_t target. always-on -> +1 (with sign of funding so we always *receive*).
    gated -> Strategy-4 four-regime switch."""
    if not gated:
        # Always positioned to receive funding: short perp when funding>0, long
        # perp when funding<0. (Pure carry, no eVRP gate.)
        return np.sign(df["fund_smooth"]).fillna(0.0)

    u = pd.Series(0.0, index=df.index)
    evrp = df["evrp"]; cont = df["contango"]; backw = df["backwardation"]
    have = evrp.notna()
    u[(have) & (evrp > 0) & cont] = 1.0     # regime 1: full short-perp carry
    u[(have) & (evrp <= 0) & cont] = 0.5    # regime 2: half
    u[(have) & (evrp <= 0) & backw] = -1.0  # regime 3: flip to long-perp carry
    # regime 4 (evrp>0 & backw) stays 0
    # If DVOL missing (alts), fall back to sign(funding) always-on.
    u[~have] = np.sign(df["fund_smooth"]).fillna(0.0)[~have]
    return u


def backtest_carry(df: pd.DataFrame, gated: bool, cost_bps: float = COST_BPS,
                   band: float = 0.25) -> pd.DataFrame:
    """Delta-neutral carry. band = min |Δu| to retrade (limits churn)."""
    u_tgt = _signed_weight(df, gated)
    # rebalance band
    u = np.zeros(len(u_tgt)); prev = 0.0; arr = u_tgt.to_numpy()
    for i in range(len(arr)):
        if abs(arr[i] - prev) > band:
            prev = arr[i]
        u[i] = prev
    u = pd.Series(u, index=df.index)

    s = (df["basis"] + df["funding_day"]).fillna(0.0)   # per-unit short-perp carry
    u_eff = u.shift(1).fillna(0.0)
    # two legs reverse on a flip -> 2x turnover per unit change
    turn = u.diff().abs().fillna(u.abs())
    cost = -(2 * cost_bps / 1e4) * turn.shift(1).fillna(0.0)

    out = pd.DataFrame({
        "u": u, "u_eff": u_eff, "funding_ann": df["funding_ann"],
        "evrp": df["evrp"], "carry_gross": u_eff * s, "cost": cost,
        "turn": turn,
    })
    out["strat_ret"] = out["carry_gross"] + out["cost"]
    out["equity"] = (1 + out["strat_ret"]).cumprod()
    return out
