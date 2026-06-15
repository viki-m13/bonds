"""HYPERVOL — signal + backtest engine.

Ports Concretum's VIX-ETN "Strategy 4" (eVRP + term-structure + vol sizing) onto
Hyperliquid perps. See README.md for the full economic mapping. In one line:

    SVXY (short-vol, +risk beta)  -> LONG the perp
    VXX  (long-vol,  -risk beta)  -> SHORT the perp
    VIX  / VIX3M term structure   -> FUNDING rate (perp roll carry)
    VIX  (30d implied vol)        -> DVOL (Deribit 30d implied vol)
    RV(SPY)                       -> RV(coin)
    eVRP = VIX - RV               -> DVOL - RV

The backtest is deliberately pessimistic and self-consistent:
  * decisions use only information up to the close of day t;
  * the chosen weight earns day t+1 price return and PAYS/RECEIVES day t+1 funding
    on its full notional (longs pay positive funding, shorts receive it);
  * every change in weight pays Hyperliquid taker fee + slippage on the traded
    notional.
No lookahead: weight[t] multiplies return[t+1] and funding[t+1].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/home/user/bonds/data/hypervol")
DPY = 365  # crypto trades every day

# ---- Honest Hyperliquid cost assumptions (base fee tier) ------------------- #
HL_TAKER_BPS = 4.5     # 0.045% taker fee (market-on-close style fills)
SLIP_BPS = 3.0         # slippage for BTC/ETH/SOL size; stressed higher in validate
COST_BPS = HL_TAKER_BPS + SLIP_BPS   # per-side, applied to traded notional


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #
def load_coin(coin: str, with_dvol: bool = True) -> pd.DataFrame:
    """Aligned daily frame: perp close/return, daily+annual funding, DVOL (iv)."""
    perp = pd.read_parquet(DATA / f"perp_{coin}.parquet")
    fund = pd.read_parquet(DATA / f"funding_{coin}.parquet")
    df = perp[["close"]].join(fund[["funding_day", "funding_ann", "n_hours"]],
                              how="left")
    if with_dvol:
        cur = coin if coin in ("BTC", "ETH") else None
        if cur is not None:
            dv = pd.read_parquet(DATA / f"dvol_{cur}.parquet")
            df = df.join(dv, how="left")
        else:
            df["iv"] = np.nan
    df["ret"] = df["close"].pct_change()
    # Drop the very first partial funding day if hours < 20 (HL day-0 launch).
    df = df[df["n_hours"].fillna(24) >= 1]
    return df


# --------------------------------------------------------------------------- #
# Signals                                                                      #
# --------------------------------------------------------------------------- #
def add_signals(df: pd.DataFrame, rv_window: int = 10,
                fund_window: int = 7) -> pd.DataFrame:
    """Realized vol, eVRP, and the funding-based term-structure state.

    rv_window   lookback (days) for realized vol — 10 mirrors the paper.
    fund_window trailing-mean window to denoise the hourly funding into a stable
                'contango vs backwardation' state.
    """
    df = df.copy()
    # Realized vol, annualized, in vol *points* (to match DVOL units).
    df["rv"] = df["ret"].rolling(rv_window).std(ddof=1) * np.sqrt(DPY) * 100
    # Implied vol = DVOL (already annualized vol points). eVRP = IV - RV.
    df["evrp"] = df["iv"] - df["rv"]
    # Term structure analog: smoothed funding. >0 == perp premium == "contango".
    df["fund_smooth"] = df["funding_ann"].rolling(fund_window).mean()
    df["contango"] = df["fund_smooth"] > 0
    df["backwardation"] = df["fund_smooth"] < 0
    return df


def regime_weight(row, mode: str, long_only: bool,
                  target_vol: float, max_w: float,
                  size_mode: str) -> float:
    """Strategy-4 four-regime map -> signed target weight in the PERP.

    mode 'directional': SVXY->long coin, VXX->short coin (faithful beta map).
    mode 'carry'      : harvest funding directly (short rich funding / long cheap),
                        gated by the same eVRP/term-structure regime.
    size_mode 'voltarget': weight = target_vol / RV  (prudent, repo house style)
    size_mode 'ivprop'   : weight = iv/100           (faithful to the paper)
    """
    rv = row["rv"]; evrp = row["evrp"]; iv = row["iv"]
    contango = row["contango"]; backw = row["backwardation"]
    if not np.isfinite(rv) or rv <= 0:
        return 0.0

    # Base position size (always positive magnitude).
    if size_mode == "ivprop":
        if not np.isfinite(iv):
            return 0.0
        base = iv / 100.0
    else:  # voltarget
        base = target_vol / (rv / 100.0)
    base = float(np.clip(base, 0.0, max_w))

    have_evrp = np.isfinite(evrp)

    if mode == "directional":
        # Regime 1: rich vol + contango  -> risk-on  -> LONG, full
        if have_evrp and evrp > 0 and contango:
            return +base
        # Regime 2: cheap vol + contango  -> LONG, half
        if have_evrp and evrp <= 0 and contango:
            return +0.5 * base
        # Regime 3: cheap vol + backwardation -> risk-off -> SHORT, full
        if have_evrp and evrp <= 0 and backw:
            return 0.0 if long_only else -base
        # Regime 4: rich vol + backwardation -> ambiguous -> flat
        return 0.0

    if mode == "carry":
        # Harvest funding: short when crowd-long pays you (contango), long when
        # backwardation pays you. eVRP gates: only fade the crowd when implied
        # vol is rich (premium positive) so realized risk is 'paid for'.
        if contango and (not have_evrp or evrp > 0):
            return 0.0 if long_only else -base      # short to collect + funding
        if backw and (not have_evrp or evrp <= 0):
            return +base                            # long to collect - funding
        return 0.0

    raise ValueError(mode)


# --------------------------------------------------------------------------- #
# Backtest                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    mode: str = "directional"        # 'directional' | 'carry'
    long_only: bool = False
    size_mode: str = "voltarget"     # 'voltarget' | 'ivprop'
    target_vol: float = 0.30         # annualized, for voltarget sizing
    max_w: float = 3.0               # leverage cap (HL allows far more)
    rv_window: int = 10
    fund_window: int = 7
    cost_bps: float = COST_BPS
    rebalance_band: float = 0.02     # only retrade if |Δweight| > band (paper: 2%)


def backtest(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Run one coin. Returns a per-day frame with net strategy return + diag.

    Timing (no lookahead): w[t] decided at close t from signals[<=t]; it earns
    ret[t+1] and pays funding_day[t+1]; turnover billed when w changes.
    """
    df = add_signals(df, cfg.rv_window, cfg.fund_window)

    raw_w = df.apply(lambda r: regime_weight(
        r, cfg.mode, cfg.long_only, cfg.target_vol, cfg.max_w, cfg.size_mode),
        axis=1).fillna(0.0)

    # Apply the rebalance band: hold prior weight unless the target moved enough.
    w = np.zeros(len(raw_w))
    prev = 0.0
    rw = raw_w.to_numpy()
    for i in range(len(rw)):
        tgt = rw[i]
        if abs(tgt - prev) > cfg.rebalance_band:
            prev = tgt
        w[i] = prev
    w = pd.Series(w, index=df.index)

    ret = df["ret"].fillna(0.0)
    fund = df["funding_day"].fillna(0.0)
    cost_rate = cfg.cost_bps / 1e4

    # Position decided yesterday earns today's return & pays today's funding.
    w_eff = w.shift(1).fillna(0.0)
    turnover = w.diff().abs().fillna(w.abs())          # billed on the day we trade
    price_pnl = w_eff * ret
    funding_pnl = -w_eff * fund                         # long pays +funding
    tc = -cost_rate * turnover.shift(1).fillna(0.0)     # cost realized next day

    out = pd.DataFrame({
        "close": df["close"], "ret": ret, "w": w, "w_eff": w_eff,
        "rv": df["rv"], "iv": df["iv"], "evrp": df["evrp"],
        "funding_ann": df["funding_ann"], "fund_smooth": df["fund_smooth"],
        "price_pnl": price_pnl, "funding_pnl": funding_pnl, "tc": tc,
        "turnover": turnover,
    })
    out["strat_ret"] = out["price_pnl"] + out["funding_pnl"] + out["tc"]
    out["equity"] = (1 + out["strat_ret"]).cumprod()
    return out


# --------------------------------------------------------------------------- #
# Performance stats                                                            #
# --------------------------------------------------------------------------- #
def stats(rets: pd.Series, dpy: int = DPY) -> dict:
    r = rets.dropna()
    if len(r) < 5 or r.std() == 0:
        return {"n": len(r), "cagr": np.nan, "vol": np.nan, "sharpe": np.nan,
                "maxdd": np.nan, "sortino": np.nan, "hit": np.nan,
                "total_ret": np.nan}
    eq = (1 + r).cumprod()
    yrs = len(r) / dpy
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(dpy)
    sharpe = r.mean() / r.std() * np.sqrt(dpy)
    downside = r[r < 0].std() * np.sqrt(dpy)
    sortino = r.mean() * dpy / downside if downside > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {"n": len(r), "cagr": cagr, "vol": vol, "sharpe": sharpe,
            "maxdd": dd, "sortino": sortino, "hit": (r > 0).mean(),
            "total_ret": eq.iloc[-1] - 1}


def fmt_stats(s: dict) -> str:
    return (f"n={s['n']:>4}  CAGR={s['cagr']:+7.1%}  vol={s['vol']:6.1%}  "
            f"Sharpe={s['sharpe']:+5.2f}  Sortino={s['sortino']:+5.2f}  "
            f"maxDD={s['maxdd']:+6.1%}  hit={s['hit']:.2f}  "
            f"totRet={s['total_ret']:+7.1%}")
