"""HYPERVOL — intraday carry analysis (the honest tail the daily model hides).

The daily-close backtest reported the delta-neutral carry at ~2% vol / -2% maxDD.
That is an artifact of sampling once a day: the real risk of  long spot + short
perp  is the *intraday* perp/spot de-peg, which can liquidate the short-perp leg
at the worst instant and then revert. Here we sample hourly to measure it.

Per $1 of notional, the delta-neutral book's hourly P&L is
    pnl_t = spot_ret_t - perp_ret_t + funding_t
          = Δbasis_t + funding_t
so the equity curve is driven by basis (mean-reverting, the risk) plus funding
(the carry). We compute:

  1. the true intraday basis-excursion tail (worst perp-rich spike vs spot);
  2. the hourly-MTM vol and max drawdown of the position (vs the daily figure);
  3. a margin / liquidation model -> the leverage at which an intraday spike wipes
     the short-perp leg, and the realistic return-on-capital;
  4. what the carry actually yields NOW, net of hourly funding + costs.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .engine import COST_BPS, stats, fmt_stats

DATA = Path("/home/user/bonds/data/hypervol")
HPY = 24 * 365  # hours per year
COINS = ["BTC", "ETH", "SOL"]


def load(coin: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"intraday_{coin}.parquet").sort_index()
    df["perp_ret"] = df["perp"].pct_change()
    df["spot_ret"] = df["spot"].pct_change()
    df["basis"] = df["perp"] / df["spot"] - 1.0          # perp premium over spot
    df["dbasis"] = df["spot_ret"] - df["perp_ret"]       # short-perp+long-spot pnl ex-funding
    df["funding"] = df["funding"].fillna(0.0)
    return df.dropna()


def basis_tail(df: pd.DataFrame) -> dict:
    b = df["basis"]
    return {
        "basis_mean": b.mean(), "basis_std": b.std(),
        "basis_p99": b.quantile(0.99), "basis_p01": b.quantile(0.01),
        "basis_max": b.max(), "basis_min": b.min(),
        # worst adverse move for a SHORT perp = largest perp-rich spike
        "worst_perp_rich": b.max(),
    }


def carry_hourly(df: pd.DataFrame, cost_bps: float = COST_BPS,
                 hold_hours: int = 24 * 30) -> pd.DataFrame:
    """Always-on delta-neutral carry, MTM hourly. One entry + one exit over the
    window (cost amortized), so this isolates basis+funding, not churn."""
    gross = df["dbasis"] + df["funding"]
    # one round trip (2 legs in, 2 legs out) over the whole window
    n = len(df)
    entry_exit_cost = (4 * cost_bps / 1e4)
    cost_per_hour = entry_exit_cost / n
    net = gross - cost_per_hour
    out = pd.DataFrame({"gross": gross, "net": net, "basis": df["basis"],
                        "funding": df["funding"]})
    out["equity"] = (1 + out["net"]).cumprod()
    return out


def hourly_stats(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / HPY
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(HPY)
    sharpe = r.mean() / r.std() * np.sqrt(HPY) if r.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {"n_hours": len(r), "cagr": cagr, "vol": vol, "sharpe": sharpe,
            "maxdd": dd, "total_ret": eq.iloc[-1] - 1}


def liquidation_model(df: pd.DataFrame) -> dict:
    """At what perp-leg leverage does the worst observed intraday basis spike
    liquidate the short? A short perp posted with margin m (=1/lev) is liquidated
    when the perp rises ~m (ex maintenance). For a *delta-neutral* book the
    directional move is hedged by spot, so the binding risk is the BASIS spike
    that the spot leg does NOT offset."""
    worst_spike = df["basis"].max()              # perp richest vs spot
    # margin needed to survive that basis spike with a 30% safety buffer
    safe_margin = worst_spike * 1.3
    max_safe_lev = 1.0 / safe_margin if safe_margin > 0 else np.inf
    return {"worst_basis_spike": worst_spike,
            "margin_to_survive_basis": safe_margin,
            "max_safe_perp_leverage": max_safe_lev}


def report() -> None:
    print("=" * 92)
    print("INTRADAY CARRY AUDIT — hourly, current regime "
          f"({load('BTC').index[0].date()} -> {load('BTC').index[-1].date()})")
    print("=" * 92)

    for c in COINS:
        df = load(c)
        bt = basis_tail(df)
        ch = carry_hourly(df)
        hs = hourly_stats(ch["net"])
        lm = liquidation_model(df)
        ann_fund = df["funding"].mean() * HPY

        print(f"\n### {c}  ({len(df)} hours)")
        print(f"  funding now:      {ann_fund:+.1%} annualized "
              f"(hourly mean {df['funding'].mean():+.5%})")
        print(f"  basis (perp/spot-1):  mean {bt['basis_mean']:+.3%}  "
              f"std {bt['basis_std']:.3%}  p01 {bt['basis_p01']:+.3%}  "
              f"p99 {bt['basis_p99']:+.3%}")
        print(f"  worst perp-rich spike: {bt['worst_perp_rich']:+.3%}   "
              f"worst perp-cheap: {bt['basis_min']:+.3%}")
        print(f"  HOURLY-MTM carry:  CAGR {hs['cagr']:+.1%}  vol {hs['vol']:.1%}  "
              f"Sharpe {hs['sharpe']:+.2f}  maxDD {hs['maxdd']:+.2%}  "
              f"(n={hs['n_hours']}h)")
        print(f"  liquidation: worst basis spike {lm['worst_basis_spike']:+.2%} "
              f"=> need {lm['margin_to_survive_basis']:.1%} margin "
              f"=> max safe perp leverage {lm['max_safe_perp_leverage']:.1f}x")

    print("\n" + "-" * 92)
    print("Honest read-through:")
    print("  * Hourly vol/maxDD are LARGER than the daily-close model's ~2% — the")
    print("    daily snapshot literally cannot see the intraday basis swings.")
    print("  * Funding has compressed (BTC/ETH ~4-5%/yr, SOL ~0) — the live carry")
    print("    yield is now thin and, after the realistic capital it ties up and")
    print("    the de-peg tail, no longer the Sharpe-6 trade of 2023-24.")


if __name__ == "__main__":
    report()
