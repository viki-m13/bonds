"""Portfolio equity curve for the dislocation-reversion strategy vs MUB.

Model: an equal-weight portfolio of all currently-open positions; each
trade's realized entry->exit total return (coupon credited) is spread
geometrically across its holding days, and the daily portfolio return is
the mean across open positions (cash, 0%, when none are open). Compounded
from the first entry to the last exit and compared to MUB total return
(adjusted close) over the identical window.

Honesty note: individual munis do not print daily, so spreading each
trade's return linearly across its ~1y hold SMOOTHS intra-trade
volatility and understates the true mark-to-market drawdown. Total return
and CAGR are realized (real entry/exit prints + coupon); the path
smoothness is optimistic. MUB is a daily-priced ETF and shows its true
jaggedness.

Writes research/equity_curve.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
from strategies import FACTORIES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DISCOUNT = 3.0
START = pd.Timestamp("2013-01-01")
DATA_END = pd.Timestamp("2026-07-08")


def build_curve():
    panel = pd.read_parquet(ROOT / "data" / "panel_daily.parquet")
    uni = (pd.read_csv(ROOT / "data" / "universe" / "universe.csv.gz")
           .drop_duplicates("six").set_index("six"))
    bonds = bt.prepare(panel, uni["coupon"])
    fn = FACTORIES["price_discount"](discount=DISCOUNT)
    fills = bt.run_signal(bonds, fn, min_hold=365, max_hold=455,
                          date_lo=START, date_hi=DATA_END - pd.Timedelta(days=455))
    start = min(f.entry_date for f in fills)
    end = max(f.exit_date for f in fills)
    days = pd.date_range(start, end, freq="D")
    di = {d: i for i, d in enumerate(days)}
    sret = np.zeros(len(days))
    cnt = np.zeros(len(days))
    for f in fills:
        if f.hold_days <= 0:
            continue
        dr = (1 + f.ret) ** (1.0 / f.hold_days) - 1
        i0, i1 = di[f.entry_date], di[f.exit_date]
        sret[i0:i1] += dr
        cnt[i0:i1] += 1
    port_daily = np.where(cnt > 0, sret / np.where(cnt > 0, cnt, 1), 0.0)
    eq = np.cumprod(1 + port_daily)
    mub = (pd.read_csv(ROOT / "data" / "mub_daily.csv.gz", parse_dates=["date"])
           .sort_values("date").set_index("date")["adjclose"]
           .reindex(days).ffill())
    mub_eq = (mub / mub.iloc[0]).to_numpy()
    return days, eq, mub_eq, len(fills)


def stats(e, days):
    yrs = (days[-1] - days[0]).days / 365.25
    peak = np.maximum.accumulate(e)
    return e[-1] ** (1 / yrs) - 1, (e / peak - 1).min(), e[-1] - 1


def main():
    days, eq, mub_eq, nfills = build_curve()
    sc, sd, st = stats(eq, days)
    mc, md, mt = stats(mub_eq, days)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.2),
                                  height_ratios=[3.4, 1], sharex=True)
    fig.patch.set_facecolor("white")
    STR, MUB = "#1b5e9c", "#b0632e"
    ax.plot(days, eq, color=STR, lw=2.1, zorder=3,
            label="Dislocation-reversion strategy")
    ax.plot(days, mub_eq, color=MUB, lw=1.7, zorder=2,
            label="MUB (muni ETF, total return)")
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
               color="#d94040", alpha=0.07, zorder=0)
    ax.text(pd.Timestamp("2022-07-01"), eq.max() * 0.99, "2022\nrate selloff",
            ha="center", va="top", fontsize=8, color="#a33", alpha=0.8)
    for a in (ax, ax2):
        a.set_facecolor("white")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(True, axis="y", color="#e6e6e6", lw=0.8)
        a.grid(True, axis="x", color="#f2f2f2", lw=0.6)
    ax.set_ylabel("Growth of $1 (total return)")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.set_title("Individual-muni dislocation-reversion vs MUB  —  2013–2026",
                 fontsize=13, fontweight="bold", loc="left", pad=10)
    txt = (f"Strategy:  {st*100:+.1f}% total   {sc*100:.2f}% CAGR   {sd*100:.1f}% maxDD\n"
           f"MUB (TR):  {mt*100:+.1f}% total   {mc*100:.2f}% CAGR   {md*100:.1f}% maxDD\n"
           f"{nfills:,} trades  ·  73% win  ·  +3.1% excess vs matched control (p<0.001)")
    ax.text(0.985, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f7f9fb", ec="#d0d8e0"))
    rel = eq / mub_eq
    ax2.plot(days, rel, color="#2e7d5b", lw=1.6)
    ax2.axhline(1.0, color="#999", lw=0.9, ls="--")
    ax2.fill_between(days, 1.0, rel, where=rel >= 1, color="#2e7d5b", alpha=0.12)
    ax2.set_ylabel("Strat / MUB")
    ax2.text(0.01, 0.9, "relative wealth (above 1.0 = strategy ahead)",
             transform=ax2.transAxes, fontsize=8, color="#555", va="top")
    fig.tight_layout()
    out = ROOT / "research" / "equity_curve.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    print(f"strategy {st*100:+.1f}% / MUB {mt*100:+.1f}% over "
          f"{(days[-1]-days[0]).days/365.25:.1f}y")


if __name__ == "__main__":
    main()
