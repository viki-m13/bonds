"""Event-driven backtest for trading individual munis like stocks.

Honesty rules, fixed before any results were produced:

  1. Fills only at prices real customers actually got that day:
     buys fill at the day's par-weighted customer-buy price (S prints,
     retail size band), sells at the day's customer-sell price (P prints).
     If the required side didn't print that day, you don't trade that day.
  2. Signals on day t use data through day t only; entry is next day
     (t+1) at the earliest, so nothing intraday is assumed (EMMA chart
     timestamps are date-granular).
  3. A bond is eligible on day t only if it printed on >= MIN_ACTIVE
     distinct days in the trailing 90 calendar days — a liquidity gate
     that uses no future information.
  4. Exits: after the minimum holding period, sell at the first day with
     a P print; hard stop at MAX_HOLD calendar days using the last
     available P price (marked stale so we can count those).
  5. Coupon accrual is credited linearly (coupon/365 per day) — muni
     prices are clean prices.
  6. Every strategy is compared against a matched random control: same
     bond, same entry-eligibility screen, random entry dates, same
     exit logic and cost model. This nets out carry, spread costs and
     panel composition, isolating timing skill.

Survivorship note: the downloadable universe is "securities that traded
between 2025-07 and 2026-07", so earlier years over-represent bonds that
survived/stayed liquid. IS results carry that tilt; the final OOS year
(2025-07..2026-07) is survivorship-free by construction. See README.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MIN_ACTIVE_DAYS_90 = 8      # trailing-liquidity gate
MAX_HOLD = 120              # calendar days, hard exit
IS_END = pd.Timestamp("2022-12-31")   # in-sample boundary
OOS_END = pd.Timestamp("2026-07-07")


@dataclass
class Fill:
    six: str
    entry_date: pd.Timestamp
    entry_px: float
    exit_date: pd.Timestamp
    exit_px: float
    coupon: float
    stale_exit: bool
    hold_days: int = field(init=False)
    ret: float = field(init=False)

    def __post_init__(self):
        self.hold_days = int((self.exit_date - self.entry_date).days)
        accrual = self.coupon / 100.0 / 365.0 * self.hold_days * 100.0
        self.ret = (self.exit_px - self.entry_px + accrual) / self.entry_px


def prepare(panel: pd.DataFrame, coupons: pd.Series) -> dict[str, pd.DataFrame]:
    """Split the panel per bond, add trailing-liquidity eligibility."""
    out = {}
    for six, g in panel.groupby("six"):
        g = g.sort_values("date").reset_index(drop=True)
        g["date"] = pd.to_datetime(g["date"])
        # trailing 90d active-day count (excluding today)
        d = g["date"]
        counts = np.searchsorted(d.values, d.values) - np.searchsorted(
            d.values, (d - pd.Timedelta(days=90)).values)
        g["active_90d"] = counts
        g["eligible"] = g["active_90d"] >= MIN_ACTIVE_DAYS_90
        g["coupon"] = coupons.get(six, np.nan)
        out[six] = g
    return out


def run_signal(bonds: dict[str, pd.DataFrame],
               signal_fn,
               min_hold: int = 10,
               date_lo: pd.Timestamp | None = None,
               date_hi: pd.Timestamp | None = None,
               per_bond_cooldown: int = 30,
               use_gate: bool = True,
               restrict: set[str] | None = None) -> list[Fill]:
    """Generic engine: signal_fn(g) -> boolean Series aligned to g.index
    (True = enter at the NEXT day's S print).

    use_gate=False disables the trailing-liquidity eligibility filter (for
    the new-issue family, which by definition has no trailing history at
    entry). restrict, if given, limits trading to that set of securities.
    """
    fills: list[Fill] = []
    for six, g in bonds.items():
        if restrict is not None and six not in restrict:
            continue
        coupon = g["coupon"].iloc[0]
        if np.isnan(coupon):
            coupon = 4.5  # conservative default; universe median ~4.9
        sig = signal_fn(g)
        if sig is None:
            continue
        gate = g["eligible"].to_numpy() if use_gate else np.ones(len(g), bool)
        idx = np.flatnonzero(sig.to_numpy() & gate)
        last_exit: pd.Timestamp | None = None
        for i in idx:
            if i + 1 >= len(g):
                continue
            # entry: next row with an S print within 7 calendar days
            sub = g.iloc[i + 1:]
            sub = sub[(sub["date"] - g["date"].iloc[i]).dt.days <= 7]
            sub = sub[sub["s_px"].notna()]
            if sub.empty:
                continue
            e = sub.iloc[0]
            entry_date, entry_px = e["date"], float(e["s_px"])
            if date_lo is not None and entry_date < date_lo:
                continue
            if date_hi is not None and entry_date > date_hi:
                continue
            if last_exit is not None and (entry_date - last_exit).days < per_bond_cooldown:
                continue
            # exit: first P print after min_hold, hard stop MAX_HOLD
            after = g[(g["date"] >= entry_date + pd.Timedelta(days=min_hold))
                      & (g["date"] <= entry_date + pd.Timedelta(days=MAX_HOLD))]
            px = after[after["p_px"].notna()]
            if len(px):
                x = px.iloc[0]
                fills.append(Fill(six, entry_date, entry_px,
                                  x["date"], float(x["p_px"]),
                                  float(coupon), False))
                last_exit = x["date"]
            else:
                # stale exit: last known P print anywhere before the stop
                hist = g[(g["date"] <= entry_date + pd.Timedelta(days=MAX_HOLD))
                         & g["p_px"].notna()]
                if hist.empty:
                    continue
                x = hist.iloc[-1]
                exit_date = entry_date + pd.Timedelta(days=MAX_HOLD)
                fills.append(Fill(six, entry_date, entry_px,
                                  exit_date, float(x["p_px"]),
                                  float(coupon), True))
                last_exit = exit_date
    return fills


def matched_random_control(bonds: dict[str, pd.DataFrame],
                           fills: list[Fill],
                           n_draws: int = 20,
                           min_hold: int = 10,
                           seed: int = 11,
                           use_gate: bool = True) -> pd.DataFrame:
    """For each real fill, draw random entry days in the same bond within the
    same evaluation window, run the identical exit logic. Candidate days
    respect the same liquidity gate as the strategy (use_gate)."""
    rng = np.random.default_rng(seed)
    if not fills:
        return pd.DataFrame()
    lo = min(f.entry_date for f in fills)
    hi = max(f.entry_date for f in fills)
    rows = []
    for f in fills:
        g = bonds[f.six]
        gate = g["eligible"] if use_gate else True
        cand = g[gate & g["s_px"].notna()
                 & (g["date"] >= lo) & (g["date"] <= hi)]
        if cand.empty:
            continue
        take = cand.sample(n=min(n_draws, len(cand)), random_state=int(
            rng.integers(0, 2**31)))
        for _, e in take.iterrows():
            entry_date, entry_px = e["date"], float(e["s_px"])
            after = g[(g["date"] >= entry_date + pd.Timedelta(days=min_hold))
                      & (g["date"] <= entry_date + pd.Timedelta(days=MAX_HOLD))]
            px = after[after["p_px"].notna()]
            coupon = f.coupon
            if len(px):
                x = px.iloc[0]
                ctl = Fill(f.six, entry_date, entry_px, x["date"],
                           float(x["p_px"]), coupon, False)
            else:
                hist = g[(g["date"] <= entry_date + pd.Timedelta(days=MAX_HOLD))
                         & g["p_px"].notna()]
                if hist.empty:
                    continue
                x = hist.iloc[-1]
                ctl = Fill(f.six, entry_date, entry_px,
                           entry_date + pd.Timedelta(days=MAX_HOLD),
                           float(x["p_px"]), coupon, True)
            rows.append({"six": f.six, "ret": ctl.ret,
                         "hold": ctl.hold_days, "match": id(f)})
    return pd.DataFrame(rows)


def summarize(fills: list[Fill], label: str,
              control: pd.DataFrame | None = None) -> dict:
    if not fills:
        return {"label": label, "n": 0}
    df = pd.DataFrame([{
        "six": f.six, "entry": f.entry_date, "exit": f.exit_date,
        "ret": f.ret, "hold": f.hold_days, "stale": f.stale_exit,
    } for f in fills])
    ann = df["ret"] / df["hold"].clip(lower=1) * 365
    out = {
        "label": label,
        "n": len(df),
        "n_bonds": df["six"].nunique(),
        "mean_ret": df["ret"].mean(),
        "median_ret": df["ret"].median(),
        "win_rate": (df["ret"] > 0).mean(),
        "mean_hold": df["hold"].mean(),
        "stale_share": df["stale"].mean(),
        "mean_ann": ann.mean(),
    }
    if control is not None and len(control):
        ctl_mean = control["ret"].mean()
        out["control_mean_ret"] = ctl_mean
        out["excess_vs_control"] = out["mean_ret"] - ctl_mean
        # bootstrap p-value on the excess (cluster by fill)
        rng = np.random.default_rng(3)
        strat = df["ret"].to_numpy()
        ctl_by_match = control.groupby("match")["ret"].mean().to_numpy()
        n = min(len(strat), len(ctl_by_match))
        boots = []
        for _ in range(2000):
            b1 = rng.choice(strat, size=len(strat), replace=True).mean()
            b2 = rng.choice(ctl_by_match, size=len(ctl_by_match),
                            replace=True).mean()
            boots.append(b1 - b2)
        boots = np.array(boots)
        out["excess_p_boot"] = float((boots <= 0).mean())
    return out
