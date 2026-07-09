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


class _Arr:
    """Precomputed numpy view of one bond, for fast exit lookups."""
    __slots__ = ("day", "s_px", "p_px", "elig", "p_day", "p_px_at",
                 "s_day", "s_px_at", "coupon", "n")

    def __init__(self, g: pd.DataFrame):
        self.day = g["date"].values.astype("datetime64[D]").astype(np.int64)
        self.s_px = g["s_px"].to_numpy(float) if "s_px" in g \
            else np.full(len(g), np.nan)
        self.p_px = g["p_px"].to_numpy(float) if "p_px" in g \
            else np.full(len(g), np.nan)
        self.elig = g["eligible"].to_numpy(bool)
        c = g["coupon"].iloc[0]
        self.coupon = 4.5 if pd.isna(c) else float(c)
        self.n = len(g)
        pmask = ~np.isnan(self.p_px)
        self.p_day = self.day[pmask]
        self.p_px_at = self.p_px[pmask]
        smask = ~np.isnan(self.s_px)
        self.s_day = self.day[smask]
        self.s_px_at = self.s_px[smask]


_ARR_CACHE: dict[int, _Arr] = {}


def _arr(six: str, g: pd.DataFrame) -> _Arr:
    a = _ARR_CACHE.get(id(g))
    if a is None:
        a = _Arr(g)
        _ARR_CACHE[id(g)] = a
    return a


def _exit_for(a: _Arr, entry_day: int, min_hold: int, hard: int):
    """(exit_day, exit_px, stale) for an entry, or None. Vectorized:
    first P print in [entry+min_hold, entry+hard]; else last P print
    on/before entry+hard (stale)."""
    lo = entry_day + min_hold
    hi = entry_day + hard
    j = np.searchsorted(a.p_day, lo, side="left")
    if j < len(a.p_day) and a.p_day[j] <= hi:
        return int(a.p_day[j]), float(a.p_px_at[j]), False
    k = np.searchsorted(a.p_day, hi, side="right") - 1
    if k >= 0:
        return hi, float(a.p_px_at[k]), True
    return None


def run_signal(bonds: dict[str, pd.DataFrame],
               signal_fn,
               min_hold: int = 10,
               date_lo: pd.Timestamp | None = None,
               date_hi: pd.Timestamp | None = None,
               per_bond_cooldown: int = 30,
               use_gate: bool = True,
               restrict: set[str] | None = None,
               max_hold: int | None = None) -> list[Fill]:
    """Generic engine: signal_fn(g) -> boolean Series aligned to g.index
    (True = enter at the NEXT day's S print).

    use_gate=False disables the trailing-liquidity eligibility filter (for
    the new-issue family, which by definition has no trailing history at
    entry). restrict, if given, limits trading to that set of securities.
    """
    hard = MAX_HOLD if max_hold is None else max_hold
    lo_day = None if date_lo is None else \
        np.datetime64(date_lo, "D").astype(np.int64)
    hi_day = None if date_hi is None else \
        np.datetime64(date_hi, "D").astype(np.int64)
    fills: list[Fill] = []
    for six, g in bonds.items():
        if restrict is not None and six not in restrict:
            continue
        sig = signal_fn(g)
        if sig is None:
            continue
        a = _arr(six, g)
        gate = a.elig if use_gate else np.ones(a.n, bool)
        idx = np.flatnonzero(sig.to_numpy() & gate)
        last_exit_day = -10**9
        for i in idx:
            sig_day = a.day[i]
            # entry: first S print strictly after the signal day, within 7d
            j = np.searchsorted(a.s_day, sig_day, side="right")
            if j >= len(a.s_day) or a.s_day[j] - sig_day > 7:
                continue
            entry_day = int(a.s_day[j])
            entry_px = float(a.s_px_at[j])
            if lo_day is not None and entry_day < lo_day:
                continue
            if hi_day is not None and entry_day > hi_day:
                continue
            if entry_day - last_exit_day < per_bond_cooldown:
                continue
            ex = _exit_for(a, entry_day, min_hold, hard)
            if ex is None:
                continue
            exit_day, exit_px, stale = ex
            fills.append(Fill(six,
                              pd.Timestamp(entry_day, unit="D"), entry_px,
                              pd.Timestamp(exit_day, unit="D"), exit_px,
                              a.coupon, stale))
            last_exit_day = exit_day
    return fills


def matched_random_control(bonds: dict[str, pd.DataFrame],
                           fills: list[Fill],
                           n_draws: int = 20,
                           min_hold: int = 10,
                           seed: int = 11,
                           use_gate: bool = True,
                           max_hold: int | None = None) -> pd.DataFrame:
    """For each real fill, draw random entry days in the same bond within the
    same evaluation window, run the identical exit logic. Candidate days
    respect the same liquidity gate as the strategy (use_gate)."""
    hard = MAX_HOLD if max_hold is None else max_hold
    rng = np.random.default_rng(seed)
    if not fills:
        return pd.DataFrame()
    lo_day = np.datetime64(min(f.entry_date for f in fills), "D").astype(np.int64)
    hi_day = np.datetime64(max(f.entry_date for f in fills), "D").astype(np.int64)
    rows = []
    for mi, f in enumerate(fills):
        g = bonds[f.six]
        a = _arr(f.six, g)
        gmask = a.elig if use_gate else np.ones(a.n, bool)
        cmask = gmask & (~np.isnan(a.s_px)) & (a.day >= lo_day) & (a.day <= hi_day)
        cidx = np.flatnonzero(cmask)
        if len(cidx) == 0:
            continue
        take = cidx if len(cidx) <= n_draws else rng.choice(
            cidx, size=n_draws, replace=False)
        for i in take:
            entry_day = int(a.day[i])
            entry_px = float(a.s_px[i])
            ex = _exit_for(a, entry_day, min_hold, hard)
            if ex is None:
                continue
            exit_day, exit_px, stale = ex
            hold = exit_day - entry_day
            acc = f.coupon / 100.0 / 365.0 * hold * 100.0
            ret = (exit_px - entry_px + acc) / entry_px
            rows.append({"six": f.six, "ret": ret, "hold": hold, "match": mi})
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
