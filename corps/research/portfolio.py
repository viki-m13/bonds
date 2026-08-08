"""Stateful monthly-rebalance portfolio simulator on the compact cache.

Honesty rules:
- Scores at month-end t use data through t only (caller's score_fn contract).
- Entries: buy at the FIRST customer-ask print in (t, t+7]; no print = no fill.
- Exits: sell at the FIRST customer-bid print in (t, t+10]; if none, the
  position STAYS OPEN and is retried at the next rebalance (illiquidity is
  suffered, not assumed away).
- A bond whose prints stop entirely (default/call/maturity — indistinguishable
  here) is force-exited at its LAST available bid, never redeemed at par.
- Marks: actual mid prints (stale = flat) + daily coupon accrual; cash earns
  the 3M T-bill. NAV is daily; Sharpe is reported at monthly frequency.

score_fn(b, t) -> float score for bond dict b at day t (higher = buy), or None
                  if not scoreable. Must use only rows with day <= t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine2 as e2


def month_ends(lo, hi):
    d = pd.date_range(pd.Timestamp(lo, unit="D"), pd.Timestamp(hi, unit="D"),
                      freq="ME")
    return [int(x) for x in d.values.astype("datetime64[D]").astype(np.int64)]


def _first_print(day, px, lo, hi):
    j = np.searchsorted(day, lo, side="left")
    if j < len(day) and day[j] <= hi:
        return int(day[j]), float(px[j])
    return None


class Position:
    __slots__ = ("six", "entry_day", "entry_px", "coupon", "exit_day", "exit_px")

    def __init__(self, six, ed, ep, coupon):
        self.six, self.entry_day, self.entry_px = six, ed, ep
        self.coupon = coupon
        self.exit_day = self.exit_px = None


def run_portfolio(bonds, score_fn, lo, hi, top_n=50, hold_until_rank=None,
                  extra_gate=None, verbose=False):
    """Monthly loop. Buy top_n by score; keep an existing position while its
    rank stays within hold_until_rank (default 2*top_n) — turnover control.
    Returns closed+open Position list with real fills."""
    if hold_until_rank is None:
        hold_until_rank = 2 * top_n
    mes = month_ends(lo, hi)
    open_pos: dict[str, Position] = {}
    closed: list[Position] = []
    # precompute per-bond print arrays
    P = {}
    for six, b in bonds.items():
        day = b["day"]
        sm = ~np.isnan(b["s_px"]); pm = ~np.isnan(b["p_px"])
        P[six] = (day[sm], b["s_px"][sm], day[pm], b["p_px"][pm])
    for t in mes:
        # score the eligible universe
        scores = {}
        for six, b in bonds.items():
            day = b["day"]
            i = np.searchsorted(day, t, side="right") - 1
            if i < 0 or t - day[i] > 30:          # no recent print
                continue
            if not b["elig"][i]:
                continue
            if extra_gate is not None:
                g = extra_gate(b)
                if g is None or not g[i]:
                    continue
            sc = score_fn(b, t, i)
            if sc is not None and np.isfinite(sc):
                scores[six] = sc
        if not scores:
            continue
        ranked = sorted(scores, key=scores.get, reverse=True)
        rank = {s: k for k, s in enumerate(ranked)}
        target = set(ranked[:top_n])
        # exits: positions that fell out of the hold band, or whose bond died
        for six in list(open_pos):
            pos = open_pos[six]
            r = rank.get(six)
            die = r is None or r >= hold_until_rank
            if not die:
                continue
            s_day, s_px, p_day, p_px = P[six]
            fp = _first_print(p_day, p_px, t + 1, t + 10)
            if fp is None:
                # no bid within 10d: if the bond still prints later, retry next
                # month; if prints are over, force-exit at last available bid
                if len(p_day) and p_day[-1] <= t:
                    pos.exit_day, pos.exit_px = int(p_day[-1]), float(p_px[-1])
                    closed.append(pos); del open_pos[six]
                continue
            pos.exit_day, pos.exit_px = fp
            closed.append(pos); del open_pos[six]
        # entries
        for six in ranked[:top_n]:
            if six in open_pos or len(open_pos) >= top_n:
                continue
            s_day, s_px, p_day, p_px = P[six]
            fp = _first_print(s_day, s_px, t + 1, t + 7)
            if fp is None:
                continue
            ed, ep = fp
            open_pos[six] = Position(six, ed, ep, bonds[six]["coupon"])
        if verbose:
            print(f"  {pd.Timestamp(t, unit='D').date()} open={len(open_pos)}", flush=True)
    # close remaining at last bid
    for six, pos in open_pos.items():
        _, _, p_day, p_px = P[six]
        if len(p_day):
            pos.exit_day, pos.exit_px = int(p_day[-1]), float(p_px[-1])
            closed.append(pos)
    return closed


def positions_to_fills(closed):
    out = []
    for p in closed:
        if p.exit_day is None or p.exit_day <= p.entry_day:
            continue
        out.append(e2.Fill(p.six, p.entry_day, p.entry_px,
                           p.exit_day, p.exit_px, p.coupon, False))
    return out


def random_control_portfolio(bonds, score_fn_universe, lo, hi, top_n, n_reps=5,
                             extra_gate=None, seed=3):
    """Control: identical mechanics, random scores — isolates selection skill
    from carry/universe drift."""
    rng = np.random.default_rng(seed)
    stats = []
    for rep in range(n_reps):
        def rnd(b, t, i, _r=rng):
            s = score_fn_universe(b, t, i)
            return None if s is None else float(_r.random())
        closed = run_portfolio(bonds, rnd, lo, hi, top_n=top_n,
                               extra_gate=extra_gate)
        fills = positions_to_fills(closed)
        if not fills:
            continue
        r = e2.mtm_nav(bonds, fills)
        if r is None:
            continue
        days, nav, daily = r
        stats.append(e2.perf_stats(days, nav, daily))
    return stats
