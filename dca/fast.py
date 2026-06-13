"""Vectorized DCA evaluation (lot-based), verified against engine.py.

Key fact: with next-open execution and proceeds-recycled-at-next-buy
accounting, the portfolio is a collection of lots. Each biweekly contribution
buys equal dollar slices of the top-k picks at the next open. A lot lives
until (a) a sell signal fires (liquidated at next open, proceeds join the next
buy), (b) the stock delists (liquidated at last close), or (c) evaluation end.

This module computes final values for many (start, end) windows quickly by a
forward pass over signal dates only.
"""
import numpy as np
import pandas as pd

from engine import schedule_dates


class FastData:
    """Pre-extracted numpy views shared across many runs."""

    def __init__(self, open_px: pd.DataFrame, close_px: pd.DataFrame,
                 member: pd.DataFrame, min_history: int = 252):
        self.index = close_px.index
        self.columns = close_px.columns
        self.open = open_px.to_numpy(float)
        self.close = close_px.to_numpy(float)
        self.member = member.to_numpy(bool)
        valid = close_px.notna()
        self.enough = (valid.rolling(min_history).count() >= min_history
                       ).to_numpy(bool)
        # last valid close index per column, for delisting liquidation
        self.last_valid = np.array([
            valid[c].to_numpy().nonzero()[0].max() if valid[c].any() else -1
            for c in close_px.columns])
        # forward-filled close for marking (delisted names hold last price)
        self.close_ff = close_px.ffill().to_numpy(float)

    def sig_positions(self, every=10, offset=0, start=None, end=None):
        d = schedule_dates(self.index, every, offset, start, end)
        p = self.index.searchsorted(d)
        return p[p + 1 < len(self.index)]


def run_fast(fd: FastData, scores: np.ndarray, k=3, every=10, offset=0,
             start=None, end=None, contribution=1000.0, cost_bps=5.0,
             sell: np.ndarray | None = None, eval_positions=None,
             trim_cap: float | None = None,
             trim_period: str | None = None, return_holdings=False):
    """Forward pass. `scores`/`sell` are numpy (days x tickers) aligned to fd.
    Returns (eval_positions, values, invested) where values[j] is portfolio
    value at fd.index[eval_positions[j]] for a DCA starting at `start`.
    If eval_positions is None, evaluates only at the final available day."""
    cost = cost_bps / 1e4
    sig = fd.sig_positions(every, offset, start, end)
    if end is not None:
        end_pos = fd.index.searchsorted(pd.Timestamp(end), side="right") - 1
    else:
        end_pos = len(fd.index) - 1
    if eval_positions is None:
        eval_positions = np.array([end_pos])
    eval_positions = np.asarray(sorted(eval_positions))

    # holdings: ticker idx -> shares
    shares: dict[int, float] = {}
    lots_at: list[tuple[int, dict]] = []   # snapshots not needed; we evaluate online
    values = np.zeros(len(eval_positions))
    invested = np.zeros(len(eval_positions))
    total_in = 0.0
    cash = 0.0
    ev_i = 0
    n_eval = len(eval_positions)

    def mark(pos):
        v = cash
        for t, sh in shares.items():
            v += sh * fd.close_ff[pos, t]
        return v

    def _pkey(ts):
        if trim_period == "monthly":
            return (ts.year, ts.month)
        if trim_period == "quarterly":
            return (ts.year, (ts.month - 1) // 3)
        return ts.year                      # annual
    prev_key = None

    prev_pos = sig[0]
    for si, p in enumerate(sig):
        # evaluate any eval positions strictly before this signal's exec
        while ev_i < n_eval and eval_positions[ev_i] < p:
            ep = eval_positions[ev_i]
            if ep >= prev_pos:
                values[ev_i] = mark(ep)
                invested[ev_i] = total_in
            ev_i += 1
        if p > end_pos - 1:
            break
        # delistings since last signal: liquidate at last close
        dead = [t for t in shares if fd.last_valid[t] <= p]
        for t in dead:
            cash += shares.pop(t) * fd.close_ff[fd.last_valid[t], t] * (1 - cost)

        total_in += contribution
        cash += contribution
        ex = p + 1
        # sells queued at signal date, executed at next open
        if sell is not None:
            for t in [t for t in shares if sell[p, t]]:
                px = fd.open[ex, t]
                if np.isnan(px):
                    px = fd.close_ff[p, t]
                cash += shares.pop(t) * px * (1 - cost)
        # concentration trim: sell each holding's excess over trim_cap (by
        # close-of-signal weight) at the next open; proceeds redeployed in the
        # buy below. Partial sells only -> most of the book (and its deferred
        # gains) is left untouched.
        do_trim = False
        if trim_cap is not None and trim_period is not None:
            key = _pkey(fd.index[p])
            if prev_key is not None and key != prev_key:
                do_trim = True
            prev_key = key
        if do_trim and shares:
            hv = {t: shares[t] * fd.close_ff[p, t] for t in shares}
            tot = sum(v for v in hv.values() if v == v)
            if tot > 0:
                for t, val in list(hv.items()):
                    if val == val and val > trim_cap * tot:
                        cpx = fd.close_ff[p, t]
                        if cpx <= 0:
                            continue
                        sh_sell = min((val - trim_cap * tot) / cpx, shares[t])
                        opx = fd.open[ex, t]
                        if np.isnan(opx):
                            opx = cpx
                        cash += sh_sell * opx * (1 - cost)
                        shares[t] -= sh_sell
                        if shares[t] <= 1e-12:
                            del shares[t]
        # pick top-k eligible
        row = scores[p].copy()
        mask = fd.member[p] & fd.enough[p] & ~np.isnan(fd.close[p])
        row[~mask] = np.nan
        ok = ~np.isnan(row)
        nok = ok.sum()
        if nok:
            kk = min(k, nok)
            picks = np.argpartition(-np.where(ok, row, -np.inf), kk - 1)[:kk]
            opx = fd.open[ex, picks]
            good = ~np.isnan(opx)
            picks, opx = picks[good], opx[good]
            if len(picks):
                per = cash / len(picks) * (1 - cost)
                for t, px in zip(picks, opx):
                    shares[t] = shares.get(t, 0.0) + per / px
                cash = 0.0
        prev_pos = p

    while ev_i < n_eval:
        values[ev_i] = mark(eval_positions[ev_i])
        invested[ev_i] = total_in
        ev_i += 1
    if return_holdings:
        cols = fd.columns
        ep = eval_positions[-1]
        holdings = {cols[t]: sh * fd.close_ff[ep, t]
                    for t, sh in shares.items()}
        return eval_positions, values, invested, holdings
    return eval_positions, values, invested


def bench_fast(bench: pd.DataFrame, every=10, offset=0, start=None, end=None,
               contribution=1000.0, cost_bps=5.0, eval_dates=None):
    """Vectorized benchmark DCA evaluated at eval_dates (defaults to end)."""
    idx = bench.index
    sig = schedule_dates(idx, every, offset, start, end)
    sig = sig[idx.searchsorted(sig) + 1 < len(idx)]
    pos = idx.searchsorted(sig)
    opx = bench["Open"].to_numpy()[pos + 1]
    cost = cost_bps / 1e4
    sh_per = contribution * (1 - cost) / opx          # shares from each lot
    csh = np.cumsum(sh_per)
    close = bench["Close"].to_numpy()
    if eval_dates is None:
        eval_dates = [idx[-1]]
    out = []
    for ed in eval_dates:
        ep = idx.searchsorted(pd.Timestamp(ed), side="right") - 1
        n = np.searchsorted(pos + 1, ep, side="right")  # lots executed by ep
        if n == 0:
            out.append((np.nan, 0.0))
            continue
        # value: executed lots at close; a pending lot (signal before ep,
        # exec after) counts as cash
        pend = contribution if (np.searchsorted(pos, ep, side="right") > n) else 0.0
        out.append((csh[n - 1] * close[ep] + pend,
                    contribution * np.searchsorted(pos, ep, side="right")))
    return out
