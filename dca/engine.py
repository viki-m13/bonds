"""Biweekly DCA backtest engine with strict next-open execution.

Timing contract (anti-leakage):
  * Signal dates are scheduled trading days (every Nth trading day, default 10
    = biweekly). A signal may use data through the CLOSE of the signal date.
  * All buys/sells execute at the OPEN of the next trading day.
  * Scores are supplied as a precomputed DataFrame (date x ticker). The engine
    reads only row `signal_date`; causality of the score matrix itself is
    audited separately (see audit.py).

Accounting:
  * `contribution` dollars arrive at every signal date and are invested at the
    next open, split equally across the top-`k` eligible names.
  * Optional sell rule: a boolean DataFrame (date x ticker), True on a signal
    date => liquidate that holding at the next open. Proceeds sit in cash and
    are redeployed together with the next contribution.
  * Cost model: `cost_bps` applied to every trade notional (buys and sells).
  * Delistings: if a held name stops trading, it is sold at its final close
    (minus costs) and the proceeds rejoin cash at the next signal date.

Benchmark: identical cadence/contributions into a single ETF, same costs.
"""
import numpy as np
import pandas as pd

TRADING_DAYS_BIWEEKLY = 10


def schedule_dates(index: pd.DatetimeIndex, every: int = TRADING_DAYS_BIWEEKLY,
                   offset: int = 0, start=None, end=None) -> pd.DatetimeIndex:
    idx = index
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    return idx[offset::every]


class DCAResult:
    def __init__(self, value, invested, trades, holdings_log, cash,
                 holdings=None):
        self.value = value              # daily portfolio value (Series)
        self.invested = invested        # cumulative contributions (Series)
        self.trades = trades            # list of dicts
        self.holdings_log = holdings_log
        self.cash = cash
        self.holdings = holdings or {}  # ticker -> current market value

    @property
    def final_multiple(self):
        return float(self.value.iloc[-1] / self.invested.iloc[-1])

    def irr(self, freq_per_year=26):
        """Money-weighted annualized return via XIRR on contribution flows."""
        from scipy.optimize import brentq
        flows = self.invested.diff().fillna(self.invested.iloc[0])
        flows = flows[flows > 0]
        t_end = self.value.index[-1]
        yrs = np.array([(t_end - d).days / 365.25 for d in flows.index])
        amts = flows.values
        fv = float(self.value.iloc[-1])

        def f(r):
            return (amts * (1 + r) ** yrs).sum() - fv
        try:
            return brentq(f, -0.99, 5.0)
        except ValueError:
            return np.nan


def run_dca(open_px: pd.DataFrame, close_px: pd.DataFrame,
            scores: pd.DataFrame, member: pd.DataFrame | None,
            k: int = 3, every: int = TRADING_DAYS_BIWEEKLY, offset: int = 0,
            start=None, end=None, contribution: float = 1000.0,
            cost_bps: float = 5.0, sell: pd.DataFrame | None = None,
            min_history: int = 252) -> DCAResult:
    idx = close_px.index
    sig_dates = schedule_dates(idx, every, offset, start, end)
    # need a next open after the last signal date
    sig_dates = sig_dates[idx.searchsorted(sig_dates) + 1 < len(idx)]
    if len(sig_dates) == 0:
        raise ValueError("no signal dates in range")

    cost = cost_bps / 1e4
    cash = 0.0
    shares: dict[str, float] = {}
    trades, hlog = [], []
    value_rows = []
    invested_rows = []
    total_in = 0.0

    open_np = open_px
    enough_hist = close_px.notna().rolling(min_history).count() >= min_history

    sig_set = set(sig_dates)
    pos = idx.searchsorted(sig_dates[0])
    if end is not None:
        end_pos = idx.searchsorted(pd.Timestamp(end), side="right")
    else:
        end_pos = len(idx)
    last_close = {}

    pending_buy = None   # (exec_date, tickers)
    pending_sell = None

    for i in range(pos, end_pos):
        d = idx[i]
        # --- execute pending orders at today's open ---
        if pending_sell is not None and pending_sell[0] == d:
            for t in pending_sell[1]:
                if t in shares:
                    px = open_np.at[d, t]
                    if np.isnan(px):
                        px = last_close.get(t, np.nan)
                    if not np.isnan(px):
                        proceeds = shares[t] * px * (1 - cost)
                        cash += proceeds
                        trades.append({"date": d, "ticker": t, "side": "sell",
                                       "px": px, "notional": proceeds})
                    del shares[t]
            pending_sell = None
        if pending_buy is not None and pending_buy[0] == d:
            ticks = [t for t in pending_buy[1] if not np.isnan(open_np.at[d, t])]
            if ticks:
                per = cash / len(ticks)
                for t in ticks:
                    px = open_np.at[d, t]
                    sh = per * (1 - cost) / px
                    shares[t] = shares.get(t, 0.0) + sh
                    trades.append({"date": d, "ticker": t, "side": "buy",
                                   "px": px, "notional": per})
                cash = 0.0
            pending_buy = None

        # --- mark to market, handle delistings ---
        v = cash
        dead = []
        for t, sh in shares.items():
            c = close_px.at[d, t]
            if np.isnan(c):
                # no print today; if gone for good, liquidate at last close
                future = close_px[t].iloc[i + 1:i + 6]
                if future.isna().all() and i + 1 < len(idx):
                    cash_add = sh * last_close.get(t, 0.0) * (1 - cost)
                    cash += cash_add
                    v += cash_add
                    trades.append({"date": d, "ticker": t, "side": "delist",
                                   "px": last_close.get(t, 0.0),
                                   "notional": cash_add})
                    dead.append(t)
                else:
                    v += sh * last_close.get(t, 0.0)
            else:
                last_close[t] = c
                v += sh * c
        for t in dead:
            del shares[t]

        # --- signal date: queue orders for next open ---
        if d in sig_set:
            total_in += contribution
            cash += contribution
            nxt = idx[i + 1]
            row = scores.loc[d] if d in scores.index else None
            if row is not None:
                elig = row.copy()
                if member is not None:
                    elig = elig.where(member.loc[d])
                elig = elig.where(enough_hist.loc[d])
                elig = elig.dropna().sort_values(ascending=False)
                picks = list(elig.index[:k])
            else:
                picks = []
            if picks:
                pending_buy = (nxt, picks)
            if sell is not None and d in sell.index:
                to_sell = [t for t in shares if sell.at[d, t]]
                if to_sell:
                    pending_sell = (nxt, to_sell)

        value_rows.append(v)
        invested_rows.append(total_in)
        hlog.append({"date": d, "n_pos": len(shares), "cash": cash})

    span = idx[pos:end_pos]
    value = pd.Series(value_rows, index=span, name="value")
    invested = pd.Series(invested_rows, index=span, name="invested")
    holdings = {t: sh * last_close.get(t, np.nan) for t, sh in shares.items()}
    return DCAResult(value, invested, trades, hlog, cash, holdings)


def run_benchmark_dca(bench: pd.DataFrame, every: int = TRADING_DAYS_BIWEEKLY,
                      offset: int = 0, start=None, end=None,
                      contribution: float = 1000.0,
                      cost_bps: float = 5.0) -> DCAResult:
    idx = bench.index
    sig_dates = schedule_dates(idx, every, offset, start, end)
    sig_dates = sig_dates[idx.searchsorted(sig_dates) + 1 < len(idx)]
    cost = cost_bps / 1e4
    sh = 0.0
    total_in = 0.0
    rows, inv, trades = [], [], []
    sig_set = set(sig_dates)
    pos = idx.searchsorted(sig_dates[0])
    pending = False
    for i in range(pos, len(idx)):
        d = idx[i]
        if pending:
            px = bench["Open"].iloc[i]
            sh += total_pending * (1 - cost) / px
            trades.append({"date": d, "px": px, "notional": total_pending})
            pending = False
        if d in sig_set:
            total_in += contribution
            total_pending = contribution
            pending = True
        rows.append(sh * bench["Close"].iloc[i] + (total_pending if pending else 0.0))
        inv.append(total_in)
    span = idx[pos:]
    return DCAResult(pd.Series(rows, index=span), pd.Series(inv, index=span),
                     trades, [], 0.0)
