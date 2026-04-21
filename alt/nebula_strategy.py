"""NEBULA — Cross-asset LETF pairs relative-value strategy.

Design
------
At each rebalance date t:
  - For each pair (A, B) with enough history:
      spread_t = log(priceA_{t-1} / priceB_{t-1})
      z_t     = (spread_t - mean_{lookback}) / std_{lookback}
      If |z_t| > threshold, take +1 on cheap leg, -1 on rich leg.
  - Equal-weight across ALL active pairs (one weight per leg, so each active
    pair uses 1/N_active of capital, split long/short).
  - Remainder (i.e. inactive pair buckets) -> BIL.

Execution
---------
  - Signal uses close[t-1], execution at open[t].
  - Holds positions for `rebal` days, rebalances on a uniform cadence.
  - Transaction cost: tc_bps per side applied on turnover.
  - Short borrow: borrow_bps_per_day charged on gross short notional daily.

Single uniform cadence in {5, 10, 21, 42}.
IS: 2010-03-11 to 2018-12-31; OOS: 2019-01-02 to 2026-04-02.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("/home/user/bonds/data/etfs")
OUT_DIR = Path("/home/user/bonds/data/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IS_START = pd.Timestamp("2010-03-11")
IS_END = pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-02")
OOS_END = pd.Timestamp("2026-04-02")

PAIRS = [
    ("QLD", "TQQQ"),
    ("SSO", "UPRO"),
    ("TMF", "UBT"),
    ("SOXL", "TECL"),
    ("FAS", "UPRO"),
    ("DRN", "UPRO"),
    ("ERX", "UCO"),
]
CASH = "BIL"

TC_BPS_SIDE = 7.5           # 7.5 bps per side (mid of 5-10)
BORROW_BPS_PER_DAY = 10.0   # 10 bps/day on shorts (~25% annualized)

# -------------------------------------------------------------------------
def load_ohlc(ticker: str) -> pd.DataFrame:
    p = DATA_DIR / f"{ticker}.csv"
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[["Open", "Close"]].astype(float)
    return df


def build_panel(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    opens = {}
    closes = {}
    for t in tickers:
        df = load_ohlc(t)
        opens[t] = df["Open"]
        closes[t] = df["Close"]
    op = pd.DataFrame(opens).sort_index()
    cl = pd.DataFrame(closes).sort_index()
    # align calendar to intersection (trading days)
    common = op.dropna(how="any").index.intersection(cl.dropna(how="any").index)
    return op.loc[common], cl.loc[common]


def metrics(rets: pd.Series) -> dict:
    rets = rets.dropna()
    if len(rets) == 0 or rets.std() == 0:
        return {"Sharpe": 0.0, "CAGR": 0.0, "MDD": 0.0, "Vol": 0.0, "N": 0}
    mu = rets.mean() * 252
    sd = rets.std(ddof=0) * np.sqrt(252)
    sharpe = mu / sd if sd > 0 else 0.0
    eq = (1 + rets).cumprod()
    years = len(rets) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
    mdd = (eq / eq.cummax() - 1).min()
    return {
        "Sharpe": float(sharpe),
        "CAGR": float(cagr),
        "MDD": float(mdd),
        "Vol": float(sd),
        "N": int(len(rets)),
    }


# -------------------------------------------------------------------------
def run_strategy(opens: pd.DataFrame, closes: pd.DataFrame,
                 pairs: list[tuple[str, str]],
                 lookback: int, z_thresh: float, rebal: int,
                 tc_bps: float = TC_BPS_SIDE,
                 borrow_bps_per_day: float = BORROW_BPS_PER_DAY,
                 ) -> pd.Series:
    """Run NEBULA pairs strategy. Returns daily net return series."""
    dates = opens.index
    # open-to-open daily return per asset
    # ret_open[t] = open[t+1] / open[t] - 1  (realized holding from open t to open t+1)
    # We'll use: for each day t, asset daily return = open[t+1]/open[t] - 1
    # Align so position held from open[t] through open[t+1].
    open_ret = opens.shift(-1) / opens - 1.0  # realized next-day open-open return
    open_ret = open_ret.iloc[:-1]  # last row has NaN
    valid_dates = open_ret.index

    # precompute log-ratio per pair on close prices
    log_ratios = {}
    for A, B in pairs:
        if A in closes.columns and B in closes.columns:
            log_ratios[(A, B)] = np.log(closes[A] / closes[B])

    # weights DataFrame — one column per ticker
    w = pd.DataFrame(0.0, index=valid_dates, columns=opens.columns)

    # Rebalance dates: uniform cadence starting from first valid date after lookback
    first_idx = lookback + 2  # need lookback history + 1 for z calc
    rebal_idx = list(range(first_idx, len(valid_dates), rebal))

    current_w = pd.Series(0.0, index=opens.columns)

    for i in range(len(valid_dates)):
        if i in rebal_idx:
            t = valid_dates[i]
            # signal uses close[t-1]: the close of the previous trading day
            # t is a date in valid_dates; close[t-1] means closes.loc[prev_close_date]
            # prev_close_date = last date in closes where date < t
            pos_in_closes = closes.index.get_indexer([t])[0]
            if pos_in_closes <= 0:
                continue
            sig_date = closes.index[pos_in_closes - 1]

            # reset
            current_w = pd.Series(0.0, index=opens.columns)
            active_count = 0
            leg_signals = []  # list of (A, B, direction) where direction=+1 means long A / short B

            for (A, B), lr in log_ratios.items():
                if sig_date not in lr.index:
                    continue
                window = lr.loc[:sig_date].iloc[-(lookback + 1):-1]  # pure historical
                if len(window) < lookback:
                    continue
                mu = window.mean()
                sd = window.std(ddof=0)
                if sd == 0 or np.isnan(sd):
                    continue
                z = (lr.loc[sig_date] - mu) / sd
                if np.isnan(z):
                    continue
                if z > z_thresh:
                    # A rich vs B -> short A, long B
                    leg_signals.append((A, B, -1))
                    active_count += 1
                elif z < -z_thresh:
                    # A cheap vs B -> long A, short B
                    leg_signals.append((A, B, +1))
                    active_count += 1

            if active_count > 0:
                # each active pair uses 1/active_count of capital; split +0.5/-0.5 (net zero, gross = 1 per pair slot)
                per = 1.0 / active_count
                for A, B, d in leg_signals:
                    current_w[A] = current_w.get(A, 0.0) + d * per * 0.5
                    current_w[B] = current_w.get(B, 0.0) - d * per * 0.5
                # allocate inactive-slot remainder to cash: none — all active. Fine.
                # The remaining 0 capital -> BIL (unused legs). Actually gross is per/pair
                # leaving (1 - active_count/active_count) = 0 in BIL. So no BIL.
                # But we defined "equal-weight across pairs; otherwise cash". The only cash
                # case is 0 active pairs.
            else:
                current_w[CASH] = 1.0

        w.iloc[i] = current_w.values

    # Compute turnover: |w_t - w_{t-1}| summed across tickers, charged at open of day t
    w_shift = w.shift(1).fillna(0.0)
    turnover = (w - w_shift).abs().sum(axis=1)
    tc = turnover * (tc_bps / 1e4)

    # Gross returns: sum over tickers of w_i * open_ret_i (open-to-open)
    # For BIL we use its own open-return series from the panel (flat-ish).
    gross = (w * open_ret.reindex(columns=w.columns).fillna(0.0)).sum(axis=1)

    # Short borrow cost: charge daily on gross short notional (sum of negative weights)
    short_notional = (-w.clip(upper=0.0)).sum(axis=1)  # positive values = short size
    borrow_cost = short_notional * (borrow_bps_per_day / 1e4)

    net = gross - tc - borrow_cost
    net.name = "nebula"
    return net


# -------------------------------------------------------------------------
def slice_window(r: pd.Series, start, end) -> pd.Series:
    return r.loc[(r.index >= start) & (r.index <= end)]


def main():
    tickers = sorted({t for pair in PAIRS for t in pair} | {CASH})
    opens, closes = build_panel(tickers)
    print(f"Loaded panel: {opens.shape}, {opens.index.min().date()} to {opens.index.max().date()}")
    print(f"Tickers: {list(opens.columns)}")

    # Grid search on IS
    lookbacks = [63, 126]
    thresholds = [1.0, 1.5, 2.0]
    rebals = [5, 10, 21, 42]

    grid_rows = []
    returns_cache = {}
    for lb, th, rb in itertools.product(lookbacks, thresholds, rebals):
        r = run_strategy(opens, closes, PAIRS, lb, th, rb)
        r_nb = run_strategy(opens, closes, PAIRS, lb, th, rb, borrow_bps_per_day=0.0)
        returns_cache[(lb, th, rb)] = r
        r_is = slice_window(r, IS_START, IS_END)
        r_is_nb = slice_window(r_nb, IS_START, IS_END)
        m = metrics(r_is)
        m_nb = metrics(r_is_nb)
        grid_rows.append({"lookback": lb, "z": th, "rebal": rb, **m,
                          "Sharpe_noBorrow": m_nb["Sharpe"],
                          "CAGR_noBorrow": m_nb["CAGR"]})
        print(f"IS lb={lb} z={th} rb={rb} -> Sharpe={m['Sharpe']:.3f} "
              f"(noBorrow={m_nb['Sharpe']:.3f}) "
              f"CAGR={m['CAGR']*100:.2f}% MDD={m['MDD']*100:.2f}%")

    grid = pd.DataFrame(grid_rows)
    # Pick winner by IS Sharpe (tie-break: higher CAGR)
    grid_sorted = grid.sort_values(["Sharpe", "CAGR"], ascending=[False, False])
    winner = grid_sorted.iloc[0]
    lb_w, th_w, rb_w = int(winner.lookback), float(winner.z), int(winner.rebal)
    print("\n=== WINNER (IS) ===")
    print(winner)

    r_full = returns_cache[(lb_w, th_w, rb_w)]
    r_is = slice_window(r_full, IS_START, IS_END)
    r_oos = slice_window(r_full, OOS_START, OOS_END)
    r_fw = slice_window(r_full, IS_START, OOS_END)

    m_is = metrics(r_is)
    m_oos = metrics(r_oos)
    m_full = metrics(r_fw)

    print("\n=== METRICS ===")
    print("IS:  ", m_is)
    print("OOS: ", m_oos)
    print("FULL:", m_full)

    out = {
        "strategy": "NEBULA",
        "description": "Cross-asset LETF pairs relative-value (long/short) with uniform rebalance.",
        "pairs": [list(p) for p in PAIRS],
        "cash": CASH,
        "tc_bps_per_side": TC_BPS_SIDE,
        "borrow_bps_per_day": BORROW_BPS_PER_DAY,
        "chosen": {"lookback": lb_w, "z_threshold": th_w, "rebal_days": rb_w},
        "metrics": {"IS": m_is, "OOS": m_oos, "FULL": m_full},
        "grid_top5": grid_sorted.head(5).to_dict(orient="records"),
        "dates": {
            "IS_start": str(IS_START.date()), "IS_end": str(IS_END.date()),
            "OOS_start": str(OOS_START.date()), "OOS_end": str(OOS_END.date()),
        },
    }

    with open(OUT_DIR / "nebula_metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    r_fw.to_csv(OUT_DIR / "nebula_returns.csv", header=True)
    grid.to_csv(OUT_DIR / "nebula_grid.csv", index=False)
    print(f"\nWrote {OUT_DIR/'nebula_metrics.json'} and nebula_returns.csv")


if __name__ == "__main__":
    main()
