"""
VORTEX — Volatility-surface timing for leveraged ETFs
=====================================================

Thesis
------
Equity LETF forward returns are predictable from the shape of the VIX term
structure:
  * CONTANGO (front VIX < long-end VIX) → risk-on drift, favour leveraged
    equity momentum.
  * BACKWARDATION (front VIX > long-end VIX) → vol shock, crash ahead,
    favour safe havens (TMF/UBT/UGL).
  * NEUTRAL / stretched VIX → cash (BIL).

We synthesize the "long-end" VIX from realized vol (63d) because VXV is
not in our FRED folder.  This is a smoothed ex-post anchor; the front
VIX oscillates around it with a clear mean-reverting, predictive signal.

Hard constraints honoured
-------------------------
* One uniform N-day rebalance cadence (grid-searched over {1,3,5,10,21}).
* No look-ahead: every signal uses data through close[t-1];
  trades fill at open[t]; PnL = weight_t * (close[t]/open[t] - 1) on entry
  bar, then weight held at close[t] * (close[t+1]/close[t] - 1) on later
  bars (simpler close→close with entry slippage at open is used).
* 10 bps/side TC applied to any weight change at each rebalance.
* Broad universe (17 LETFs) + BIL cash.
* No daily vol scaling. Vol is used only as a SIGNAL.
* IS  : 2010-03-11 .. 2018-12-31  (param grid).
* OOS : 2019-01-02 .. 2026-04-02  (one-shot, locked params).
"""
from __future__ import annotations

import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF_DIR = ROOT / "data/etfs"
FRED_DIR = ROOT / "data/fred"
RESULTS_DIR = ROOT / "data/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IS_START = pd.Timestamp("2010-03-11")
IS_END = pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-02")
OOS_END = pd.Timestamp("2026-04-02")

TC_RATE = 10.0 / 1e4   # 10 bps/side

EQUITY_LETFS = ["UPRO", "TQQQ", "SSO", "QLD", "SOXL", "TECL", "FAS"]
SAFE_HAVEN = ["TMF", "UBT", "UGL"]
FULL_UNIVERSE = [
    "TQQQ", "UPRO", "QLD", "SSO", "SOXL", "TECL", "FAS", "ERX", "DRN",
    "EDC", "YINN", "UCO", "UGL", "NUGT", "TMF", "UBT", "TYD",
]
CASH = "BIL"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_etf(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(ETF_DIR / f"{ticker}.csv", parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    return df[["Open", "Close"]].astype(float)


def load_fred(name: str) -> pd.Series:
    df = pd.read_csv(FRED_DIR / f"{name}.csv", parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    return df.iloc[:, 0].astype(float)


def build_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    tickers = FULL_UNIVERSE + [CASH, "SPY"]
    opens = {}
    closes = {}
    for t in tickers:
        d = load_etf(t)
        opens[t] = d["Open"]
        closes[t] = d["Close"]
    opens = pd.DataFrame(opens).sort_index()
    closes = pd.DataFrame(closes).sort_index()

    # Master trading calendar = SPY dates intersected with BIL
    cal = closes[["SPY", CASH]].dropna().index
    opens = opens.reindex(cal).ffill()
    closes = closes.reindex(cal).ffill()

    vix = load_fred("VIXCLS").reindex(cal).ffill()
    spy_close = closes["SPY"]
    return opens, closes, vix, spy_close


# --------------------------------------------------------------------------- #
# Signal construction (all use data through close[t-1])
# --------------------------------------------------------------------------- #
def build_signals(closes: pd.DataFrame, vix: pd.Series, spy: pd.Series) -> pd.DataFrame:
    idx = closes.index

    # 1) Synthetic long-end VIX from 63d annualised realized SPY vol.
    #    Scale to VIX-comparable units (VIX is IV in % annualised).
    spy_ret = spy.pct_change()
    rv63 = spy_ret.rolling(63).std() * np.sqrt(252) * 100.0

    # 2) Vol-regime score: (front VIX - long VIX) / long VIX
    #    Here "long VIX" = max(rv63 scaled, floor) to avoid dividing by small n.
    long_vix = rv63.clip(lower=5.0)
    regime = (vix - long_vix) / long_vix

    # 3) VIX 21-day percentile rank vs trailing 252d.
    def pct_rank(x):
        # rank of last obs vs the window (0..1)
        if np.isnan(x[-1]):
            return np.nan
        return (np.sum(x <= x[-1]) - 1) / max(len(x) - 1, 1)

    vix_pct_rank = vix.rolling(252).apply(pct_rank, raw=True)

    # 4) Realized-vs-implied spread (realized - implied); negative = implied rich
    rvi_spread = rv63 - vix

    # 5) Smooth regime score (3d) to avoid day-to-day whipsaw.
    regime_s = regime.rolling(3).mean()

    sig = pd.DataFrame(
        {
            "vix": vix,
            "long_vix": long_vix,
            "regime": regime_s,
            "vix_pct": vix_pct_rank,
            "rvi_spread": rvi_spread,
            "rv63": rv63,
        },
        index=idx,
    )
    # Shift by 1: signals visible at open[t] came from close[t-1].
    return sig.shift(1)


def compute_momentum(closes: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """21-day total return per ticker, computed through close[t-1]."""
    mom = closes.pct_change(lookback)
    return mom.shift(1)


# --------------------------------------------------------------------------- #
# Strategy core
# --------------------------------------------------------------------------- #
def vortex_weights(
    closes: pd.DataFrame,
    signals: pd.DataFrame,
    momentum: pd.DataFrame,
    cadence: int,
    k: int,
    contango_thr: float,
    backwardation_thr: float,
    vix_pct_ceiling: float,
) -> pd.DataFrame:
    """Build daily weight matrix with a single N-day rebalance cadence."""
    idx = closes.index
    n_assets = closes.shape[1]
    W = pd.DataFrame(0.0, index=idx, columns=closes.columns)

    # Rebalance dates: every `cadence` bars, starting at position 0.
    rebal_mask = np.zeros(len(idx), dtype=bool)
    rebal_mask[::cadence] = True

    current_w = pd.Series(0.0, index=closes.columns)
    # Cash default initially
    current_w[CASH] = 1.0

    # Precompute for speed
    sig_vals = signals
    mom_vals = momentum

    for i, dt in enumerate(idx):
        if rebal_mask[i]:
            reg = sig_vals.at[dt, "regime"]
            vp = sig_vals.at[dt, "vix_pct"]
            rvi = sig_vals.at[dt, "rvi_spread"]

            target = pd.Series(0.0, index=closes.columns)

            if pd.isna(reg) or pd.isna(vp):
                target[CASH] = 1.0  # warm-up
            elif reg <= contango_thr and vp <= vix_pct_ceiling and (pd.isna(rvi) or rvi <= 0):
                # Contango + VIX not at panic ceiling + implied > realized
                #   → long top-K equity LETFs by 21-d momentum
                m = mom_vals.loc[dt, EQUITY_LETFS].dropna()
                if len(m) == 0:
                    target[CASH] = 1.0
                else:
                    top = m.nlargest(min(k, len(m)))
                    # Require positive momentum; drop negatives
                    top = top[top > 0]
                    if len(top) == 0:
                        target[CASH] = 1.0
                    else:
                        w = 1.0 / len(top)
                        for tkr in top.index:
                            target[tkr] = w
            elif reg >= backwardation_thr:
                # Deep backwardation → safe haven
                m = mom_vals.loc[dt, SAFE_HAVEN].dropna()
                if len(m) == 0:
                    target[CASH] = 1.0
                else:
                    # Go long whichever safe-haven LETFs have non-negative 21d mom;
                    # if none, just equal weight all three (flight-to-safety trade).
                    pos = m[m > 0]
                    pick = pos if len(pos) > 0 else m
                    w = 1.0 / len(pick)
                    for tkr in pick.index:
                        target[tkr] = w
            else:
                target[CASH] = 1.0

            current_w = target
        W.iloc[i] = current_w.values

    return W


def backtest(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    """
    Returns net portfolio returns.

    Execution convention: at bar t, the weight was *decided* using data through
    close[t-1]. We fill at open[t]. Bar-t PnL therefore uses the open→close
    intraday return, and close→close on subsequent holding days.

    The `weights` matrix from `vortex_weights` already represents the weight
    held between bar i and the next rebalance. To implement the open-fill we
    compute:
      daily_ret[t] = weight_on_bar[t] * (close[t] / ref[t] - 1)
      where ref[t] = open[t] if bar t is a rebalance (new weight starts at open),
            else close[t-1] (continue from prior close).
    Transaction cost: TC_RATE * sum(|Δw|) applied on the rebalance day.
    """
    # Rebalance detection: bar i is rebalance if weights changed vs i-1.
    w = weights
    w_prev = w.shift(1).fillna(0.0)
    dw = (w - w_prev).abs().sum(axis=1)
    rebal = dw > 1e-10

    close_prev = closes.shift(1)
    # Reference price per asset per bar
    ref = close_prev.copy()
    # On rebalance days, ref = open (for assets with weight on this bar).
    # But we need per-asset: use open on rebalance bar for any asset held.
    open_df = opens
    # A cleaner approach: bar return per asset is:
    #   if rebal[t]: (close[t]/open[t] - 1)
    #   else:        (close[t]/close[t-1] - 1)
    intraday = closes / open_df - 1.0
    c2c = closes / close_prev - 1.0

    # Asset bar returns under this convention
    asset_ret = c2c.copy()
    asset_ret.loc[rebal] = intraday.loc[rebal]

    # Portfolio gross return
    port_ret = (w * asset_ret).sum(axis=1)

    # Transaction costs on rebalance days
    tc = dw * TC_RATE
    net = port_ret - tc
    net = net.fillna(0.0)
    return net


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def perf(ret: pd.Series) -> dict:
    r = ret.dropna()
    if len(r) < 20:
        return {"sharpe": np.nan, "cagr": np.nan, "mdd": np.nan, "vol": np.nan, "n": int(len(r))}
    mu = r.mean() * 252
    sd = r.std(ddof=0) * np.sqrt(252)
    sharpe = mu / sd if sd > 0 else np.nan
    cum = (1 + r).cumprod()
    years = len(r) / 252
    cagr = cum.iloc[-1] ** (1 / years) - 1
    mdd = (cum / cum.cummax() - 1).min()
    return {"sharpe": float(sharpe), "cagr": float(cagr), "mdd": float(mdd),
            "vol": float(sd), "n": int(len(r))}


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def main() -> None:
    print("Loading panels...")
    opens, closes, vix, spy = build_panels()
    print(f"Calendar: {opens.index[0].date()} -> {opens.index[-1].date()}  "
          f"({len(opens)} bars)")

    signals = build_signals(closes, vix, spy)
    momentum = compute_momentum(closes, 21)

    # --------------------------------------------------------------- IS grid
    cadences = [1, 3, 5, 10, 21]
    ks = [2, 3, 4, 5]
    contango_ths = [-0.05, -0.10, -0.15]    # require regime <= this (contango)
    back_ths = [0.20, 0.35, 0.50]            # require regime >= this (backwardation)
    vix_ceils = [0.80, 0.90]                 # VIX percentile ceiling

    is_slice = slice(IS_START, IS_END)
    oos_slice = slice(OOS_START, OOS_END)

    grid_rows = []
    best = None
    total = len(cadences) * len(ks) * len(contango_ths) * len(back_ths) * len(vix_ceils)
    done = 0
    for cad, k, cth, bth, vc in product(cadences, ks, contango_ths, back_ths, vix_ceils):
        W = vortex_weights(closes, signals, momentum, cad, k, cth, bth, vc)
        ret = backtest(opens, closes, W)
        is_ret = ret.loc[is_slice]
        m = perf(is_ret)
        row = {
            "cadence": cad, "k": k, "contango_thr": cth,
            "backwardation_thr": bth, "vix_pct_ceil": vc,
            "is_sharpe": m["sharpe"], "is_cagr": m["cagr"], "is_mdd": m["mdd"],
        }
        grid_rows.append(row)
        if best is None or (m["sharpe"] is not np.nan and m["sharpe"] > best["is_sharpe"]):
            best = row.copy()
            best_ret = ret
            best_W = W
        done += 1
        if done % 30 == 0 or done == total:
            print(f"  grid {done}/{total} best IS Sharpe so far: "
                  f"{best['is_sharpe']:.3f} @ cad={best['cadence']} k={best['k']} "
                  f"cth={best['contango_thr']} bth={best['backwardation_thr']} "
                  f"vc={best['vix_pct_ceil']}")

    # Save grid
    grid_df = pd.DataFrame(grid_rows).sort_values("is_sharpe", ascending=False)
    grid_df.to_csv(RESULTS_DIR / "vortex_grid.csv", index=False)

    # --------------------------------------------------------------- final eval
    params = {
        "cadence": best["cadence"],
        "k": best["k"],
        "contango_thr": best["contango_thr"],
        "backwardation_thr": best["backwardation_thr"],
        "vix_pct_ceil": best["vix_pct_ceil"],
    }
    final_ret = best_ret
    # Split
    is_m = perf(final_ret.loc[is_slice])
    oos_m = perf(final_ret.loc[oos_slice])
    full_m = perf(final_ret.loc[IS_START:OOS_END])

    out = {
        "strategy": "VORTEX",
        "params": params,
        "is": is_m,
        "oos": oos_m,
        "full": full_m,
        "is_window": [str(IS_START.date()), str(IS_END.date())],
        "oos_window": [str(OOS_START.date()), str(OOS_END.date())],
        "tc_bps_per_side": 10,
        "universe": FULL_UNIVERSE + [CASH],
    }

    with open(RESULTS_DIR / "vortex_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    # Save returns
    ret_df = pd.DataFrame({"ret": final_ret})
    ret_df.index.name = "Date"
    ret_df.to_csv(RESULTS_DIR / "vortex_returns.csv")

    # Report
    print("\n=== VORTEX — final results ===")
    print(f"Chosen params: {params}")
    for name, m in [("IS", is_m), ("OOS", oos_m), ("FULL", full_m)]:
        print(f"  {name:4s}  Sharpe={m['sharpe']:.3f}  CAGR={m['cagr']*100:.2f}%  "
              f"MDD={m['mdd']*100:.2f}%  Vol={m['vol']*100:.2f}%  n={m['n']}")


if __name__ == "__main__":
    main()
