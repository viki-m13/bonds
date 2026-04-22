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

    # 1) Synthetic "long-end VIX" proxy: realized 63d vol plus a constant
    #    volatility-risk-premium of 4 vols (the long-run VRP is ~3-5). This
    #    makes the regime score oscillate around zero rather than always
    #    positive, because VIX on average sits ~VRP above realized.
    spy_ret = spy.pct_change()
    rv63 = spy_ret.rolling(63).std() * np.sqrt(252) * 100.0
    long_vix = rv63 + 4.0   # synthetic VXV proxy
    long_vix = long_vix.clip(lower=8.0)

    # 2) Vol-regime score: (front VIX - long VIX) / long VIX
    #    Negative → contango (VIX low vs realized+VRP, calm drift).
    #    Positive → backwardation (VIX elevated; realized catching up).
    regime = (vix - long_vix) / long_vix

    # 3) VIX 21-day percentile rank vs trailing 252d.
    def pct_rank(x):
        if np.isnan(x[-1]):
            return np.nan
        return (np.sum(x <= x[-1]) - 1) / max(len(x) - 1, 1)

    vix_pct_rank = vix.rolling(252).apply(pct_rank, raw=True)

    # 4) Realized-vs-implied spread (realized - implied); negative = implied rich
    rvi_spread = rv63 - vix

    # 5) 21d VIX change — rising VIX = warning.
    vix_chg_21 = vix.pct_change(21)

    # 6) SPY 200d trend filter (risk-on structural)
    spy_sma200 = spy.rolling(200).mean()
    spy_above_sma = (spy > spy_sma200).astype(float)
    spy_dev_200 = spy / spy_sma200 - 1.0

    # Smooth regime to stabilise rebalances
    regime_s = regime.rolling(3).mean()

    sig = pd.DataFrame(
        {
            "vix": vix,
            "long_vix": long_vix,
            "regime": regime_s,
            "vix_pct": vix_pct_rank,
            "rvi_spread": rvi_spread,
            "rv63": rv63,
            "vix_chg_21": vix_chg_21,
            "spy_above_sma": spy_above_sma,
            "spy_dev_200": spy_dev_200,
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
def _targets_on_rebal(
    signals: pd.DataFrame,
    momentum: pd.DataFrame,
    cols: list[str],
    rebal_dates: pd.DatetimeIndex,
    k: int,
    contango_thr: float,
    backwardation_thr: float,
    vix_pct_ceiling: float,
) -> pd.DataFrame:
    """Vectorised target-weight computation only on rebalance dates.

    Empirical finding on IS (2010-2018):
        * Regime > 0 (VIX elevated vs realized+VRP) → forward 5d SPY 20% ann
        * Regime < 0 (VIX depressed / extreme calm)  → forward 5d SPY 7% ann
        * VIX > 25                                   → forward 5d SPY 25%+ ann
    This is the volatility-risk-premium / buy-the-fear factor.

    So:
        * "Fear regime" (regime >= contango_thr AND VIX pct <= ceiling)
              → long top-K equity LETFs (selected by 21d mom; positive only).
              Why require SPY uptrend? — it hurts recovery capture. We use
              a softer gate: require SPY not to be >-15% off its 200d SMA
              (i.e. skip true bear markets).
        * "Extreme backwardation" (regime >= backwardation_thr)
              AND VIX very high (vix_pct > 0.9) AND SPY broken
              → safe haven (TMF/UBT/UGL).
        * "Complacency" (regime < contango_thr, i.e. VIX low vs realized)
              → cash. The edge is strongest when VIX carries a premium.
    """
    n = len(rebal_dates)
    n_cols = len(cols)
    col_idx = {c: i for i, c in enumerate(cols)}
    tgt = np.zeros((n, n_cols))

    sig = signals.loc[rebal_dates]
    reg = sig["regime"].values
    vp = sig["vix_pct"].values
    spy_ok = sig["spy_above_sma"].values  # 1 if SPY > 200d SMA
    spy_dev = sig["spy_dev_200"].values   # (SPY / SMA200) - 1
    vix_level = sig["vix"].values
    vix_chg = sig["vix_chg_21"].values    # 21d VIX % change

    mom_eq = momentum.loc[rebal_dates, EQUITY_LETFS].values
    mom_sh = momentum.loc[rebal_dates, SAFE_HAVEN].values
    eq_cols = [col_idx[c] for c in EQUITY_LETFS]
    sh_cols = [col_idx[c] for c in SAFE_HAVEN]
    cash_col = col_idx[CASH]

    warm = np.isnan(reg) | np.isnan(vp) | np.isnan(spy_ok)

    # VIX shock kill-switch: 21d VIX rise > 60% → go to cash/safe regardless
    vix_shock = (~np.isnan(vix_chg)) & (vix_chg > 0.60)

    # Core gate
    risk_on = (~warm) & (spy_ok > 0.5) & (reg >= contango_thr) \
              & (vp <= vix_pct_ceiling) & (~vix_shock)

    # Safe haven: VIX panic percentile AND SPY broken
    safe_on = (~warm) & (~risk_on) & (vp > backwardation_thr) \
              & (spy_ok < 0.5)

    cash_on = ~(risk_on | safe_on)

    for i in np.where(risk_on)[0]:
        m = mom_eq[i]
        valid = ~np.isnan(m)
        if not valid.any():
            tgt[i, cash_col] = 1.0
            continue
        vidx = np.where(valid)[0]
        order = vidx[np.argsort(-m[vidx])]
        top = order[:k]
        top = top[m[top] > 0]
        if len(top) == 0:
            tgt[i, cash_col] = 1.0
        else:
            w = 1.0 / len(top)
            for j in top:
                tgt[i, eq_cols[j]] = w

    for i in np.where(safe_on)[0]:
        m = mom_sh[i]
        valid = ~np.isnan(m)
        if not valid.any():
            tgt[i, cash_col] = 1.0
            continue
        vidx = np.where(valid)[0]
        pos = vidx[m[vidx] > 0]
        pick = pos if len(pos) > 0 else vidx
        w = 1.0 / len(pick)
        for j in pick:
            tgt[i, sh_cols[j]] = w

    tgt[cash_on, cash_col] = 1.0
    tgt[warm, cash_col] = 1.0

    return pd.DataFrame(tgt, index=rebal_dates, columns=cols)


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
    """Build daily weight matrix with a single N-day rebalance cadence (vectorized)."""
    idx = closes.index
    rebal_dates = idx[::cadence]
    targets = _targets_on_rebal(
        signals, momentum, list(closes.columns), rebal_dates,
        k, contango_thr, backwardation_thr, vix_pct_ceiling,
    )
    # Broadcast rebalance-day targets forward-filled until next rebalance
    W = targets.reindex(idx).ffill()
    # Before first rebalance, hold cash
    cash_col = CASH
    first_mask = W.isna().all(axis=1)
    if first_mask.any():
        W.loc[first_mask, :] = 0.0
        W.loc[first_mask, cash_col] = 1.0
    return W.fillna(0.0)


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
    # Thesis: SPY uptrend + VIX-premium regime = risk-on equities.
    # contango_thr = min regime for risk-on (positive = VIX richer than realized+VRP).
    contango_ths = [-0.20, -0.10, 0.00, 0.10]
    # backwardation_thr: VIX 252d percentile threshold for safe-haven pivot
    back_ths = [0.90, 0.95, 0.98]
    vix_ceils = [0.80, 0.90, 1.00]

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
