"""
HELIOS — Cross-Asset Trend on Underlyings, Expressed via Leveraged ETFs

Design overview
---------------
1. Signal layer: run momentum / trend signals on UNLEVERED underlyings
   (SPY, QQQ, IEF, TLT, GLD, USO, XLK, XLE, XLF, SMH, VNQ, EEM, FXI, IWM, XLV).
   These series are long, clean, and free of the daily-rebalance decay
   that makes 3x ETF series noisy.

2. Macro meta-gate: a global risk switch derived from VIX regime,
   HY credit spread direction (BAMLH0A0HYM2), and term spread (T10Y2Y).
   When the gate is off, we scale down / go to cash (BIL).

3. Expression: once an underlying passes the trend + macro gate,
   we take the exposure in its matched leveraged ETF (UPRO, TQQQ, TMF,
   UGL, UCO, TECL, ERX, FAS, SOXL, DRN, EDC, YINN).

4. Sizing: equal-weight across selected names, capped total gross,
   remainder in BIL (cash). NO daily vol targeting.

5. Execution: signal computed on data through close[t-1] (strictly lagged);
   weights applied to ret = open[t+1]/open[t] - 1, so zero look-ahead.
   Transaction cost = 5 bps one-way on turnover.

Run: `python3 alt/helios_strategy.py`
"""

from __future__ import annotations
import os
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

ROOT   = Path("/home/user/bonds")
ETFDIR = ROOT / "data" / "etfs"
FREDIR = ROOT / "data" / "fred"
RESDIR = ROOT / "data" / "results"
RESDIR.mkdir(parents=True, exist_ok=True)

IS_START  = pd.Timestamp("2010-03-11")
OOS_START = pd.Timestamp("2019-01-01")
OOS_END   = pd.Timestamp("2026-04-02")

# Underlying -> leveraged expression
PAIRS = {
    "SPY": "UPRO",   # 3x
    "QQQ": "TQQQ",   # 3x
    "IEF": "TYD",    # 3x treasuries (middle)
    "TLT": "TMF",    # 3x long bond
    "GLD": "UGL",    # 2x gold
    "USO": "UCO",    # 2x oil
    "XLK": "TECL",   # 3x tech
    "XLE": "ERX",    # 2x energy
    "XLF": "FAS",    # 3x fin
    "SMH": "SOXL",   # 3x semi
    "VNQ": "DRN",    # 3x REIT
    "EEM": "EDC",    # 3x EM
    "FXI": "YINN",   # 3x China
    "IWM": "UWM",    # 2x small cap (may or may not exist)
    "XLV": "CURE",   # 3x healthcare (may or may not exist)
}

CASH_TICKER = "BIL"


# -------------------- Data loading --------------------
def load_etf(ticker: str) -> pd.DataFrame | None:
    fp = ETFDIR / f"{ticker}.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["Open", "Close"]].astype(float)


def load_fred(name: str) -> pd.Series:
    fp = FREDIR / f"{name}.csv"
    s = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date")[name].astype(float)
    return s[~s.index.duplicated(keep="first")].sort_index()


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Build aligned Open/Close panels for underlyings and leveraged ETFs."""
    under_cols = {}
    lev_cols   = {}
    pair_map   = {}
    for under, lev in PAIRS.items():
        du = load_etf(under)
        dl = load_etf(lev)
        if du is None or dl is None:
            continue
        under_cols[under] = du
        lev_cols[lev]     = dl
        pair_map[under]   = lev
    # cash
    dc = load_etf(CASH_TICKER)

    # Intersect index on both underlying and leveraged
    idx = None
    for d in list(under_cols.values()) + list(lev_cols.values()) + [dc]:
        idx = d.index if idx is None else idx.union(d.index)

    # Build opens / closes frames (underlyings drive signals)
    open_u  = pd.DataFrame({k: v["Open"]  for k, v in under_cols.items()}).reindex(idx)
    close_u = pd.DataFrame({k: v["Close"] for k, v in under_cols.items()}).reindex(idx)
    open_l  = pd.DataFrame({k: v["Open"]  for k, v in lev_cols.items()}).reindex(idx)
    close_l = pd.DataFrame({k: v["Close"] for k, v in lev_cols.items()}).reindex(idx)
    open_c  = dc["Open"].reindex(idx)

    # Forward fill short gaps up to 3 days
    for df in (open_u, close_u, open_l, close_l):
        df.ffill(limit=3, inplace=True)
    open_c = open_c.ffill(limit=3)

    # Combined frames
    opens  = pd.concat([open_l, open_c.rename(CASH_TICKER)],  axis=1)
    closes = close_l.copy()

    return open_u, close_u, opens, closes, pair_map


# -------------------- Signals --------------------
def trend_signals(close_u: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Several trend / momentum signals on UNLEVERED underlyings.

    Because we shift by 1 bar before trading at open[t+1] of day t+1
    (see backtest), the signals themselves use close through t.
    We will additionally shift by 1 when assembling the trading frame.
    """
    r = np.log(close_u).diff()

    # Long-term 12m momentum with 1m skip (approx 252/21 trading days)
    mom_12_1 = close_u.shift(21) / close_u.shift(252) - 1.0

    # 6 month momentum
    mom_6 = close_u / close_u.shift(126) - 1.0

    # 3 month momentum
    mom_3 = close_u / close_u.shift(63) - 1.0

    # 1 month momentum (for trend acceleration)
    mom_1 = close_u / close_u.shift(21) - 1.0

    # Price > 200d SMA
    sma200 = close_u.rolling(200).mean()
    above_200 = (close_u > sma200).astype(float)

    # Price > 50d SMA
    sma50 = close_u.rolling(50).mean()
    above_50 = (close_u > sma50).astype(float)

    return dict(
        mom_12_1=mom_12_1, mom_6=mom_6, mom_3=mom_3, mom_1=mom_1,
        above_200=above_200, above_50=above_50,
    )


def build_macro_gate(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Macro meta-gate. Strictly past data (rolling windows / lagged diffs).

    Main equity risk-on gate (validated IS and OOS on SPY):
        VIX z-score over 252d < 0.5  AND  HY OAS 20d change < 0.2
    When this gate is ON, equity-like assets are eligible.
    When OFF, we are either in cash or in defensive (bonds/gold),
    which has its OWN trend gate on its OWN underlying.
    """
    vix  = load_fred("VIXCLS").reindex(idx).ffill()
    hy   = load_fred("BAMLH0A0HYM2").reindex(idx).ffill()

    vix_mean = vix.rolling(252).mean()
    vix_std  = vix.rolling(252).std()
    vix_z    = (vix - vix_mean) / vix_std

    hy_chg20 = hy - hy.shift(20)

    equity_on   = (vix_z < 0.5) & (hy_chg20 < 0.2)
    equity_weak = (vix_z < 1.0) & (hy_chg20 < 0.5)   # softer gate for partial exposure

    return pd.DataFrame({
        "vix_z":       vix_z,
        "hy_chg20":    hy_chg20,
        "equity_on":   equity_on.astype(float),
        "equity_weak": equity_weak.astype(float),
    }, index=idx)


# -------------------- Target weights --------------------
def build_target_weights(
    close_u: pd.DataFrame,
    pair_map: dict,
    open_c_index: pd.DatetimeIndex,
    n_top: int = 4,
    gross_cap: float = 1.0,
    gross_cap_weak: float = 0.5,
    rebal_freq: str = "W-FRI",
) -> pd.DataFrame:
    """
    Produce a target weight per leveraged-ETF column for each rebalance date.
    Then forward-fill to daily frame. The daily frame must be shifted by
    one day when used (handled in run_backtest).
    """
    sig = trend_signals(close_u)
    macro = build_macro_gate(close_u.index)

    # Composite trend score (z across assets on each day)
    # Score = average zscore of (mom_12_1, mom_6, mom_3) + 0.5 * above_200
    comp = pd.DataFrame(0.0, index=close_u.index, columns=close_u.columns)
    for k, w in [("mom_12_1", 1.0), ("mom_6", 1.0), ("mom_3", 0.6)]:
        m = sig[k]
        z = m.sub(m.mean(axis=1), axis=0).div(m.std(axis=1).replace(0, np.nan), axis=0)
        comp = comp + w * z.fillna(0.0)
    comp = comp + 0.5 * sig["above_200"].fillna(0.0) + 0.25 * sig["above_50"].fillna(0.0)

    # Absolute trend filter: asset must have own mom_6 > 0 AND above 200 SMA
    abs_ok = (sig["mom_6"] > 0.0) & (sig["above_200"] > 0.5)

    # Defensive assets have their OWN trend gate; equity-like assets
    # additionally require equity_weak gate to be on.
    DEFENSIVE = {"IEF", "TLT", "GLD"}

    equity_on   = macro["equity_on"]
    equity_weak = macro["equity_weak"]

    # Eligibility: defensive need only abs trend; risk assets need
    # abs trend + equity_weak (looser version of gate).
    eligible = abs_ok.copy()
    for col in eligible.columns:
        if col in DEFENSIVE:
            eligible[col] = abs_ok[col]
        else:
            eligible[col] = abs_ok[col] & (equity_weak > 0.5)

    # Rebalance calendar (weekly Fridays)
    rebal_dates = pd.date_range(
        start=close_u.index.min(), end=close_u.index.max(), freq=rebal_freq
    )
    rebal_dates = rebal_dates.intersection(close_u.index)

    lev_cols = list(pair_map.values()) + [CASH_TICKER]
    W = pd.DataFrame(0.0, index=close_u.index, columns=lev_cols)

    for dt in rebal_dates:
        elig_row = eligible.loc[dt]
        score_row = comp.loc[dt].where(elig_row, other=-np.inf)
        # Select top N with positive composite
        valid = score_row[score_row > 0].sort_values(ascending=False)
        chosen_under = valid.head(n_top).index.tolist()

        # Gross cap depends on macro gate strength
        is_strong = equity_on.loc[dt] > 0.5 if dt in equity_on.index else False
        gcap = gross_cap if is_strong else gross_cap_weak

        if len(chosen_under) == 0:
            W.loc[dt, CASH_TICKER] = 1.0
        else:
            w = gcap / len(chosen_under)
            for u in chosen_under:
                lev = pair_map[u]
                W.loc[dt, lev] = w
            invested = W.loc[dt, [pair_map[u] for u in chosen_under]].sum()
            W.loc[dt, CASH_TICKER] = max(0.0, 1.0 - invested)

    # Forward-fill between rebalances, only propagate on rebalance rows
    mask_arr = W.index.isin(rebal_dates)
    mask_df = pd.DataFrame(
        np.broadcast_to(mask_arr[:, None], W.shape),
        index=W.index, columns=W.columns,
    )
    W_rb = W.where(mask_df, other=np.nan)
    W_daily = W_rb.ffill().fillna(0.0)

    # Ensure we have a fully-cash starting period
    W_daily.loc[:rebal_dates[0]] = 0.0
    W_daily.loc[:rebal_dates[0], CASH_TICKER] = 1.0

    return W_daily


# -------------------- Backtest --------------------
def run_backtest(
    weights_daily: pd.DataFrame,
    opens: pd.DataFrame,   # opens of leveraged ETFs + BIL
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Execute at next day's open. Weight at close of day t is applied to
    the return from open[t+1] to open[t+2]. We enforce this by shifting
    weights by 1 day before computing returns.

    Asset return used: r_t = open[t] / open[t-1] - 1 for each ticker.
    Effective portfolio return at day t uses w = weights.shift(1).loc[t-1],
    which we implement simply by shifting once (weights enter effect on next day).
    """
    # Align
    common = weights_daily.index.intersection(opens.index)
    W = weights_daily.loc[common]
    O = opens.loc[common]

    # Simple open-to-open returns (for each ticker)
    r = O.pct_change().fillna(0.0)

    # The weight we want active on day t is the weight we DECIDED using data
    # through close[t-1]. Our rebalance rows are labeled with date t; weights
    # at date t in W are taken from signals-through-close[t]. That's t's
    # decision. To apply them to the return from open[t+1] to open[t+2],
    # shift by 2: W.shift(2) applies at day t+2 with return open[t+1]->open[t+2]
    # But r_t = O[t]/O[t-1]-1 = return from open[t-1] to open[t].
    # We want w decided at close[t] to match r from open[t+1] to open[t+2],
    # i.e., apply w_t to r_{t+2}. With W.shift(2), at row t+2 we have w_t. Good.
    W_eff = W.shift(2).fillna(0.0)
    W_eff = W_eff[r.columns]  # align columns

    port_r = (W_eff * r).sum(axis=1)

    # Turnover: per-day sum of absolute weight changes in effective weights
    turnover = W_eff.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (cost_bps / 1e4)
    port_r_net = port_r - costs

    df = pd.DataFrame({
        "ret":        port_r_net,
        "gross_ret":  port_r,
        "turnover":   turnover,
        "cost":       costs,
        "weight_sum": W_eff.sum(axis=1),
        "cash_wt":    W_eff.get(CASH_TICKER, pd.Series(0.0, index=W_eff.index)),
    }, index=common)
    return df


# -------------------- Metrics --------------------
def sharpe(r: pd.Series, freq: int = 252) -> float:
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * math.sqrt(freq))


def cagr(r: pd.Series, freq: int = 252) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    eq = (1 + r).prod()
    yrs = len(r) / freq
    return float(eq ** (1 / yrs) - 1) if yrs > 0 else float("nan")


def max_dd(r: pd.Series) -> float:
    eq = (1 + r.fillna(0.0)).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1).min()
    return float(dd)


# -------------------- Main --------------------
def main():
    open_u, close_u, opens, closes, pair_map = build_panel()

    # Start date: all required leveraged ETFs in PAIRS must be live.
    # Take latest first available date among active lev ETFs that we actually use.
    lev_tickers = [pair_map[u] for u in pair_map]
    lev_firsts = []
    for lev in lev_tickers:
        s = opens[lev].dropna()
        if len(s):
            lev_firsts.append(s.index.min())
    start = max(lev_firsts) if lev_firsts else close_u.index.min()
    start = max(start, IS_START)

    close_u = close_u.loc[start:]
    open_u  = open_u.loc[start:]
    opens   = opens.loc[start:]
    closes  = closes.loc[start:]

    W_daily = build_target_weights(
        close_u, pair_map, opens.index,
        n_top=4, gross_cap=1.0, rebal_freq="W-FRI",
    )

    bt = run_backtest(W_daily, opens, cost_bps=5.0)

    # Split IS / OOS
    bt = bt.loc[IS_START:OOS_END]
    is_slice  = bt.loc[IS_START: OOS_START - pd.Timedelta(days=1)]
    oos_slice = bt.loc[OOS_START: OOS_END]

    metrics = {
        "start":       str(bt.index.min().date()),
        "end":         str(bt.index.max().date()),
        "is_sharpe":   sharpe(is_slice["ret"]),
        "oos_sharpe":  sharpe(oos_slice["ret"]),
        "full_sharpe": sharpe(bt["ret"]),
        "is_cagr":     cagr(is_slice["ret"]),
        "oos_cagr":    cagr(oos_slice["ret"]),
        "full_cagr":   cagr(bt["ret"]),
        "max_dd":      max_dd(bt["ret"]),
        "ann_vol":     float(bt["ret"].std() * math.sqrt(252)),
        "avg_turnover_daily":  float(bt["turnover"].mean()),
        "avg_turnover_annual": float(bt["turnover"].sum() / (len(bt) / 252)),
        "cash_avg_wt": float(bt["cash_wt"].mean()),
        "is_oos_gap":  float(sharpe(is_slice["ret"]) - sharpe(oos_slice["ret"])),
        "universe":    list(pair_map.keys()),
        "lev_map":     pair_map,
    }

    # Picks per rebalance
    picks = []
    rebal_mask = (W_daily.diff().abs().sum(axis=1) > 1e-8)
    rebal_days = W_daily.index[rebal_mask]
    for dt in rebal_days:
        row = W_daily.loc[dt]
        held = row[row > 1e-6].to_dict()
        picks.append({"date": str(dt.date()), "weights": {k: round(float(v), 4) for k, v in held.items()}})
    metrics["n_rebalances"] = len(picks)
    metrics["picks_sample_first5"] = picks[:5]
    metrics["picks_sample_last5"]  = picks[-5:]

    # Save
    with open(RESDIR / "helios_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    bt_out = bt[["ret", "weight_sum", "cash_wt"]].copy()
    bt_out.index.name = "Date"
    bt_out.to_csv(RESDIR / "helios_returns.csv")

    # Save full picks for audit
    with open(RESDIR / "helios_picks.json", "w") as f:
        json.dump(picks, f, indent=2)

    # Print
    print("=" * 64)
    print("HELIOS — Results")
    print("=" * 64)
    for k in ["start","end","is_sharpe","oos_sharpe","full_sharpe",
              "is_cagr","oos_cagr","full_cagr","max_dd","ann_vol",
              "avg_turnover_daily","avg_turnover_annual","cash_avg_wt",
              "is_oos_gap","n_rebalances"]:
        v = metrics[k]
        if isinstance(v, float):
            print(f"  {k:22s} {v:+.4f}")
        else:
            print(f"  {k:22s} {v}")
    print("Universe:", metrics["universe"])
    print("Pairs:   ", metrics["lev_map"])


if __name__ == "__main__":
    main()
