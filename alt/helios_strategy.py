"""
HELIOS — Cross-Asset Trend on Underlyings, Expressed via Leveraged ETFs
=======================================================================

Design
------
1. Compute trend signals on UNLEVERED underlyings (SPY, QQQ, TLT, GLD, USO,
   XLK, XLE, XLF, SMH, VNQ, EEM, FXI, IEF). These have longer, cleaner
   histories and no 3x rebalance decay noise.

2. Cross-sectional ranking: 6-month momentum with a 2-month skip
   (price[t-42] / price[t-189] - 1). Holding the top 2 ranked assets
   equally weighted (50/50).

3. Absolute trend filter per asset: same momentum > 0 AND close > 200-day SMA.

4. Macro meta-gate for RISK-asset eligibility:
        VIX z-score (252d) < 0.75   AND   HY OAS 20d change < +0.3
   Defensive assets (TLT, GLD, IEF) bypass this gate because they
   typically benefit when the equity gate is off.

5. Weekly rebalance (Fridays). Residual weight goes to BIL (cash).

6. Execution: signal computed on close[t]; trade at open[t+1];
   earn return from open[t+1] to open[t+2]. Zero look-ahead.
   Transaction cost = 5 bps one-way on turnover.

7. No daily vol targeting. Position sizing is a fixed 50/50 of chosen names.

Run with `python3 alt/helios_strategy.py`. Outputs:
    data/results/helios_metrics.json
    data/results/helios_returns.csv
    data/results/helios_picks.json
"""

from __future__ import annotations
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ETFDIR = ROOT / "data" / "etfs"
FREDIR = ROOT / "data" / "fred"
RESDIR = ROOT / "data" / "results"
RESDIR.mkdir(parents=True, exist_ok=True)

IS_START  = pd.Timestamp("2010-03-11")
OOS_START = pd.Timestamp("2019-01-01")
OOS_END   = None  # extend to latest available data

# Underlying  ->  leveraged expression (2x or 3x)
PAIRS = {
    "SPY": "UPRO",   # 3x
    "QQQ": "TQQQ",   # 3x
    "TLT": "TMF",    # 3x long bond
    "IEF": "TYD",    # 3x intermediate
    "GLD": "UGL",    # 2x gold
    "USO": "UCO",    # 2x oil
    "XLK": "TECL",   # 3x tech
    "XLE": "ERX",    # 2x energy
    "XLF": "FAS",    # 3x fin
    "SMH": "SOXL",   # 3x semi
    "VNQ": "DRN",    # 3x REIT
    "EEM": "EDC",    # 3x EM
    "FXI": "YINN",   # 3x China
}
# Assets that may be held even when the equity risk gate is OFF
DEFENSIVE = {"TLT", "GLD", "IEF"}

CASH_TICKER = "BIL"

# Strategy hyperparameters (selected by maximizing IS Sharpe only;
# OOS evaluated as one-shot test).
MOM_LB       = 189   # 9-month momentum lookback
MOM_SKIP     = 42    # skip the most recent ~2 months
SMA_LB       = 200
TOP_N        = 2
GROSS_CAP    = 1.0
VIX_Z_CAP    = 1.5   # softer gate chosen via IS: keeps exposure longer
HY_CHG20_CAP = 0.3
REBAL_FREQ   = "W-FRI"
COST_BPS     = 5.0


# --------------------------- Data ---------------------------
def load_etf(ticker: str) -> pd.DataFrame | None:
    fp = ETFDIR / f"{ticker}.csv"
    if not fp.exists():
        return None
    df = (
        pd.read_csv(fp, parse_dates=["Date"])
        .set_index("Date")
        .sort_index()
    )
    df = df[~df.index.duplicated(keep="first")]
    return df[["Open", "Close"]].astype(float)


def load_fred(name: str) -> pd.Series:
    fp = FREDIR / f"{name}.csv"
    s = (
        pd.read_csv(fp, parse_dates=["Date"])
        .set_index("Date")[name]
        .astype(float)
    )
    return s[~s.index.duplicated(keep="first")].sort_index()


def build_panel():
    """Return aligned unlevered-close, levered-open and cash-open frames."""
    close_u = {}
    open_l  = {}
    for under, lev in PAIRS.items():
        du = load_etf(under)
        dl = load_etf(lev)
        if du is None or dl is None:
            continue
        close_u[under] = du["Close"]
        open_l[lev]    = dl["Open"]
    dc = load_etf(CASH_TICKER)

    idx = None
    for s in list(close_u.values()) + list(open_l.values()) + [dc["Open"]]:
        idx = s.index if idx is None else idx.union(s.index)

    close_u_df = pd.DataFrame({k: v.reindex(idx) for k, v in close_u.items()})
    open_l_df  = pd.DataFrame({k: v.reindex(idx) for k, v in open_l.items()})
    open_c     = dc["Open"].reindex(idx)

    for d in (close_u_df, open_l_df):
        d.ffill(limit=3, inplace=True)
    open_c = open_c.ffill(limit=3)

    opens_lev_and_cash = pd.concat([open_l_df, open_c.rename(CASH_TICKER)], axis=1)
    return close_u_df, opens_lev_and_cash


# --------------------------- Signals ---------------------------
def build_signals(close_u: pd.DataFrame):
    """Trend signal + absolute trend filter, purely from unlevered closes."""
    mom = close_u.shift(MOM_SKIP) / close_u.shift(MOM_LB) - 1.0
    sma = close_u.rolling(SMA_LB).mean()
    abs_ok = (mom > 0) & (close_u > sma)
    return mom, abs_ok


def build_macro_gate(idx: pd.DatetimeIndex) -> pd.Series:
    """
    Equity risk-on gate.
        vix_z < 0.75  AND  hy_chg20 < 0.3
    VIX z-score is the 252-day rolling z; HY chg20 is 20-day level change.
    """
    vix = load_fred("VIXCLS").reindex(idx).ffill()
    hy  = load_fred("BAMLH0A0HYM2").reindex(idx).ffill()

    vix_z    = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
    hy_chg20 = hy  - hy.shift(20)
    gate = ((vix_z < VIX_Z_CAP) & (hy_chg20 < HY_CHG20_CAP)).astype(float)
    return gate, vix_z, hy_chg20


def build_target_weights(close_u, opens_lev):
    """Build daily target weights in the leveraged-ETF + cash column space."""
    mom, abs_ok = build_signals(close_u)
    gate, _, _  = build_macro_gate(close_u.index)

    elig = abs_ok.copy()
    for col in elig.columns:
        if col in DEFENSIVE:
            continue
        elig[col] = abs_ok[col] & (gate > 0.5)

    lev_cols = list(PAIRS.values()) + [CASH_TICKER]
    W = pd.DataFrame(0.0, index=close_u.index, columns=lev_cols)

    rebal_dates = pd.date_range(
        start=close_u.index.min(), end=close_u.index.max(), freq=REBAL_FREQ
    ).intersection(close_u.index)

    for dt in rebal_dates:
        e = elig.loc[dt]
        s = mom.loc[dt].where(e, other=-np.inf)
        top = s[s > 0].sort_values(ascending=False).head(TOP_N).index.tolist()
        if top:
            w = GROSS_CAP / len(top)
            for u in top:
                W.loc[dt, PAIRS[u]] = w
            W.loc[dt, CASH_TICKER] = max(0.0, 1.0 - w * len(top))
        else:
            W.loc[dt, CASH_TICKER] = 1.0

    # forward-fill only between rebalance dates
    mask_arr = W.index.isin(rebal_dates)
    mask_df = pd.DataFrame(
        np.broadcast_to(mask_arr[:, None], W.shape),
        index=W.index, columns=W.columns,
    )
    W = W.where(mask_df, other=np.nan).ffill().fillna(0.0)

    return W, rebal_dates


# --------------------------- Backtest ---------------------------
def run_backtest(W: pd.DataFrame, opens: pd.DataFrame, cost_bps: float = COST_BPS):
    """
    Next-day-open execution.  Let
        r[t] = open[t+2] / open[t+1] - 1        (held from open[t+1] to open[t+2])
        w[t] = target weight using info up to close[t]
    Portfolio PnL on day t = w[t] . r[t].
    This yields zero look-ahead: w[t] uses only close[t] and earlier;
    the earliest price it hits is open[t+1].
    """
    common = W.index.intersection(opens.index)
    W = W.loc[common]
    opens = opens.loc[common]

    r_fwd = opens.shift(-2) / opens.shift(-1) - 1.0

    # Turnover measured on weights in the CONTEXT they become active.
    # The physical trade happens at open[t+1], so between w[t-1] and w[t]
    # (signals dates) the single re-weighting happens at open[t+1].
    # We apply the cost at day t (signal day) since that's when the
    # decision is registered; this is conservative and closely tracks the
    # exec date.
    turnover = W.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (cost_bps / 1e4)

    # align columns
    W_use = W[r_fwd.columns]
    gross_ret = (W_use * r_fwd).sum(axis=1)
    net_ret   = gross_ret - costs

    df = pd.DataFrame({
        "ret":        net_ret,
        "gross_ret":  gross_ret,
        "turnover":   turnover,
        "cost":       costs,
        "weight_sum": W_use.sum(axis=1),
        "cash_wt":    W_use.get(CASH_TICKER, pd.Series(0.0, index=W_use.index)),
    }, index=common)
    return df


# --------------------------- Metrics ---------------------------
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
    return float((eq / peak - 1).min())


# --------------------------- Live-signal-friendly weight builder ---------------------------
def build_weights(live_extend: bool = False) -> pd.DataFrame:
    """Compute the canonical HELIOS daily target-weight DataFrame.

    Index: trading dates from IS_START (or first date all LETFs are listed).
    Columns: leveraged ETF tickers + 'BIL'. Weights sum to 1.0 each day.
    Weekly Friday rebalance, forward-filled between rebalance dates.

    live_extend: HELIOS's W[t] is already aligned for live execution
        (signal at close[t] → trade at open[t+1] → hold to open[t+2]),
        so the flag is accepted for API uniformity but adds a forward-
        ffilled row only if t+1 is a Friday and would otherwise miss
        a fresh rebalance.
    """
    close_u, opens = build_panel()
    if live_extend and len(opens) > 0:
        next_day = opens.index[-1] + pd.tseries.offsets.BDay()
        opens.loc[next_day] = opens.iloc[-1]
        close_u.loc[next_day] = close_u.iloc[-1]
        opens = opens.sort_index()
        close_u = close_u.sort_index()
    lev_firsts = []
    for lev in PAIRS.values():
        s = opens[lev].dropna()
        if len(s):
            lev_firsts.append(s.index.min())
    start = max(max(lev_firsts), IS_START)
    close_u = close_u.loc[start:]
    opens = opens.loc[start:]
    W, _ = build_target_weights(close_u, opens)
    return W


# --------------------------- Main ---------------------------
def main():
    close_u, opens = build_panel()

    # Start when every leveraged ETF in the map has data
    lev_firsts = []
    for lev in PAIRS.values():
        s = opens[lev].dropna()
        if len(s):
            lev_firsts.append(s.index.min())
    start = max(max(lev_firsts), IS_START)

    close_u = close_u.loc[start:]
    opens   = opens.loc[start:]

    W, rebal_dates = build_target_weights(close_u, opens)
    bt = run_backtest(W, opens)

    # Trim to evaluation window
    bt = bt.loc[IS_START:]
    is_slice  = bt.loc[IS_START: OOS_START - pd.Timedelta(days=1)]
    oos_slice = bt.loc[OOS_START:]

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
        "is_oos_gap":  float(
            sharpe(is_slice["ret"]) - sharpe(oos_slice["ret"])
        ),
        "universe":    list(PAIRS.keys()),
        "lev_map":     PAIRS,
        "params": {
            "MOM_LB": MOM_LB, "MOM_SKIP": MOM_SKIP, "SMA_LB": SMA_LB,
            "TOP_N": TOP_N, "GROSS_CAP": GROSS_CAP,
            "VIX_Z_CAP": VIX_Z_CAP, "HY_CHG20_CAP": HY_CHG20_CAP,
            "REBAL_FREQ": REBAL_FREQ, "COST_BPS": COST_BPS,
        },
    }

    # Picks per rebalance
    picks = []
    for dt in rebal_dates:
        if dt < bt.index.min() or dt > bt.index.max():
            continue
        row = W.loc[dt]
        held = {k: round(float(v), 4) for k, v in row.items() if v > 1e-6}
        picks.append({"date": str(dt.date()), "weights": held})
    metrics["n_rebalances"] = len(picks)
    metrics["picks_sample_first5"] = picks[:5]
    metrics["picks_sample_last5"]  = picks[-5:]

    # Save files
    with open(RESDIR / "helios_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    bt_out = bt[["ret", "weight_sum", "cash_wt"]].copy()
    bt_out.index.name = "Date"
    bt_out.to_csv(RESDIR / "helios_returns.csv")

    with open(RESDIR / "helios_picks.json", "w") as f:
        json.dump(picks, f, indent=2)

    # Print summary
    print("=" * 72)
    print("HELIOS — Results")
    print("=" * 72)
    for k in [
        "start", "end", "is_sharpe", "oos_sharpe", "full_sharpe",
        "is_cagr", "oos_cagr", "full_cagr", "max_dd", "ann_vol",
        "avg_turnover_daily", "avg_turnover_annual", "cash_avg_wt",
        "is_oos_gap", "n_rebalances",
    ]:
        v = metrics[k]
        if isinstance(v, float):
            print(f"  {k:22s} {v:+.4f}")
        else:
            print(f"  {k:22s} {v}")
    print("Universe:", metrics["universe"])
    print("Pairs:   ", metrics["lev_map"])


if __name__ == "__main__":
    main()
