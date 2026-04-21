"""
KRAKEN — Breadth-Gated Defensive Concentrator
==============================================

Core idea: stay out during bad regimes, concentrate heavily in leveraged ETFs
during good regimes. The edge is time-series participation: cut down-capture
dramatically while keeping up-capture.

Architecture:
  1. BREADTH: fraction of SPDR sector ETFs above 200d MA.
  2. MACRO:  HY credit spread below 60d EMA AND VIX below threshold.
  3. STATE = STRONG_BULL if breadth >= B_HIGH AND macro OK, else CASH (BIL).
  4. In STRONG_BULL: long top-K leveraged ETFs by 12-1 momentum from a
     broad universe of 18 (2x/3x) ETFs.
  5. Monthly rebalance + interim exit to cash if state flips mid-month.
  6. Anti-whipsaw: require CONFIRM consecutive days of state change.

Execution conventions:
  - Signal computed from data up to close[t-1].
  - Trade at open[t]; daily PnL = open[t] -> open[t+1] return.
  - 10 bps one-way transaction cost on |weight change|.
  - No daily vol scaling.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

# -------------------- paths --------------------
REPO   = "/home/user/bonds"
ETF_D  = os.path.join(REPO, "data", "etfs")
FRED_D = os.path.join(REPO, "data", "fred")
RES_D  = os.path.join(REPO, "data", "results")
os.makedirs(RES_D, exist_ok=True)

# -------------------- config --------------------
LEV_UNIVERSE = [
    # Equity-index bulls
    "TQQQ", "UPRO", "QLD", "SSO",
    # Sector bulls
    "SOXL", "TECL", "FAS", "LABU", "ERX", "DRN",
    # Global equity bulls
    "EDC", "YINN",
    # Commodities / rates bulls (diversifiers — can help when equity momentum is bad)
    "NUGT", "UGL", "UCO", "TMF", "UBT", "TYD",
]
SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]

# regime params
BREADTH_MA      = 200
BREADTH_HIGH    = 0.60      # fraction of sectors above 200d MA needed to be bullish
HY_EMA_WIN      = 60        # EMA window on HY OAS
VIX_MAX         = 25.0      # absolute VIX ceiling
CONFIRM_DAYS    = 3         # days the state must persist before flipping

# momentum params — use shorter momentum on leveraged ETFs (12-1 whipsaws after bears)
MOM_LB          = 63        # ~3 months
MOM_SKIP        = 5         # skip last week
TOP_K           = 2
MOM_MIN_TRACK   = MOM_LB + MOM_SKIP

# rebalance
REBAL_DAYS      = 21        # monthly-ish
TC_BPS_ONEWAY   = 10.0      # 10 bps one-way (above 5 bps requirement)

# windows
START_FULL = pd.Timestamp("2010-03-11")
IS_END     = pd.Timestamp("2018-12-31")
OOS_START  = pd.Timestamp("2019-01-01")
OOS_END    = pd.Timestamp("2026-04-02")


# -------------------- data --------------------
def load_etf(sym: str) -> pd.DataFrame:
    p = os.path.join(ETF_D, f"{sym}.csv")
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[["Open", "Close"]].astype(float)


def load_fred(name: str) -> pd.Series:
    p = os.path.join(FRED_D, f"{name}.csv")
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[name].astype(float)


def build_panel():
    opens = {}
    closes = {}
    for s in LEV_UNIVERSE + SECTOR_ETFS + ["BIL", "SPY"]:
        df = load_etf(s)
        opens[s]  = df["Open"]
        closes[s] = df["Close"]
    opens  = pd.DataFrame(opens).sort_index()
    closes = pd.DataFrame(closes).sort_index()
    # use SPY as trading calendar
    cal = closes["SPY"].dropna().index
    opens  = opens.reindex(cal)
    closes = closes.reindex(cal)
    vix    = load_fred("VIXCLS").reindex(cal).ffill()
    hy     = load_fred("BAMLH0A0HYM2").reindex(cal).ffill()
    return opens, closes, vix, hy


# -------------------- regime signals --------------------
def breadth_signal(closes: pd.DataFrame) -> pd.Series:
    cols = [c for c in SECTOR_ETFS if c in closes.columns]
    sub = closes[cols]
    ma  = sub.rolling(BREADTH_MA, min_periods=BREADTH_MA).mean()
    above = (sub > ma).astype(float)
    # fraction only counts sectors that have full history
    valid = sub.notna() & ma.notna()
    num   = above.where(valid, 0.0).sum(axis=1)
    den   = valid.sum(axis=1).replace(0, np.nan)
    return (num / den).rename("breadth")


def macro_signal(vix: pd.Series, hy: pd.Series) -> pd.Series:
    hy_ema = hy.ewm(span=HY_EMA_WIN, adjust=False, min_periods=HY_EMA_WIN).mean()
    hy_ok  = (hy < hy_ema).astype(float)          # credit compressing
    vix_ok = (vix < VIX_MAX).astype(float)         # not in panic
    return (hy_ok * vix_ok).rename("macro_ok")


def regime_state(breadth: pd.Series, macro_ok: pd.Series) -> pd.Series:
    raw = ((breadth >= BREADTH_HIGH) & (macro_ok == 1.0)).astype(int)
    # anti-whipsaw: require CONFIRM_DAYS consecutive days of the new state
    if CONFIRM_DAYS <= 1:
        return raw.rename("state_raw")
    confirmed = raw.copy()
    state = 0
    last_persist = 0
    for i in range(len(raw)):
        r = int(raw.iloc[i])
        if r == state:
            last_persist = 0
        else:
            last_persist += 1
            if last_persist >= CONFIRM_DAYS:
                state = r
                last_persist = 0
        confirmed.iloc[i] = state
    return confirmed.rename("state")


# -------------------- momentum signal --------------------
def momentum_12_1(closes: pd.DataFrame) -> pd.DataFrame:
    # 12-1: ratio of close[t-MOM_SKIP] to close[t-MOM_SKIP-MOM_LB], computed
    # entirely from data known at close[t]. We will additionally .shift(1) when
    # applying so signal at time t uses only data through close[t-1].
    num = closes.shift(MOM_SKIP)
    den = closes.shift(MOM_SKIP + MOM_LB)
    mom = (num / den) - 1.0
    return mom


# -------------------- backtest engine --------------------
@dataclass
class BacktestOut:
    daily_ret: pd.Series
    weights:   pd.DataFrame
    turnover:  pd.Series
    state:     pd.Series
    nav:       pd.Series


def run_backtest(opens: pd.DataFrame, closes: pd.DataFrame,
                 state: pd.Series, mom: pd.DataFrame) -> BacktestOut:
    # Trading calendar = intersection where SPY has open+close
    cal = opens.index.intersection(closes.index)
    opens  = opens.loc[cal]
    closes = closes.loc[cal]

    # open-to-open daily return for each asset
    open_ret = opens.pct_change().shift(-1)  # ret from open[t] to open[t+1], attributed to t
    # actually we want: hold weight from open[t] to open[t+1], earn (open[t+1]/open[t] - 1)
    # So at bar t, pnl = w[t] * (open[t+1]/open[t] - 1)
    # open_ret = opens.shift(-1) / opens - 1  is same thing
    open_ret = (opens.shift(-1) / opens) - 1.0
    open_ret = open_ret.fillna(0.0)

    # Align state and momentum: both already reflect data available at close[t-1]
    # (momentum uses shift MOM_SKIP; we shift state by 1 bar to be safe).
    state_lag = state.shift(1).fillna(0).astype(int)
    mom_lag   = mom.shift(1)

    universe = [c for c in LEV_UNIVERSE if c in closes.columns]
    bil_col  = "BIL" if "BIL" in closes.columns else None
    assert bil_col is not None

    weights = pd.DataFrame(0.0, index=cal, columns=list(closes.columns))
    # rebalance days: first trading day at index 0, then every REBAL_DAYS
    rebal_idx = set(range(0, len(cal), REBAL_DAYS))
    last_w = pd.Series(0.0, index=weights.columns)
    # ensure first bar has start-of-period weight set once we have enough history

    daily_ret = pd.Series(0.0, index=cal)
    turnover  = pd.Series(0.0, index=cal)

    # we need to have weights in place at open[t]; so we use state_lag[t] and
    # mom_lag[t] which reflect info through close[t-1].
    for i, dt in enumerate(cal):
        bull = bool(state_lag.iloc[i])
        # Decide target weight
        rebal_today = (i in rebal_idx)
        # Interim exit: if regime flipped off, exit to cash immediately
        was_bull = (last_w[universe].sum() > 0.0)
        if was_bull and not bull:
            target = pd.Series(0.0, index=weights.columns)
            target[bil_col] = 1.0
            rebal_today = True
        elif (not was_bull) and bull:
            # regime just turned on — enter on the next scheduled rebal OR
            # immediately if it's been off a while (rebal_today already True if modulo hits)
            # Better: enter immediately for maximum participation
            rebal_today = True

        if rebal_today:
            if bull:
                # pick top-K by momentum among assets with valid mom and a price today
                m = mom_lag.iloc[i].reindex(universe)
                # require prices and momentum valid
                valid = m.notna() & closes.iloc[i].reindex(universe).notna()
                m = m[valid]
                if len(m) >= TOP_K:
                    # only positive momentum names (12-1 positive)
                    m_pos = m[m > 0].sort_values(ascending=False)
                    picks = m_pos.head(TOP_K).index.tolist()
                    if len(picks) == 0:
                        target = pd.Series(0.0, index=weights.columns)
                        target[bil_col] = 1.0
                    else:
                        target = pd.Series(0.0, index=weights.columns)
                        w_each = 1.0 / len(picks)   # equal weight across picks
                        for p in picks:
                            target[p] = w_each
                else:
                    target = pd.Series(0.0, index=weights.columns)
                    target[bil_col] = 1.0
            else:
                target = pd.Series(0.0, index=weights.columns)
                target[bil_col] = 1.0
            new_w = target
        else:
            new_w = last_w.copy()

        # turnover & TC
        delta = (new_w - last_w).abs().sum()
        tc = delta * (TC_BPS_ONEWAY / 1e4)
        turnover.iloc[i] = delta

        # realize PnL from open[t] to open[t+1]
        r_day = float((new_w * open_ret.iloc[i]).sum()) - tc
        daily_ret.iloc[i] = r_day
        weights.iloc[i] = new_w.values
        last_w = new_w

    nav = (1.0 + daily_ret).cumprod()
    return BacktestOut(daily_ret=daily_ret, weights=weights,
                       turnover=turnover, state=state_lag, nav=nav)


# -------------------- metrics --------------------
def metrics(ret: pd.Series, label: str = "") -> dict:
    r = ret.dropna()
    if len(r) == 0:
        return {"label": label, "n": 0}
    ann = 252
    mu  = r.mean() * ann
    sd  = r.std(ddof=0) * np.sqrt(ann)
    sr  = mu / sd if sd > 0 else float("nan")
    nav = (1 + r).cumprod()
    n_years = len(r) / ann
    cagr = nav.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0.0
    dd = nav / nav.cummax() - 1.0
    mdd = dd.min()
    return {
        "label": label,
        "n": int(len(r)),
        "start": str(r.index[0].date()),
        "end":   str(r.index[-1].date()),
        "sharpe": float(sr),
        "cagr":   float(cagr),
        "ann_vol": float(sd),
        "mdd":    float(mdd),
        "navx":   float(nav.iloc[-1]),
    }


def print_block(m: dict):
    print(f"  {m['label']:<14}  "
          f"SR={m['sharpe']:+.2f}  CAGR={m['cagr']*100:+6.2f}%  "
          f"Vol={m['ann_vol']*100:5.2f}%  MDD={m['mdd']*100:+6.2f}%  "
          f"NAVx={m['navx']:.2f}  n={m['n']} [{m['start']}..{m['end']}]")


# -------------------- main --------------------
def main():
    print("=" * 84)
    print(" KRAKEN — Breadth-Gated Defensive Concentrator")
    print("=" * 84)

    print("Loading data...")
    opens, closes, vix, hy = build_panel()

    # Clip to strategy full window
    mask = (opens.index >= START_FULL) & (opens.index <= OOS_END)
    opens  = opens.loc[mask]
    closes = closes.loc[mask]
    vix    = vix.loc[mask]
    hy     = hy.loc[mask]
    print(f"  calendar: {opens.index.min().date()} .. {opens.index.max().date()} "
          f"({len(opens)} bars)")

    print("Computing breadth (sector-above-MA) and macro signals...")
    breadth = breadth_signal(closes)
    macro_ok = macro_signal(vix, hy)
    state = regime_state(breadth, macro_ok)
    print(f"  breadth mean={breadth.mean():.2f}  state_on_frac={state.mean():.2%}")

    print("Computing 12-1 momentum on leveraged universe...")
    mom = momentum_12_1(closes[[c for c in LEV_UNIVERSE if c in closes.columns]])

    print("Running backtest...")
    out = run_backtest(opens, closes, state, mom)

    r = out.daily_ret
    # split IS/OOS
    is_mask  = (r.index >= START_FULL) & (r.index <= IS_END)
    oos_mask = (r.index >= OOS_START) & (r.index <= OOS_END)

    M_full = metrics(r, "FULL")
    M_is   = metrics(r[is_mask], "IS")
    M_oos  = metrics(r[oos_mask], "OOS")

    print("\nResults:")
    for M in (M_full, M_is, M_oos):
        print_block(M)

    # time in market
    invested = (out.weights.drop(columns=["BIL", "SPY"], errors="ignore").sum(axis=1) > 0.0)
    tim = invested.mean()
    print(f"\n  time-in-market (any leveraged exposure): {tim:.2%}")
    print(f"  avg turnover per bar: {out.turnover.mean():.4f}  "
          f"(annualized: {out.turnover.mean()*252:.2f})")

    state_counts = {
        "state_on":  int(out.state.sum()),
        "state_off": int((out.state == 0).sum()),
        "total":     int(len(out.state)),
    }
    print(f"  state counts: {state_counts}")

    # gap check
    gap = abs(M_is["sharpe"] - M_oos["sharpe"])
    print(f"  |IS Sharpe - OOS Sharpe| = {gap:.2f}  (want <= 0.5)")

    # ---- pass/fail vs hard requirements ----
    passed = True
    checks = []
    for name, cond in [
        ("Full Sharpe >= 2.0",      M_full["sharpe"] >= 2.0),
        ("Full CAGR   >= 20%",      M_full["cagr"]   >= 0.20),
        ("IS Sharpe   >= 1.5",      M_is["sharpe"]   >= 1.5),
        ("OOS Sharpe  >= 1.5",      M_oos["sharpe"]  >= 1.5),
        ("|IS-OOS gap| <= 0.5",     gap              <= 0.5),
    ]:
        checks.append((name, bool(cond)))
        if not cond:
            passed = False
    print("\nHard requirement checks:")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  >> OVERALL: {'PASS' if passed else 'FAIL'}")

    # ---- save ----
    payload = {
        "config": {
            "universe": LEV_UNIVERSE,
            "sectors":  SECTOR_ETFS,
            "breadth_ma": BREADTH_MA,
            "breadth_high": BREADTH_HIGH,
            "hy_ema_win": HY_EMA_WIN,
            "vix_max": VIX_MAX,
            "confirm_days": CONFIRM_DAYS,
            "mom_lb": MOM_LB,
            "mom_skip": MOM_SKIP,
            "top_k": TOP_K,
            "rebal_days": REBAL_DAYS,
            "tc_bps_oneway": TC_BPS_ONEWAY,
            "is_end": str(IS_END.date()),
            "oos_start": str(OOS_START.date()),
            "oos_end": str(OOS_END.date()),
        },
        "metrics": {
            "full": M_full,
            "is":   M_is,
            "oos":  M_oos,
            "gap":  gap,
        },
        "time_in_market": float(tim),
        "avg_turnover":   float(out.turnover.mean()),
        "state_counts":   state_counts,
        "passed":         bool(passed),
    }
    with open(os.path.join(RES_D, "kraken_metrics.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)

    # returns CSV
    out.daily_ret.rename("ret").to_frame().to_csv(
        os.path.join(RES_D, "kraken_returns.csv"))
    out.weights.to_csv(os.path.join(RES_D, "kraken_weights.csv"))

    print(f"\nSaved: {os.path.join(RES_D, 'kraken_metrics.json')}")
    print(f"Saved: {os.path.join(RES_D, 'kraken_returns.csv')}")
    return payload


if __name__ == "__main__":
    main()
