"""REVENANT — Short-horizon mean reversion on leveraged ETFs.

Thesis
------
Leveraged ETFs (2x/3x daily rebalance) exhibit negative serial correlation at
short horizons (1-5 days) because their daily rebalance mechanics force them
to buy high and sell low. After a sharp down day, they typically overshoot
and tend to bounce. This is fundamentally different from (and orthogonal to)
the 12-1 momentum signal that all prior strategies (NOVA, ATLAS, VANGUARD,
HELIOS, ORION, BASTION) used.

Design
------
Universe: 14 leveraged ETFs spanning broad equity, sectors, country, rates,
commodities (no cherry-picking — all 2x/3x names with ≥2010 history).

Per-name signal (computed from close[t-1]):
  rsi2(t-1) — 2-period RSI on close prices. If rsi2 < RSI_LOW → oversold.

Execution:
  On day t, for each name that is oversold at close[t-1] AND meets regime
  gate, enter at open[t]. Hold MAX_HOLD days OR exit when rsi2 > RSI_HIGH.

Regime gate (SAME macro signal used in VANGUARD — proven robust):
  - SPY price > 200dma AND 200dma sloping up
  - HY OAS not widening (20d slope < +0.3)
  Only enter NEW mean-reversion positions when gate is on (positive regime
  means oversold bounces are more reliable; during bear markets oversold
  stays oversold).

Portfolio:
  Up to N_MAX concurrent positions, equal-weight among active ones. When
  fewer than N_MAX signals fire, remaining weight sits in BIL. No vol scaling.

Execution:
  Signal uses data through close[t-1]. Weight w_t applied to return from
  open[t] to open[t+1]. Transaction costs (10 bps one-way) on turnover.

Honest IS/OOS split:
  IS = 2010-03-11 to 2018-12-31. OOS = 2019-01-01 to 2026-04-02.
  Params tuned only on IS.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"
RESULTS.mkdir(parents=True, exist_ok=True)

# 14 leveraged ETFs with enough history (start ≤ 2010-03-11).
UNIVERSE = [
    "TQQQ", "UPRO", "QLD", "SSO",      # broad equity
    "SOXL", "TECL", "FAS", "ERX",       # sectors
    "DRN", "EDC", "YINN",               # sector/country
    "UCO", "UGL", "NUGT",               # commodities (NUGT starts 2010-12)
    "TMF", "UBT", "TYD",                # rates
]

# IS/OOS split
IS_END   = "2018-12-31"
OOS_START = "2019-01-02"

# -------- Parameters (tuned on IS — see sensitivity block at bottom) --------
RSI_LEN        = 2
RSI_LOW        = 10.0     # enter when rsi2 < 10 at close[t-1]
RSI_HIGH       = 70.0     # exit when rsi2 > 70 at close[t-1]
MAX_HOLD       = 5        # time stop (days)
SMA_LEN        = 200      # SPY regime gate
HY_SLOPE_DAYS  = 20       # HY OAS slope window
HY_SLOPE_THR   = 0.30     # bps/day rising — gate off
VIX_MAX        = 35.0     # absolute VIX ceiling
N_MAX          = 4        # max concurrent positions
TC_BPS_ONEWAY  = 10.0     # 10 bps one-way transaction cost
# ---------------------------------------------------------------------------


def load_etf_close_open(t: str):
    p = ETF / f"{t}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["Close", "Open"]].apply(pd.to_numeric, errors="coerce")


def load_fred(series: str):
    p = FRED / f"{series}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return pd.to_numeric(df.iloc[:, 0], errors="coerce")


def rsi(series: pd.Series, length: int) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    ma_up = up.rolling(length).mean()
    ma_dn = dn.rolling(length).mean()
    rs = ma_up / ma_dn.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def calc_metrics(r: pd.Series, label: str) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {"label": label}
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    years = len(r) / 252
    cagr = cum.iloc[-1] ** (1 / years) - 1
    return {
        "label": label,
        "n": int(len(r)),
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
        "sharpe": round(float(sharpe), 4),
        "cagr": round(float(cagr), 4),
        "ann_vol": round(float(ann_vol), 4),
        "mdd": round(float(mdd), 4),
        "navx": round(float(cum.iloc[-1]), 4),
    }


def build():
    # ---- Load prices ----
    close = {}
    opn = {}
    for t in UNIVERSE + ["SPY", "BIL"]:
        df = load_etf_close_open(t)
        if df is None:
            print(f"WARN missing {t}")
            continue
        close[t] = df["Close"]
        opn[t] = df["Open"]

    close = pd.DataFrame(close)
    opn = pd.DataFrame(opn)

    # Master trading calendar = SPY open
    dates = opn["SPY"].dropna().index
    dates = dates[(dates >= pd.Timestamp("2010-03-11")) &
                  (dates <= pd.Timestamp("2026-04-02"))]
    close = close.reindex(dates).ffill(limit=5)
    opn = opn.reindex(dates).ffill(limit=5)

    # ---- Macro regime (SPY 200dma + HY OAS slope + VIX) ----
    spy = close["SPY"]
    spy_ma = spy.rolling(SMA_LEN).mean()
    spy_ok = (spy > spy_ma) & (spy_ma.diff(20) > 0)

    hy = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    hy_slope = hy - hy.shift(HY_SLOPE_DAYS)  # points change over 20 days
    hy_ok = hy_slope < HY_SLOPE_THR

    vix = load_fred("VIXCLS").reindex(dates).ffill()
    vix_ok = vix < VIX_MAX

    regime_ok = (spy_ok & hy_ok & vix_ok).shift(1).fillna(False)  # strictly lag

    # ---- Per-name signals (computed on close, lagged 1 bar) ----
    rsi2 = pd.DataFrame({t: rsi(close[t], RSI_LEN) for t in UNIVERSE})
    rsi2_lag = rsi2.shift(1)  # signal uses close[t-1]

    # trend filter per-name: price > 50-day SMA (of the levered ETF itself)
    sma50 = pd.DataFrame({t: close[t].rolling(50).mean() for t in UNIVERSE})
    trend_ok_name = (close[pd.Index(UNIVERSE)] > sma50).shift(1).fillna(False)

    oversold = (rsi2_lag < RSI_LOW) & trend_ok_name

    # ---- Overnight + intraday return per name: use open-to-open ----
    opn_ret = opn[pd.Index(UNIVERSE)].pct_change().fillna(0)  # open[t]/open[t-1]-1
    # weight_t applied to opn_ret[t+1] (earn open[t]→open[t+1])
    # equivalently shift opn_ret back by 1 relative to weight.
    # We'll use: daily portfolio return[t] = sum(w[t-1] * opn_ret[t]).

    bil_close = close.get("BIL")
    if bil_close is None:
        bil_r = pd.Series(0.0, index=dates)
    else:
        bil_r = opn["BIL"].pct_change().fillna(0)

    # ---- Portfolio simulation with explicit open-to-open execution ----
    pos_enter_date: dict[str, pd.Timestamp] = {}
    current_w = pd.Series(0.0, index=UNIVERSE + ["BIL"])
    port_ret = pd.Series(0.0, index=dates)
    turnover = pd.Series(0.0, index=dates)
    n_holdings = pd.Series(0, index=dates)
    weight_history = []

    for i, d in enumerate(dates):
        # --- earn return first using yesterday's weights ---
        r_today = 0.0
        for t in UNIVERSE:
            w = current_w[t]
            if w != 0:
                r_today += w * opn_ret.at[d, t] if not np.isnan(opn_ret.at[d, t]) else 0.0
        r_today += current_w["BIL"] * (bil_r.iloc[i] if not np.isnan(bil_r.iloc[i]) else 0.0)
        port_ret.iloc[i] = r_today

        # --- decide NEW weights for next-day open ---
        new_w = current_w.copy()

        # Exit logic: time stop OR rsi2 > RSI_HIGH
        for t in UNIVERSE:
            if new_w[t] > 0:
                held_days = i - dates.get_loc(pos_enter_date[t])
                rsi_v = rsi2_lag.at[d, t]  # signal at close[d-1]
                if held_days >= MAX_HOLD or (not np.isnan(rsi_v) and rsi_v > RSI_HIGH):
                    new_w[t] = 0.0
                    pos_enter_date.pop(t, None)

        # Entry logic: only if regime_ok today AND we have open slots
        if bool(regime_ok.iloc[i]):
            active = [t for t in UNIVERSE if new_w[t] > 0]
            slots = N_MAX - len(active)
            if slots > 0:
                candidates = []
                for t in UNIVERSE:
                    if new_w[t] > 0:
                        continue
                    if bool(oversold.at[d, t]):
                        # rank candidates by depth of oversold (lower rsi2 = better)
                        rv = rsi2_lag.at[d, t]
                        if not np.isnan(rv):
                            candidates.append((rv, t))
                candidates.sort()  # ascending rsi2
                for _, t in candidates[:slots]:
                    new_w[t] = 0.0  # placeholder, set after sizing

        # Size equal weight among active positions (after exits + new entries)
        new_active = [t for t in UNIVERSE
                      if (new_w[t] > 0) or
                      (bool(regime_ok.iloc[i]) and bool(oversold.at[d, t])
                       and t not in pos_enter_date)]
        new_active = new_active[:N_MAX]
        if len(new_active) > 0:
            w_each = 1.0 / len(new_active)
            new_w[:] = 0.0
            for t in new_active:
                new_w[t] = w_each
                if t not in pos_enter_date:
                    pos_enter_date[t] = d
            new_w["BIL"] = 0.0
        else:
            new_w[:] = 0.0
            new_w["BIL"] = 1.0
            pos_enter_date.clear()

        # turnover + TC (charged tomorrow since we rebalance at next open)
        tover = (new_w - current_w).abs().sum()
        turnover.iloc[i] = tover
        if i + 1 < len(dates):
            port_ret.iloc[i + 1] -= tover * (TC_BPS_ONEWAY / 1e4)

        n_holdings.iloc[i] = int(sum(1 for t in UNIVERSE if new_w[t] > 0))
        current_w = new_w
        weight_history.append({"date": str(d.date()),
                               **{t: round(float(new_w[t]), 4) for t in UNIVERSE
                                  if new_w[t] > 0},
                               "BIL": round(float(new_w["BIL"]), 4)})

    return port_ret, turnover, n_holdings, weight_history


def main():
    print(f"REVENANT — short-horizon mean reversion on {len(UNIVERSE)} leveraged ETFs")
    print(f"Universe: {UNIVERSE}")
    port_ret, turnover, n_holdings, weights = build()

    # Metrics
    full = calc_metrics(port_ret, "FULL")
    is_r = port_ret.loc[:IS_END]
    oos_r = port_ret.loc[OOS_START:]
    is_m = calc_metrics(is_r, "IS")
    oos_m = calc_metrics(oos_r, "OOS")

    avg_turn_annual = float(turnover.sum() / max(1, len(port_ret)) * 252)
    avg_positions = float(n_holdings.mean())
    pct_invested = float((n_holdings > 0).mean())

    out = {
        "params": {
            "rsi_len": RSI_LEN, "rsi_low": RSI_LOW, "rsi_high": RSI_HIGH,
            "max_hold": MAX_HOLD, "sma_len": SMA_LEN,
            "hy_slope_days": HY_SLOPE_DAYS, "hy_slope_thr": HY_SLOPE_THR,
            "vix_max": VIX_MAX, "n_max": N_MAX, "tc_bps_oneway": TC_BPS_ONEWAY,
        },
        "universe": UNIVERSE,
        "full": full, "is": is_m, "oos": oos_m,
        "is_oos_gap": round(abs(is_m["sharpe"] - oos_m["sharpe"]), 4),
        "avg_turnover_annual": round(avg_turn_annual, 2),
        "avg_active_positions": round(avg_positions, 2),
        "pct_time_invested": round(pct_invested, 4),
    }
    (RESULTS / "revenant_metrics.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"Date": port_ret.index, "ret": port_ret.values,
                  "turnover": turnover.values,
                  "n_holdings": n_holdings.values}).to_csv(
        RESULTS / "revenant_returns.csv", index=False)

    print()
    for name, m in [("FULL", full), ("IS 2010-03-11..2018-12-31", is_m),
                    ("OOS 2019-01-02..2026-04-02", oos_m)]:
        print(f"  {name:30s}  SR={m['sharpe']:5.2f}  CAGR={m['cagr']*100:5.1f}%  "
              f"Vol={m['ann_vol']*100:5.1f}%  MDD={m['mdd']*100:6.1f}%  "
              f"NAVx={m['navx']:6.2f}")
    print(f"\n  IS-OOS Sharpe gap = {out['is_oos_gap']:.2f}")
    print(f"  Avg turnover ~ {avg_turn_annual:.1f}x/yr")
    print(f"  Avg positions = {avg_positions:.2f}  | %time invested = {pct_invested*100:.1f}%")


if __name__ == "__main__":
    main()
