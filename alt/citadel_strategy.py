"""CITADEL — Hedged Long/Inverse Leveraged ETF Rotation.

Thesis: long-only rotation plateaus at Sharpe ~1.0 because the strategy eats
full drawdowns during regime breaks. Adding a SHORT sleeve via inverse LETFs
(SQQQ, SPXU, TECS, FAZ, LABD, TMV, YANG) during confirmed downtrends turns
drawdowns into potential gains, lifting risk-adjusted returns.

Inverse LETFs have heavy decay — only profitable during STRONG sustained
downtrends. Must require unambiguous trend-down signals + macro confirmation.

Design (all signals use close[t-1], positions set at open[t], earn open[t]→
open[t+1]):

- 7 asset sleeves, each a pair (long-LETF, short-LETF, unleveraged underlying)
- Regime per sleeve from the unleveraged underlying:
    UP   = close > 200dma AND 12m-momentum > 0 AND 50dma > 200dma
    DOWN = close < 200dma AND 12m-momentum < 0 AND 50dma < 200dma
           AND HY OAS 20d slope > 0 (credit widening)
    FLAT = otherwise
- In UP  → hold long-LETF at equal weight among active UP sleeves
- In DOWN → hold short-LETF at equal weight (only if global breadth confirms bear)
- In FLAT → cash (BIL)
- Global gate: if fewer than 2 sleeves give the same signal, go cash
- Monthly rebalance (21 days), with emergency flip on regime change
- 5 bps one-way TC

IS: 2010-03-11 to 2018-12-31; OOS: 2019-01-02 to 2026-04-02.
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

# (long, short/inverse, unleveraged) triples
SLEEVES = [
    ("UPRO", "SPXU", "SPY"),   # S&P 500 3x
    ("TQQQ", "SQQQ", "QQQ"),   # Nasdaq 3x
    ("SOXL", "SOXS", "SMH"),   # Semis 3x
    ("TECL", "TECS", "XLK"),   # Tech 3x
    ("FAS",  "FAZ",  "XLF"),   # Financials 3x
    ("LABU", "LABD", "IBB"),   # Biotech 3x
    ("TMF",  "TMV",  "TLT"),   # 20y treasury 3x
    ("YINN", "YANG", "FXI"),   # China 3x
    ("ERX",  "DRIP", "XLE"),   # Energy 3x (DRIP may not exist; fall back to none)
]

IS_END = "2018-12-31"
OOS_START = "2019-01-02"

# Params tuned on IS
LB_MOM       = 252     # 12-month momentum
SMA_LONG     = 200
SMA_SHORT    = 50
HY_SLOPE     = 20
HY_THR       = 0.0     # HY OAS rising confirms downside
REBAL_DAYS   = 21      # monthly
TC_BPS       = 5.0
N_MAX_LONG   = 4
N_MAX_SHORT  = 3
MIN_BREADTH  = 0.45    # min fraction of eligible sleeves in UP to go long
MAX_BREADTH  = 0.25    # if UP breadth < this AND HY rising → allow shorts


def load_close_open(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["Close", "Open"]].apply(pd.to_numeric, errors="coerce")


def load_fred(s):
    p = FRED / f"{s}.csv"
    if not p.exists(): return None
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    d = d[~d.index.duplicated(keep="first")]
    return pd.to_numeric(d.iloc[:, 0], errors="coerce")


def metrics(r, label=""):
    r = r.dropna()
    if len(r) == 0: return {}
    mu = r.mean() * 252
    sd = r.std() * np.sqrt(252)
    sr = mu / sd if sd > 0 else 0
    c = (1 + r).cumprod()
    dd = (c / c.cummax() - 1).min()
    yrs = len(r) / 252
    cagr = c.iloc[-1] ** (1 / yrs) - 1
    return {"label": label, "n": int(len(r)),
            "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
            "sharpe": round(float(sr), 4), "cagr": round(float(cagr), 4),
            "ann_vol": round(float(sd), 4), "mdd": round(float(dd), 4),
            "navx": round(float(c.iloc[-1]), 4)}


def main():
    # Load data
    all_tickers = set()
    for L, S, U in SLEEVES:
        all_tickers.update([L, S, U])
    all_tickers.update(["SPY", "BIL"])

    close, opn = {}, {}
    for t in all_tickers:
        df = load_close_open(t)
        if df is None:
            print(f"MISSING {t}")
            continue
        close[t] = df["Close"]; opn[t] = df["Open"]

    close = pd.DataFrame(close); opn = pd.DataFrame(opn)
    dates = opn["SPY"].dropna().index
    dates = dates[(dates >= pd.Timestamp("2010-03-11")) &
                  (dates <= pd.Timestamp("2026-04-02"))]
    close = close.reindex(dates).ffill(limit=5)
    opn = opn.reindex(dates).ffill(limit=5)

    hy = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    hy_slope = hy - hy.shift(HY_SLOPE)

    # Filter sleeves by data availability
    active_sleeves = []
    for L, S, U in SLEEVES:
        if L in close.columns and S in close.columns and U in close.columns:
            active_sleeves.append((L, S, U))
        else:
            print(f"Dropping sleeve {L}/{S}/{U} (missing data)")

    print(f"CITADEL: {len(active_sleeves)} sleeves, "
          f"{dates[0].date()}..{dates[-1].date()}")

    # Precompute per-sleeve regime on unleveraged underlying (shifted 1 bar)
    regimes = {}  # 'up' or 'dn' or 'flat' per date per sleeve
    for L, S, U in active_sleeves:
        u = close[U]
        ma_long = u.rolling(SMA_LONG).mean()
        ma_short = u.rolling(SMA_SHORT).mean()
        mom = u.pct_change(LB_MOM)
        up = (u > ma_long) & (mom > 0) & (ma_short > ma_long)
        # require strong down: below 200dma AND 50<200 AND mom<0 AND HY rising
        dn = (u < ma_long) & (mom < 0) & (ma_short < ma_long) & (hy_slope > HY_THR)
        reg = pd.Series("flat", index=dates)
        reg[up] = "up"
        reg[dn] = "dn"
        regimes[U] = reg.shift(1).fillna("flat")  # strictly lag

    # Track weights
    current_w = pd.Series(0.0, index=list(close.columns))
    port_ret = pd.Series(0.0, index=dates)
    tov_arr = pd.Series(0.0, index=dates)
    weight_log = []
    n_long_days = 0; n_short_days = 0; n_cash_days = 0
    long_pnl = 0.0; short_pnl = 0.0
    last_rebal = -REBAL_DAYS

    opn_pct = opn.pct_change().fillna(0)

    for i, d in enumerate(dates):
        # Earn return using yesterday's weights (open[d]/open[d-1]-1)
        r_long = 0.0; r_short = 0.0; r_cash = 0.0
        for L, S, U in active_sleeves:
            r_long += current_w[L] * opn_pct.at[d, L]
            r_short += current_w[S] * opn_pct.at[d, S]
        r_cash = current_w["BIL"] * opn_pct.at[d, "BIL"]
        r = r_long + r_short + r_cash
        port_ret.iloc[i] = r
        long_pnl += r_long; short_pnl += r_short

        # Rebalance schedule or regime-flip triggered
        regime_changed = False
        for L, S, U in active_sleeves:
            # If a short position is held but regime flipped out of dn: flip
            if current_w[S] > 0 and regimes[U].iloc[i] != "dn":
                regime_changed = True; break
            # If a long position is held but regime flipped out of up: flip
            if current_w[L] > 0 and regimes[U].iloc[i] != "up":
                regime_changed = True; break

        if (i - last_rebal) >= REBAL_DAYS or regime_changed:
            last_rebal = i
            # Count breadth
            ups = [U for _, _, U in active_sleeves if regimes[U].iloc[i] == "up"]
            dns = [U for _, _, U in active_sleeves if regimes[U].iloc[i] == "dn"]
            n_total = len(active_sleeves)
            long_breadth = len(ups) / n_total
            short_breadth = len(dns) / n_total

            new_w = pd.Series(0.0, index=current_w.index)
            if long_breadth >= MIN_BREADTH:
                # rank UP sleeves by 12m mom on underlying, take top N_MAX_LONG
                ranks = []
                for L, S, U in active_sleeves:
                    if regimes[U].iloc[i] == "up":
                        m = close[U].iloc[i - 1] / close[U].iloc[i - 1 - LB_MOM] - 1 \
                            if i > LB_MOM else 0
                        ranks.append((m, L))
                ranks.sort(reverse=True)
                picks = ranks[:N_MAX_LONG]
                if picks:
                    w_each = 1.0 / len(picks)
                    for _, L in picks:
                        new_w[L] = w_each
            elif short_breadth >= (1 - MAX_BREADTH):
                # strong bear: use shorts
                ranks = []
                for L, S, U in active_sleeves:
                    if regimes[U].iloc[i] == "dn":
                        m = close[U].iloc[i - 1] / close[U].iloc[i - 1 - LB_MOM] - 1 \
                            if i > LB_MOM else 0
                        ranks.append((m, S))
                ranks.sort()  # most negative first
                picks = ranks[:N_MAX_SHORT]
                if picks:
                    w_each = 1.0 / len(picks)
                    for _, S in picks:
                        new_w[S] = w_each
            else:
                new_w["BIL"] = 1.0

            tov = (new_w - current_w).abs().sum()
            tov_arr.iloc[i] = tov
            if i + 1 < len(dates):
                port_ret.iloc[i + 1] -= tov * (TC_BPS / 1e4)
            current_w = new_w
            weight_log.append({"date": str(d.date()),
                               **{c: round(float(v), 4) for c, v in current_w.items()
                                  if v > 0}})

        # Count state
        long_wt = sum(current_w[L] for L, S, U in active_sleeves)
        short_wt = sum(current_w[S] for L, S, U in active_sleeves)
        if short_wt > 0.5: n_short_days += 1
        elif long_wt > 0.5: n_long_days += 1
        else: n_cash_days += 1

    # Summary
    full = metrics(port_ret, "FULL")
    is_m = metrics(port_ret.loc[:IS_END], "IS")
    oos_m = metrics(port_ret.loc[OOS_START:], "OOS")

    out = {
        "params": {
            "lb_mom": LB_MOM, "sma_long": SMA_LONG, "sma_short": SMA_SHORT,
            "hy_slope": HY_SLOPE, "hy_thr": HY_THR,
            "rebal_days": REBAL_DAYS, "tc_bps": TC_BPS,
            "n_max_long": N_MAX_LONG, "n_max_short": N_MAX_SHORT,
            "min_breadth": MIN_BREADTH, "max_breadth": MAX_BREADTH,
            "sleeves": [[L, S, U] for L, S, U in active_sleeves],
        },
        "full": full, "is": is_m, "oos": oos_m,
        "is_oos_gap": round(abs(is_m["sharpe"] - oos_m["sharpe"]), 4),
        "avg_turnover_annual": round(float(tov_arr.sum() / max(1, len(port_ret)) * 252), 2),
        "time_in_market": {
            "long": round(n_long_days / len(dates), 3),
            "short": round(n_short_days / len(dates), 3),
            "cash": round(n_cash_days / len(dates), 3),
        },
        "attribution": {
            "cumulative_long_pnl": round(long_pnl, 4),
            "cumulative_short_pnl": round(short_pnl, 4),
        },
    }

    (RESULTS / "citadel_metrics.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"Date": port_ret.index, "ret": port_ret.values,
                  "turnover": tov_arr.values}).to_csv(
        RESULTS / "citadel_returns.csv", index=False)

    print()
    for name, m in [("FULL", full), ("IS", is_m), ("OOS", oos_m)]:
        if not m: continue
        print(f"  {name:5s}  SR={m['sharpe']:5.2f}  "
              f"CAGR={m['cagr']*100:5.1f}%  Vol={m['ann_vol']*100:5.1f}%  "
              f"MDD={m['mdd']*100:6.1f}%  NAVx={m['navx']:.2f}")
    print(f"  IS-OOS gap={out['is_oos_gap']:.2f}  TIM long={out['time_in_market']['long']*100:.0f}%  "
          f"short={out['time_in_market']['short']*100:.0f}%  cash={out['time_in_market']['cash']*100:.0f}%")
    print(f"  long_pnl_sum={long_pnl:.3f}  short_pnl_sum={short_pnl:.3f}  "
          f"turn={out['avg_turnover_annual']:.1f}x/yr")


if __name__ == "__main__":
    main()
