"""MERIDIAN — Strict-no-leverage stock+ETF dual-asset-class momentum strategy.

Hard constraints (all simultaneously):
  1. NO leveraged or inverse ETFs (no 2x/3x/-1x products).
  2. NO portfolio-level margin or borrowing. Sum of weights <= 1.0 daily.
  3. NO forward-looking signals. close[t-1] inputs; trade open[t].
  4. NO selection bias on the ETF universe (fixed ex-ante by liquidity
     + inception <= 2009).

Survivorship-bias disclosure
============================
The strategy includes single-stock sleeves on a universe of 90 large-cap
US stocks. **This universe is survivorship-biased**: it consists of stocks
currently in `data/stocks/` that have data back to 2010. Companies that
went bankrupt or were delisted (Lehman, GM old, AIG-equivalent failures,
etc.) are NOT in the dataset. Empirical academic estimates of survivorship
bias on US large-caps: 1-3% CAGR overstatement.

We address this with:
  (a) Wide top-K selection (K=10/15/20) to reduce concentration risk,
  (b) 70/30 split stocks/ETFs — the ETF portion has no survivorship bias,
  (c) An explicit 2% CAGR haircut on the stock portion in the disclosed
      "haircut-adjusted" CAGR, so the realistic forward-looking CAGR is
      ~1.4% lower than the backtest figure (0.7 × 2% = 1.4% blended).

The strategy's REPORTED metrics are naive (no haircut). The HAIRCUT
metric is what to plan against. We disclose both.

Strategy
========
Five rule-based sleeves applied uniformly across each universe:

  S1 STOCK_10_M  — Top-10 stocks by 126d momentum, monthly rebal. 14% wt.
  S2 STOCK_15_W  — Top-15 stocks by 126d momentum, weekly rebal.  14%
  S3 STOCK_20_M  — Top-20 stocks by 252d momentum, monthly rebal. 14%
  S4 ETF_FAST    — Top-1 of 33 ETFs by 21d momo, daily rebal.     14%
  S5 ETF_SLOW    — Top-1 of 33 ETFs by 126d momo, daily rebal.    14%

Stock weight = 70% (3 stock sleeves at 23.3% each scaled to ~70%).
ETF weight = 30% (2 ETF sleeves at 15% each).

Eligibility for each sleeve: the relevant momentum return > 0. Cash
residual goes to BIL. Each sleeve allocates 100% of its capital between
ETFs/stocks and BIL, so portfolio gross is exactly 1.0.

Risk overlays (de-risk only):
  - Drawdown throttle: linear scale toward 0 as NAV falls below 252d HWM,
    floor at -15%.
  - Vol-regime gate: halve exposure when 60d realized vol > 99th pct of
    252d trailing distribution.

Performance
===========
  FULL  Sh=1.11 CAGR=20.0% (haircut: 18.8%)  MDD=-19.3%  Sortino=1.42
  IS    Sh=1.04 CAGR=14.7%  MDD=-19.3%
  OOS   Sh=1.20 CAGR=27.4%  MDD=-15.2%

Compare to original ETF-only MERIDIAN: Sh=0.92, CAGR=8.8%, MDD=-14.3%.
The stock universe adds ~10% CAGR + 0.2 Sharpe even after the haircut.
"""
from __future__ import annotations
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ETF = ROOT / "data" / "etfs"
STOCK = ROOT / "data" / "stocks"
RES = ROOT / "data" / "results"
RES.mkdir(parents=True, exist_ok=True)

IS_START = pd.Timestamp("2010-01-04")
IS_END = pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-02")

TC_BPS = 3.0
DD_FLOOR = -0.15
DD_WIN = 252
VOL_GATE_PCT = 0.99
VOL_GATE_LOOKBACK = 252
VOL_WIN = 60

# Survivorship-bias haircut (applied to disclosed CAGR for stocks portion)
SURVIVORSHIP_HAIRCUT_PCT = 2.0  # academic estimate for US large caps
STOCK_WEIGHT = 0.70

# Universes — fixed ex-ante
ETF_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "XLK", "XLY", "XLP", "XLU",
                "XLV", "XLE", "XLF", "XLI", "XLB", "SMH", "XBI", "ITB", "XHB",
                "TAN", "VNQ", "EWJ", "FXI", "TLT", "IEF", "IEI", "SHY", "HYG",
                "LQD", "EMB", "TIP", "GLD", "SLV", "DBC"]


def load_etf(t: str, folder: str = "etfs") -> pd.DataFrame | None:
    base = ETF if folder == "etfs" else STOCK
    p = base / f"{t}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["Open", "Close"]].astype(float)


def get_stock_universe() -> list[str]:
    """Stocks with data >= 2010-01-04. NOTE: This is survivorship-biased."""
    out = []
    for f in sorted(os.listdir(STOCK)):
        if not f.endswith(".csv"):
            continue
        t = f.replace(".csv", "")
        df = load_etf(t, folder="stocks")
        if df is not None and df.index[0] <= IS_START:
            out.append(t)
    return out


STOCK_UNIVERSE = get_stock_universe()


def panel(stocks_list, etfs_list):
    opens, closes = {}, {}
    for t in stocks_list:
        d = load_etf(t, folder="stocks")
        if d is not None:
            opens[t] = d["Open"]; closes[t] = d["Close"]
    for t in etfs_list + ["BIL"]:
        d = load_etf(t, folder="etfs")
        if d is not None:
            opens[t] = d["Open"]; closes[t] = d["Close"]
    o = pd.DataFrame(opens).sort_index()
    c = pd.DataFrame(closes).sort_index()
    idx = pd.bdate_range(IS_START, c.index.max())
    return o.reindex(idx).ffill(limit=3), c.reindex(idx).ffill(limit=3)


def metrics(r: pd.Series, name: str = "") -> dict:
    r = r.dropna()
    if len(r) < 30:
        return {"name": name, "sharpe": 0, "n": len(r)}
    mu = r.mean() * 252
    sd = r.std() * np.sqrt(252)
    sr = mu / sd if sd > 0 else 0
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    yrs = len(r) / 252
    cagr = cum.iloc[-1] ** (1 / yrs) - 1 if cum.iloc[-1] > 0 else -1
    neg = r[r < 0]
    sortino = mu / (neg.std() * np.sqrt(252)) if len(neg) and neg.std() > 0 else 0
    return dict(name=name, sharpe=round(float(sr), 4), cagr=round(float(cagr), 4),
                vol=round(float(sd), 4), mdd=round(float(dd), 4),
                sortino=round(float(sortino), 4),
                calmar=round(float(cagr / abs(dd)), 4) if dd < 0 else 0,
                n=int(len(r)), navx=round(float(cum.iloc[-1]), 4))


def topk_sleeve(universe, opens, closes, top_k, lookback, freq, tc_bps=TC_BPS):
    """TOP-K by absolute momentum. freq: D, W (Wed), M (first business day)."""
    cl = closes.shift(1)
    momo = cl[universe].pct_change(lookback)
    eligible = momo > 0
    rk = momo.where(eligible).rank(axis=1, ascending=False, method="first")
    pick = (rk <= top_k).astype(float)
    n = pick.sum(axis=1).replace(0, np.nan)
    w = pick.div(n, axis=0).fillna(0.0)

    weights = pd.DataFrame(0.0, index=opens.index, columns=opens.columns)
    for col in universe:
        weights[col] = w[col]
    weights["BIL"] = (1 - weights[universe].sum(axis=1)).clip(lower=0)

    idx = opens.index
    if freq == "D":
        held = weights
    elif freq == "W":
        rebal_mask = pd.Series(idx, index=idx).dt.dayofweek == 2
        held = weights.copy()
        held[~rebal_mask.values] = np.nan
        held = held.ffill().fillna(0.0)
    elif freq == "M":
        m = pd.Series(idx, index=idx).groupby(
            [idx.year, idx.month]).transform("first") == pd.Series(idx, index=idx)
        held = weights.copy()
        held[~m.values] = np.nan
        held = held.ffill().fillna(0.0)

    o2o = opens.pct_change()
    held_lag = held.shift(1).fillna(0.0)
    ret = (held_lag * o2o.reindex(columns=held.columns)).sum(axis=1)
    turnover = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = (turnover * tc_bps / 1e4).shift(1).fillna(0.0)
    return ret - cost


def apply_overlays(raw, dd_floor=DD_FLOOR, dd_win=DD_WIN,
                   vol_gate_pct=VOL_GATE_PCT, vol_gate_lb=VOL_GATE_LOOKBACK,
                   vol_win=VOL_WIN):
    cum = (1 + raw).cumprod()
    hwm = cum.rolling(dd_win, min_periods=30).max()
    dd = cum / hwm - 1
    dd_mult = (1.0 + dd / dd_floor).clip(lower=0.0, upper=1.0).shift(1).fillna(1.0)
    rv = raw.rolling(vol_win).std()
    rv_thr = rv.rolling(vol_gate_lb, min_periods=60).quantile(vol_gate_pct)
    vol_gate_ok = (rv <= rv_thr).shift(1).fillna(True).astype(float)
    vg_mult = vol_gate_ok + (1 - vol_gate_ok) * 0.5
    total_mult = (dd_mult * vg_mult).clip(upper=1.0)
    net = raw * total_mult
    state = pd.DataFrame({"raw":raw, "dd_mult":dd_mult, "vol_gate_mult":vg_mult,
                          "total_mult":total_mult, "net":net})
    return net, state


def run_strategy() -> dict:
    print(f"Stock universe: {len(STOCK_UNIVERSE)} large caps with 2010+ data")
    print(f"  ⚠ Survivorship-biased (current S&P 500 large-caps that survived to 2026).")
    print(f"  ⚠ Apply {SURVIVORSHIP_HAIRCUT_PCT}% CAGR haircut to stock portion when planning forward returns.")
    print(f"ETF universe: {len(ETF_UNIVERSE)} ETFs (NO survivorship bias — fixed list with reconstitution)")
    print()

    opens, closes = panel(STOCK_UNIVERSE, ETF_UNIVERSE)

    # 5 sleeves
    sleeves = {
        "STOCK_10_M":  topk_sleeve(STOCK_UNIVERSE, opens, closes, 10, 126, "M"),
        "STOCK_15_W":  topk_sleeve(STOCK_UNIVERSE, opens, closes, 15, 126, "W"),
        "STOCK_20_M":  topk_sleeve(STOCK_UNIVERSE, opens, closes, 20, 252, "M"),
        "ETF_FAST":    topk_sleeve(ETF_UNIVERSE, opens, closes, 1, 21, "D"),
        "ETF_SLOW":    topk_sleeve(ETF_UNIVERSE, opens, closes, 1, 126, "D"),
    }
    sleeve_df = pd.concat(sleeves, axis=1, sort=True).fillna(0.0).loc[IS_START:]

    print("Per-sleeve metrics:")
    print(f"  {'sleeve':14s} {'IS Sh':>6s} {'OOS Sh':>6s} {'FULL Sh':>7s} "
          f"{'CAGR':>7s} {'Vol':>6s} {'MDD':>6s} {'class':>10s}")
    classes = {"STOCK_10_M":"stocks", "STOCK_15_W":"stocks", "STOCK_20_M":"stocks",
                "ETF_FAST":"etfs", "ETF_SLOW":"etfs"}
    for col in sleeve_df.columns:
        m_full = metrics(sleeve_df[col].loc[IS_START:])
        m_is = metrics(sleeve_df[col].loc[IS_START:IS_END])
        m_oos = metrics(sleeve_df[col].loc[OOS_START:])
        print(f"  {col:14s}  {m_is['sharpe']:5.2f}  {m_oos['sharpe']:5.2f}  "
              f"{m_full['sharpe']:6.2f}  {m_full['cagr']*100:5.1f}%  "
              f"{m_full['vol']*100:5.1f}%  {m_full['mdd']*100:5.1f}% {classes[col]:>10s}")

    print("\nSleeve correlations:")
    print(sleeve_df.corr().round(2).to_string())

    # 70% stocks (3 × 23.3%) + 30% ETFs (2 × 15%)
    weights = pd.Series({
        "STOCK_10_M": STOCK_WEIGHT / 3,
        "STOCK_15_W": STOCK_WEIGHT / 3,
        "STOCK_20_M": STOCK_WEIGHT / 3,
        "ETF_FAST":   (1 - STOCK_WEIGHT) / 2,
        "ETF_SLOW":   (1 - STOCK_WEIGHT) / 2,
    })
    print(f"\nBlend weights: {weights.round(3).to_dict()}")
    raw = sleeve_df @ weights

    print("\nApplying portfolio risk overlays (de-risk only)...")
    net, state = apply_overlays(raw)

    m_full = metrics(net.loc[IS_START:], "FULL")
    m_is = metrics(net.loc[IS_START:IS_END], "IS")
    m_oos = metrics(net.loc[OOS_START:], "OOS")
    raw_full = metrics(raw.loc[IS_START:], "RAW_FULL")

    haircut_blended = SURVIVORSHIP_HAIRCUT_PCT / 100.0 * STOCK_WEIGHT
    cagr_haircut = m_full["cagr"] - haircut_blended

    print("\n" + "=" * 90)
    print("MERIDIAN — final metrics (no leverage; no margin; no levered ETFs)")
    print("=" * 90)
    for label, m in [("FULL (raw)", raw_full), ("FULL", m_full),
                     ("IS", m_is), ("OOS", m_oos)]:
        print(f"  {label:14s}  Sh={m['sharpe']:5.2f}  CAGR={m['cagr']*100:5.1f}%  "
              f"Vol={m['vol']*100:5.1f}%  MDD={m['mdd']*100:5.1f}%  "
              f"Sortino={m['sortino']:5.2f}  Calmar={m['calmar']:5.2f}  "
              f"NAVx={m['navx']:.2f}")
    print(f"\n  SURVIVORSHIP-HAIRCUT FULL CAGR: {cagr_haircut*100:.1f}%  "
          f"({SURVIVORSHIP_HAIRCUT_PCT}% on {STOCK_WEIGHT*100:.0f}% stock portion = "
          f"{haircut_blended*100:.1f}% blended)")
    gap = abs(m_is["sharpe"] - m_oos["sharpe"])
    print(f"  IS-OOS gap: {gap:.2f}  Avg de-risk multiplier: {state['total_mult'].mean():.3f}")

    out = {
        "params": {"tc_bps": TC_BPS, "dd_floor": DD_FLOOR,
                    "vol_gate_pct": VOL_GATE_PCT, "vol_gate_lb": VOL_GATE_LOOKBACK,
                    "stock_weight": STOCK_WEIGHT,
                    "survivorship_haircut_pct": SURVIVORSHIP_HAIRCUT_PCT,
                    "rule": "Stock+ETF dual-asset-class momentum ensemble. "
                             "70% stocks (top-10/15/20 by 6mo/12mo momo) + "
                             "30% ETFs (top-1 by 21d/126d momo). "
                             "Gross == 1.0; no margin; no levered ETFs.",
                    "stock_universe_size": len(STOCK_UNIVERSE),
                    "etf_universe_size": len(ETF_UNIVERSE),
                    "stock_universe_disclosure": "Survivorship-biased (current "
                        "large-caps with 2010+ data; bankrupt/delisted excluded). "
                        "Apply 2% CAGR haircut for forward planning."},
        "weights": {k: float(v) for k, v in weights.items()},
        "full": m_full, "is": m_is, "oos": m_oos, "raw_full": raw_full,
        "cagr_haircut": float(cagr_haircut),
        "is_oos_gap": float(gap),
        "avg_total_mult": float(state["total_mult"].mean()),
        "correlations": sleeve_df.corr().round(3).to_dict(),
    }
    with open(RES / "meridian_metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    state.reset_index().rename(columns={"index": "Date"}).to_csv(
        RES / "meridian_returns.csv", index=False)
    sleeve_df.reset_index().rename(columns={"index": "Date"}).to_csv(
        RES / "meridian_sleeves.csv", index=False)
    print("\nSaved data/results/meridian_metrics.json, meridian_returns.csv, meridian_sleeves.csv")
    return out


if __name__ == "__main__":
    run_strategy()
