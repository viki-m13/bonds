"""Does adding small/mid-cap, intl ADRs, and small-cap tech help PURE & COMPOSITE?

Tests the stock-universe extension on:
  - PURE       (top-3 stocks 126d, weekly Wed)
  - COMPOSITE  (3 stock sleeves: 3-W, 5-W, 7-M + 2 ETF sleeves unchanged)

Compares BASE (90 large-caps) vs EXTENDED (90 + small/mid + intl + small-tech)
on the IS/OOS/FULL splits.

NOTE on bias: extended pool is ALSO survivorship-biased (current liquid names
with 2010+ history). Bias is *worse* than the large-cap pool because small caps
and intl ADRs delist/fail more often. A bigger haircut should be applied.
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
STOCK_EXT = ROOT / "data" / "stocks_extended"
RES = ROOT / "data" / "results"

IS_START = pd.Timestamp("2010-01-04")
IS_END = pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-02")
TC_BPS = 3.0
DD_FLOOR_PURE = -0.25
DD_FLOOR_COMP = -0.20
DD_WIN = 252
VOL_GATE_PCT = 0.99
VOL_GATE_LOOKBACK = 252
VOL_WIN = 60

ETF_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "XLK", "XLY", "XLP", "XLU",
                "XLV", "XLE", "XLF", "XLI", "XLB", "SMH", "XBI", "ITB", "XHB",
                "TAN", "VNQ", "EWJ", "FXI", "TLT", "IEF", "IEI", "SHY", "HYG",
                "LQD", "EMB", "TIP", "GLD", "SLV", "DBC"]


def load(t, folder):
    base = {"etfs": ETF, "stocks": STOCK, "stocks_ext": STOCK_EXT}[folder]
    p = base / f"{t}.csv"
    if not p.exists(): return None
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if {"Open", "Close"}.issubset(df.columns):
        return df[["Open", "Close"]].astype(float)
    return None


def get_stock_universe_base():
    out = []
    for f in sorted(os.listdir(STOCK)):
        if not f.endswith(".csv"): continue
        t = f.replace(".csv", "")
        df = load(t, "stocks")
        if df is not None and df.index[0] <= IS_START:
            out.append(t)
    return out


def get_stock_universe_extended(min_start=IS_START):
    """Combine base (large-cap) + extended folder names with ≥IS_START history."""
    base = get_stock_universe_base()
    ext = []
    if STOCK_EXT.exists():
        for f in sorted(os.listdir(STOCK_EXT)):
            if not f.endswith(".csv"): continue
            t = f.replace(".csv", "")
            if t in base: continue
            df = load(t, "stocks_ext")
            if df is not None and df.index[0] <= min_start:
                ext.append(t)
    return sorted(set(base + ext)), sorted(ext)


def metrics(r, name=""):
    r = r.dropna()
    if len(r) < 30: return {"name": name}
    mu = r.mean() * 252; sd = r.std() * np.sqrt(252)
    sr = mu / sd if sd > 0 else 0
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    yrs = len(r) / 252
    cagr = cum.iloc[-1] ** (1 / yrs) - 1 if cum.iloc[-1] > 0 else -1
    neg = r[r < 0]
    sortino = mu / (neg.std() * np.sqrt(252)) if len(neg) and neg.std() > 0 else 0
    return dict(name=name, sharpe=round(float(sr), 3),
                cagr=round(float(cagr), 4), vol=round(float(sd), 4),
                mdd=round(float(dd), 4), sortino=round(float(sortino), 3),
                calmar=round(float(cagr / abs(dd)), 3) if dd < 0 else 0,
                n=int(len(r)))


def panel(stocks, etfs):
    """Load Open/Close panels. stocks may include both base and extended folders."""
    base_set = set(get_stock_universe_base())
    opens_d, closes_d = {}, {}
    for t in stocks:
        folder = "stocks" if t in base_set else "stocks_ext"
        d = load(t, folder)
        if d is not None:
            opens_d[t] = d["Open"]; closes_d[t] = d["Close"]
    for t in etfs:
        d = load(t, "etfs")
        if d is not None:
            opens_d[t] = d["Open"]; closes_d[t] = d["Close"]
    return pd.DataFrame(opens_d), pd.DataFrame(closes_d)


def topk_sleeve(univ, opens, closes, k, lookback, rebal):
    idx = pd.bdate_range(IS_START, closes.index.max())
    o = opens.reindex(idx).ffill(limit=3)
    c = closes.reindex(idx).ffill(limit=3)
    cl = c.shift(1)
    cols = [t for t in univ if t in cl.columns]
    momo = cl[cols].pct_change(lookback)
    elig = momo > 0
    rk = momo.where(elig).rank(axis=1, ascending=False, method="first")
    pick = (rk <= k).astype(float)
    n = pick.sum(axis=1).replace(0, np.nan)
    w = pick.div(n, axis=0).fillna(0.0)

    if rebal == "W":
        mask = pd.Series(idx, index=idx).dt.dayofweek == 2
    elif rebal == "M":
        mask = idx.to_series().dt.is_month_end | (
            idx.to_series().shift(-1).dt.month != idx.to_series().dt.month)
    else:
        mask = pd.Series(True, index=idx)

    held = w.copy(); held[~mask.values] = np.nan
    held = held.ffill().fillna(0.0)
    o2o = o.pct_change()
    held_lag = held.shift(1).fillna(0.0)
    raw = (held_lag * o2o.reindex(columns=held.columns)).sum(axis=1)
    tov = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = (tov * TC_BPS / 1e4).shift(1).fillna(0.0)
    return raw - cost, held


def overlay(raw, dd_floor=-0.25):
    cum = (1 + raw).cumprod()
    hwm = cum.rolling(DD_WIN, min_periods=30).max()
    dd = (cum / hwm - 1)
    dd_mult = (1.0 + dd / dd_floor).clip(lower=0.0, upper=1.0).shift(1).fillna(1.0)
    rv = raw.rolling(VOL_WIN).std()
    rv_thr = rv.rolling(VOL_GATE_LOOKBACK, min_periods=60).quantile(VOL_GATE_PCT)
    vol_gate_ok = (rv <= rv_thr).shift(1).fillna(True).astype(float)
    vg_mult = vol_gate_ok + (1 - vol_gate_ok) * 0.5
    total = (dd_mult * vg_mult).clip(upper=1.0)
    return raw * total


def run_pure(univ, label):
    opens, closes = panel(univ, ["BIL"])
    raw, held = topk_sleeve(univ, opens, closes, 3, 126, "W")
    net = overlay(raw, dd_floor=DD_FLOOR_PURE)
    return {
        "label": label, "univ_n": len(univ),
        "raw_full": metrics(raw.loc[IS_START:], "RAW"),
        "full": metrics(net.loc[IS_START:], "FULL"),
        "is": metrics(net.loc[IS_START:IS_END], "IS"),
        "oos": metrics(net.loc[OOS_START:], "OOS"),
        "ret": net,
    }, held


def run_composite(stock_univ, label):
    """COMPOSITE: 3 stock sleeves (3-W, 5-W, 7-M) + 2 ETF sleeves (FAST/SLOW). Equal-vol blend."""
    opens, closes = panel(stock_univ, ETF_UNIVERSE)
    s1, _ = topk_sleeve(stock_univ, opens, closes, 3, 126, "W")
    s2, _ = topk_sleeve(stock_univ, opens, closes, 5, 126, "W")
    s3, _ = topk_sleeve(stock_univ, opens, closes, 7, 252, "M")
    s4, _ = topk_sleeve(ETF_UNIVERSE, opens, closes, 1, 21, "D")
    s5, _ = topk_sleeve(ETF_UNIVERSE, opens, closes, 1, 126, "D")

    sleeves = pd.concat({"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5}, axis=1)
    sleeves = sleeves.fillna(0.0).loc[IS_START:]
    # Production weights: 70/3 each stock, 15% each ETF (matches meridian_strategy.py)
    w = pd.Series({"S1": 0.70/3, "S2": 0.70/3, "S3": 0.70/3, "S4": 0.15, "S5": 0.15})
    raw = (sleeves * w).sum(axis=1)
    net = overlay(raw, dd_floor=DD_FLOOR_COMP)
    return {
        "label": label, "univ_n": len(stock_univ),
        "weights": w.to_dict(),
        "raw_full": metrics(raw.loc[IS_START:], "RAW"),
        "full": metrics(net.loc[IS_START:], "FULL"),
        "is": metrics(net.loc[IS_START:IS_END], "IS"),
        "oos": metrics(net.loc[OOS_START:], "OOS"),
        "ret": net,
    }


def fmt(m):
    return (f"Sh={m['sharpe']:.2f} CAGR={m['cagr']*100:5.1f}% "
            f"MDD={m['mdd']*100:6.1f}% Calmar={m.get('calmar', 0):.2f}")


def report():
    base = get_stock_universe_base()
    extended, added = get_stock_universe_extended()
    print(f"BASE universe: {len(base)} large-caps")
    print(f"ADDED via extended/: {len(added)} new names")
    print(f"EXTENDED universe: {len(extended)} total")
    print()

    print("=== PURE: top-3 stocks 126d weekly ===")
    for univ, label in [(base, "BASE"), (extended, "EXT")]:
        r, _ = run_pure(univ, label)
        print(f"  {label}: n={r['univ_n']}")
        print(f"    RAW   {fmt(r['raw_full'])}")
        print(f"    FULL  {fmt(r['full'])}")
        print(f"    IS    {fmt(r['is'])}")
        print(f"    OOS   {fmt(r['oos'])}")

    print("\n=== COMPOSITE: 3 stock sleeves (3-W, 5-W, 7-M) + 2 ETF sleeves ===")
    results = {}
    for univ, label in [(base, "BASE"), (extended, "EXT")]:
        r = run_composite(univ, label)
        results[label] = r
        print(f"  {label}: stock_n={r['univ_n']}")
        print(f"    Weights: {dict((k, round(v,3)) for k,v in r['weights'].items())}")
        print(f"    RAW   {fmt(r['raw_full'])}")
        print(f"    FULL  {fmt(r['full'])}")
        print(f"    IS    {fmt(r['is'])}")
        print(f"    OOS   {fmt(r['oos'])}")

    # Save side-by-side metrics
    base_pure, _ = run_pure(base, "BASE")
    ext_pure, _ = run_pure(extended, "EXT")
    base_comp = run_composite(base, "BASE")
    ext_comp = run_composite(extended, "EXT")
    out = {
        "base_universe_size": len(base),
        "extended_universe_size": len(extended),
        "added_count": len(added),
        "added_tickers": added,
        "PURE": {
            "BASE": {k: v for k, v in base_pure.items() if k != "ret"},
            "EXT":  {k: v for k, v in ext_pure.items() if k != "ret"},
        },
        "COMPOSITE": {
            "BASE": {k: v for k, v in base_comp.items() if k != "ret"},
            "EXT":  {k: v for k, v in ext_comp.items() if k != "ret"},
        },
    }
    RES.mkdir(exist_ok=True)
    with open(RES / "meridian_extended_universe_test.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {RES / 'meridian_extended_universe_test.json'}")


if __name__ == "__main__":
    report()
