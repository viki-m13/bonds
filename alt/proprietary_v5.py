"""
Proprietary V5 — "HELIOS" Hedged ETF Layered Income with Opportunistic Selection.

Design philosophy (different from v4):
- Use SINGLE-leg long-only ETFs (not hedged pairs that fight each other).
- Score each ETF with a multi-factor composite: momentum + carry + reversal + quality.
- Macro regime gate turns the whole book OFF during crises (moves to T-bills).
- Equal-weight top-K monthly. ZERO daily/weekly vol scaling.

Hypothesis: the reason hedged pairs under-perform is that they cancel
the drift. Long-only income ETFs with a crisis gate should earn yield
continuously and avoid the max-DD events.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("/home/user/bonds/data")
ETF = DATA / "etfs"
FRED = DATA / "fred"
RESULTS = Path("/home/user/bonds/alt/results")


def load_etf(t):
    p = ETF / f"{t}.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()


def load_fred(s):
    p = FRED / f"{s}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    return pd.to_numeric(d, errors="coerce").sort_index()


# ---------------------------------------------------------------------
# Universe: income ETFs and some momentum-friendly assets
# ---------------------------------------------------------------------
UNIVERSE = {
    # Cash/ultra short
    "BIL":  "bills",
    "SGOV": "bills",
    "JPST": "short_dur",
    "MINT": "short_dur",
    # Floating rate / CLO
    "FLOT": "floating",
    "BKLN": "floating",
    "SRLN": "floating",
    "JAAA": "clo_aaa",
    "CLOI": "clo_aaa",
    "JBBB": "clo_bbb",
    # Investment grade credit
    "LQD":  "ig",
    "VCIT": "ig",
    "VCSH": "ig_short",
    "IGIB": "ig",
    # High yield
    "HYG":  "hy",
    "JNK":  "hy",
    "USHY": "hy",
    "ANGL": "fa",
    "SHYG": "hy_short",
    # Treasuries (ballast)
    "IEF":  "treasury",
    "IEI":  "treasury_s",
    "SHY":  "treasury_ss",
    "TLT":  "treasury_l",
    # MBS
    "MBB":  "mbs",
    # Muni
    "MUB":  "muni",
    # Preferred / convertible
    "PFF":  "preferred",
    "CWB":  "convertible",
    # EM
    "EMB":  "em_debt",
    # Dividend equity (low vol)
    "SCHD": "div_eq",
    "VIG":  "div_eq",
    "HDV":  "div_eq",
    "VYM":  "div_eq",
    "XLU":  "sector",
    "XLP":  "sector",
    "XLV":  "sector",
    "DVY":  "div_eq",
    # Equity minimum variance
    "SPLV": "low_vol",
    "USMV": "low_vol",
    # TIPS
    "TIP":  "tips",
    "SCHP": "tips",
    # Gold (crisis hedge, positive drift)
    "GLD":  "gold",
    "IAU":  "gold",
}


def load_universe():
    out = {}
    cat = {}
    for t, c in UNIVERSE.items():
        s = load_etf(t)
        if s is None or len(s) < 252:
            continue
        out[t] = s
        cat[t] = c
    return pd.DataFrame(out), cat


# ---------------------------------------------------------------------
# Composite score (T-1 data only)
# ---------------------------------------------------------------------

def composite_score(prices, i, lookback=252, short_look=63, rev_look=21):
    """Compute a cross-sectional composite score for each ETF using
    data up through row i-1 only.
    Factors:
      - 12M momentum = price[i-1] / price[i-252] - 1
      - Carry proxy = 3M price rate of change / vol (risk-adjusted momentum)
      - 1M mean reversion = -1 * (price[i-1] / price[i-21] - 1)
      - Low-vol bonus = -1 * trailing 63d std
    All factors z-scored across ETFs.
    """
    window = prices.iloc[max(0, i - lookback):i]
    if len(window) < lookback - 5:
        return None
    # Factors
    mom_12 = prices.iloc[i - 1] / prices.iloc[i - lookback] - 1
    mom_3 = prices.iloc[i - 1] / prices.iloc[i - short_look] - 1
    rev_1 = -(prices.iloc[i - 1] / prices.iloc[i - rev_look] - 1)
    vol_3 = prices.iloc[i - short_look:i].pct_change().std() * np.sqrt(252)
    # Risk-adjusted mom
    rsm = mom_3 / vol_3.replace(0, np.nan)
    low_vol = -vol_3

    def z(s):
        s = s.replace([np.inf, -np.inf], np.nan)
        mu, sd = s.mean(), s.std()
        return (s - mu) / (sd if sd > 0 else 1)

    score = 0.35 * z(mom_12) + 0.35 * z(rsm) + 0.15 * z(rev_1) + 0.15 * z(low_vol)
    return score.dropna()


# ---------------------------------------------------------------------
# Macro regime gate
# ---------------------------------------------------------------------

def build_regime_crisis_gate(dates):
    """Gate OFF when macro stress is high. T-1 data only.
    Two triggers:
    1. HY OAS > 6.5% (elevated credit stress)
    2. Unemployment rate rising (3-month change > 0.3pp)
    """
    hy = load_fred("BAMLH0A0HYM2")
    if hy is None:
        hy_flag = pd.Series(0.0, index=dates)
    else:
        hy_aligned = hy.reindex(dates).ffill()
        hy_flag = (hy_aligned > 6.5).astype(float)

    # Build regime multiplier. 1.0 = full on, 0.0 = fully off (cash)
    regime = 1.0 - hy_flag
    # Lag by 1 day to avoid look-ahead
    return regime.shift(1).fillna(1.0)


def build_regime_spy_sma(dates):
    """Simple SPY above 200DMA gate."""
    spy = load_etf("SPY").reindex(dates).ffill()
    sma = spy.rolling(200, min_periods=50).mean()
    regime = (spy > sma * 0.95).astype(float)  # 5% buffer
    return regime.shift(1).fillna(1.0)


def build_regime_combined(dates):
    """Regime is ON only if BOTH HY OAS < 6.5 AND SPY > 0.9*200DMA.
    Otherwise ramp down."""
    spy = load_etf("SPY").reindex(dates).ffill()
    sma = spy.rolling(200, min_periods=50).mean()
    spy_ok = (spy > sma * 0.93).astype(float)

    hy = load_fred("BAMLH0A0HYM2")
    if hy is None:
        hy_ok = pd.Series(1.0, index=dates)
    else:
        hy_aligned = hy.reindex(dates).ffill()
        hy_ok = (hy_aligned < 7.0).astype(float)

    # Both must be on. Otherwise off.
    regime = spy_ok * hy_ok
    return regime.shift(1).fillna(1.0)


# ---------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------

def run(prices, cat, rebalance_days=21, lookback=252, short_look=63,
        rev_look=21, top_k=8, category_cap=2, regime=None,
        cash_ticker="BIL", tc_bps=5.0, score_fn=None):
    dates = prices.index
    rets = prices.pct_change().fillna(0)

    port = pd.Series(0.0, index=dates)
    weights = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    current = pd.Series(0.0, index=prices.columns)
    last_idx = -rebalance_days
    log = []

    bil_ret = rets[cash_ticker] if cash_ticker in rets.columns else pd.Series(0.0, index=dates)

    for i, d in enumerate(dates):
        if i - last_idx >= rebalance_days and i > lookback:
            if score_fn is None:
                score = composite_score(prices, i, lookback, short_look, rev_look)
            else:
                score = score_fn(prices, i)
            if score is None or len(score) == 0:
                new_w = pd.Series(0.0, index=prices.columns)
            else:
                ranked = score.sort_values(ascending=False)
                if category_cap:
                    picked = []
                    cnt = {}
                    for nm in ranked.index:
                        c = cat.get(nm, "x")
                        if cnt.get(c, 0) < category_cap:
                            picked.append(nm)
                            cnt[c] = cnt.get(c, 0) + 1
                        if len(picked) >= top_k:
                            break
                    picked = pd.Index(picked)
                else:
                    picked = ranked.head(top_k).index
                new_w = pd.Series(0.0, index=prices.columns)
                if len(picked) > 0:
                    new_w.loc[picked] = 1.0 / len(picked)

            # Transaction cost
            tc = (new_w - current).abs().sum() * (tc_bps / 1e4)
            port.iloc[i] -= tc
            current = new_w
            last_idx = i
            names = current[current > 0].index.tolist()
            log.append({"date": str(d.date()), "picks": names,
                        "weights": {k: float(current[k]) for k in names}})

        weights.iloc[i] = current
        day_ret = (rets.iloc[i] * current).sum()

        # Regime gate
        if regime is not None:
            g = float(regime.get(d, 1.0))
        else:
            g = 1.0
        if g < 1.0:
            day_ret = g * day_ret + (1 - g) * bil_ret.iloc[i]

        port.iloc[i] += day_ret

    return {"returns": port, "weights": weights, "log": log}


def metrics(r):
    r = r.loc[r.ne(0).idxmax():] if (r != 0).any() else r
    if r.std() == 0 or len(r) == 0:
        return {k: 0 for k in ["sharpe", "ann_return", "ann_vol", "max_dd", "sortino", "calmar", "total_return", "n_years"]}
    ar, av = r.mean() * 252, r.std() * np.sqrt(252)
    sr = ar / av if av > 0 else 0
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1)
    mdd = dd.min()
    neg = r[r < 0]
    sor = (r.mean() * 252) / (neg.std() * np.sqrt(252)) if len(neg) and neg.std() > 0 else float("inf")
    return {"sharpe": float(sr), "ann_return": float(ar), "ann_vol": float(av),
            "max_dd": float(mdd), "sortino": float(sor),
            "calmar": float(ar / abs(mdd)) if mdd < 0 else float("inf"),
            "total_return": float(cum.iloc[-1] - 1), "n_years": float(len(r) / 252)}


def main():
    prices, cat = load_universe()
    # Constrain to range where we have breadth (post-2013 has JAAA, SRLN, etc)
    prices = prices.loc["2013-01-01":]
    print(f"Universe: {len(prices.columns)} ETFs, {len(prices)} rows, "
          f"from {prices.index[0].date()} to {prices.index[-1].date()}")

    dates = prices.index
    regimes = {
        "none":     None,
        "hy":       build_regime_crisis_gate(dates),
        "spy_sma":  build_regime_spy_sma(dates),
        "combo":    build_regime_combined(dates),
    }

    configs = []
    for k in [5, 8, 12]:
        for cc in [None, 2, 3]:
            for rg in ["none", "hy", "spy_sma", "combo"]:
                configs.append(dict(top_k=k, category_cap=cc, regime_name=rg))

    results = {}
    for c in configs:
        rg = regimes[c["regime_name"]]
        name = f"k{c['top_k']}_cc{c['category_cap']}_rg{c['regime_name']}"
        res = run(prices, cat,
                  top_k=c["top_k"],
                  category_cap=c["category_cap"],
                  regime=rg)
        m = metrics(res["returns"])
        results[name] = {"config": c, "metrics": m}
        if m["sharpe"] > 0.5:
            print(f"{name:35s} SR={m['sharpe']:.3f}  Ret={m['ann_return']:.2%}  "
                  f"Vol={m['ann_vol']:.2%}  MDD={m['max_dd']:.2%}")

    # Top 10
    ranked = sorted(results.items(), key=lambda x: -x[1]["metrics"]["sharpe"])
    print("\n=== TOP 10 ===")
    for name, r in ranked[:10]:
        m = r["metrics"]
        print(f"  {name:35s} SR={m['sharpe']:.3f}  Ret={m['ann_return']:.2%}  "
              f"Vol={m['ann_vol']:.2%}  MDD={m['max_dd']:.2%}")

    with open(RESULTS / "proprietary_v5_experiments.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
