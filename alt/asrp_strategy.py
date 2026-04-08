#!/usr/bin/env python3
"""
ASRP — Adaptive Sector Risk Parity (recreated from CRT repo)
==============================================================

Components:
1. Regime Detection: SPY vs 100-day SMA → bull/bear
2. Multi-Factor Stock Selection: momentum, quality, persistence, trend
3. Safe Haven Allocation: TLT/GLD/IEF with correlation-adaptive weights
4. Monthly rebalance, T+1 open execution, 5bps slippage

Adapted to use our bonds repo ETF data (no individual stocks —
use sector ETFs + dividend ETFs as the equity sleeve).
"""
import pandas as pd, numpy as np, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")

DATA_DIR = Path("/home/user/bonds/data")
ETF_DIR = DATA_DIR / "etfs"
RESULTS_DIR = Path("/home/user/bonds/alt/results")

# Load prices
prices = {}
for f in sorted(ETF_DIR.glob("*.csv")):
    if f.name.startswith("_"): continue
    try:
        df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
        df = df[~df.index.duplicated(keep="first")].sort_index()
        prices[f.stem] = df
    except: continue

# Universe
EQUITY_UNIVERSE = [t for t in [
    "XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLB","XLRE","XLC",
    "SCHD","HDV","DVY","VIG","QQQ","IWM","SMH","ARKK","KWEB",
    "EFA","EEM","EWJ","AMLP","VNQ","IYR","DIA","MDY",
    "SPY",  # SPY can also be a holding
] if t in prices]

SAFE_HAVENS = [t for t in ["TLT","GLD","IEF"] if t in prices]
BENCHMARK = "SPY"

# Constants (from CRT repo — standard, non-optimized)
SMA_REGIME = 100
SMA_TREND = 200
VOL_LOOKBACK = 63
MOM_LOOKBACK = 252
MOM_SKIP = 21
QUALITY_LOOKBACK = 63
N_STOCKS_BULL = 10  # Fewer than CRT's 30 since we have ETFs not stocks
N_STOCKS_BEAR = 5
EQ_PCT_BULL = 0.80
EQ_PCT_BEAR = 0.30
SLIPPAGE_BPS = 5

print(f"Equity universe: {len(EQUITY_UNIVERSE)} ETFs")
print(f"Safe havens: {SAFE_HAVENS}")

# Precompute signals
closes = {}; returns = {}; vol63 = {}; mom252 = {}; mom126 = {}
mom63 = {}; mom21 = {}; sma200 = {}; quality = {}; persistence = {}

for t in EQUITY_UNIVERSE + SAFE_HAVENS:
    if t not in prices: continue
    df = prices[t]
    c = df["Close"]
    r = c.pct_change()
    closes[t] = c
    returns[t] = r
    vol63[t] = r.rolling(VOL_LOOKBACK, min_periods=21).std() * np.sqrt(252)
    mom252[t] = c / c.shift(MOM_LOOKBACK) - 1
    mom126[t] = c / c.shift(126) - 1
    mom63[t] = c / c.shift(63) - 1
    mom21[t] = c / c.shift(MOM_SKIP) - 1
    sma200[t] = c.rolling(SMA_TREND).mean()
    m63 = r.rolling(QUALITY_LOOKBACK, min_periods=42).mean() * 252
    s63 = r.rolling(QUALITY_LOOKBACK, min_periods=42).std() * np.sqrt(252)
    quality[t] = (m63 - 0.02) / s63.clip(lower=0.01)
    persistence[t] = r.rolling(QUALITY_LOOKBACK, min_periods=42).apply(lambda x: (x>0).mean(), raw=True)

# SPY regime
spy_close = prices[BENCHMARK]["Close"]
spy_sma = spy_close.rolling(SMA_REGIME).mean()

# SPY-TLT correlation (for adaptive hedging)
spy_ret = spy_close.pct_change()
tlt_ret = prices["TLT"]["Close"].pct_change() if "TLT" in prices else None
spy_tlt_corr = spy_ret.rolling(63).corr(tlt_ret) if tlt_ret is not None else None


def is_bear(date):
    if date in spy_sma.index:
        s = spy_sma.loc[date]
        if not pd.isna(s) and spy_close.loc[date] <= s:
            return True
    return False


def rank_etfs(date, n):
    """Multi-factor ETF ranking (adapted from CRT stock ranking)."""
    scored = []
    for t in EQUITY_UNIVERSE:
        if t not in mom252 or date not in mom252[t].index: continue
        m12 = mom252[t].loc[date]
        m1 = mom21[t].loc[date] if date in mom21[t].index else 0
        q = quality[t].loc[date] if date in quality[t].index else 0
        p = persistence[t].loc[date] if date in persistence[t].index else 0
        v = vol63[t].loc[date] if date in vol63[t].index else 0
        sm = sma200[t].loc[date] if date in sma200[t].index else 0
        price = closes[t].loc[date] if date in closes[t].index else 0

        if pd.isna(m12) or pd.isna(q) or pd.isna(v) or v <= 0.01: continue
        if pd.isna(p): p = 0.5

        # Multi-factor filters
        mom_skip = m12 - (m1 if not pd.isna(m1) else 0)
        if mom_skip <= 0: continue
        if q <= 0: continue
        if not pd.isna(sm) and price > 0 and price <= sm: continue

        # Ensemble momentum
        moms = [mom_skip]
        m63v = mom63[t].loc[date] if date in mom63[t].index else None
        m126v = mom126[t].loc[date] if date in mom126[t].index else None
        if m63v is not None and not pd.isna(m63v): moms.append(m63v)
        if m126v is not None and not pd.isna(m126v): moms.append(m126v)
        avg_mom = np.mean([x for x in moms if x > 0]) if moms else 0
        if avg_mom <= 0: continue

        composite = avg_mom * max(q, 0.01) * max(p, 0.4)
        scored.append((t, composite, 1.0/v))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]


def safe_haven_weights(date, bear=False):
    """Correlation-adaptive safe haven allocation."""
    corr_val = 0
    if spy_tlt_corr is not None and date in spy_tlt_corr.index:
        c = spy_tlt_corr.loc[date]
        if not pd.isna(c): corr_val = c

    if corr_val > 0.2:
        hw = {"GLD": 0.60, "IEF": 0.40}
    elif corr_val < -0.2:
        hw = {"TLT": 0.50, "GLD": 0.25, "IEF": 0.25}
    else:
        hw = {"TLT": 0.33, "GLD": 0.34, "IEF": 0.33}

    # Trend filter in bear regime
    if bear:
        filtered = {}
        for h, w in hw.items():
            if h in sma200 and date in sma200[h].index:
                sma = sma200[h].loc[date]
                price = closes[h].loc[date] if h in closes and date in closes[h].index else 0
                if not pd.isna(sma) and price > 0:
                    filtered[h] = w if price >= sma else w * 0.25
                else:
                    filtered[h] = w
            else:
                filtered[h] = w
        hw = filtered

    total = sum(hw.values())
    if total > 1.0:
        hw = {k: v/total for k, v in hw.items()}
    return hw


def get_weights(date):
    """Compute full portfolio weights."""
    bear = is_bear(date)
    eq_pct = EQ_PCT_BEAR if bear else EQ_PCT_BULL
    hedge_pct = 1.0 - eq_pct
    n = N_STOCKS_BEAR if bear else N_STOCKS_BULL

    top = rank_etfs(date, n)
    weights = {}
    if top:
        total_iv = sum(iv for _, _, iv in top)
        for t, _, iv in top:
            weights[t] = (iv / total_iv) * eq_pct
    else:
        weights["SPY"] = eq_pct

    haven = safe_haven_weights(date, bear)
    for h, w in haven.items():
        weights[h] = weights.get(h, 0) + w * hedge_pct

    return weights


def backtest(start, end, tx_bps=5):
    """Monthly rebalance with T+1 open execution."""
    spy = prices[BENCHMARK]
    dates = spy.loc[start:end].index
    slip = tx_bps / 10000

    daily_rets = []
    current_w = {}
    last_month = None
    trades = 0

    for date in dates:
        idx = spy.index.get_loc(date)
        if idx < 252:
            daily_rets.append(0.0)
            continue

        month = date.month
        rebalance = (last_month is not None and month != last_month)
        last_month = month

        if rebalance:
            new_w = get_weights(date)
            dr = 0.0

            # Exit changed positions at open
            for t, w in current_w.items():
                if t not in new_w or abs(new_w.get(t, 0) - w) > 0.005:
                    df = prices.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            prev_c = df.iloc[si-1]["Close"]
                            today_o = df.loc[date, "Open"] if "Open" in df.columns else prev_c
                            dr += (today_o * (1-slip) / prev_c - 1) * w
                    trades += 1
                else:
                    df = prices.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w

            # Enter new positions at open
            for t, w in new_w.items():
                if t not in current_w or abs(current_w.get(t, 0) - w) > 0.005:
                    df = prices.get(t)
                    if df is not None and date in df.index:
                        today_o = df.loc[date, "Open"] if "Open" in df.columns else df.loc[date, "Close"]
                        buy = today_o * (1+slip)
                        today_c = df.loc[date, "Close"]
                        if buy > 0:
                            dr += (today_c / buy - 1) * w
                    trades += 1

            current_w = new_w
        else:
            dr = 0.0
            for t, w in current_w.items():
                df = prices.get(t)
                if df is not None and date in df.index:
                    si = df.index.get_loc(date)
                    if si > 0:
                        dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w

        daily_rets.append(dr)

    return pd.Series(daily_rets, index=dates), trades


def metrics(rets):
    if len(rets) == 0 or rets.std() == 0: return {}
    excess = rets - 0.02/252
    n_years = len(rets)/252
    sr = excess.mean()/excess.std()*np.sqrt(252)
    cum = (1+rets).cumprod()
    total = cum.iloc[-1]-1
    cagr = (1+total)**(1/n_years)-1 if n_years>=1 else total
    mdd = ((cum-cum.cummax())/cum.cummax()).min()
    ds = excess[excess<0]
    sortino = excess.mean()/ds.std()*np.sqrt(252) if len(ds)>0 and ds.std()>0 else 0
    vol = rets.std()*np.sqrt(252)
    calmar = cagr/abs(mdd) if mdd!=0 else 0
    return {"sharpe":round(sr,3),"cagr":round(cagr*100,2),"max_dd":round(mdd*100,2),
            "sortino":round(sortino,3),"vol":round(vol*100,2),"calmar":round(calmar,3),
            "total_ret":round(total*100,2),"nav":round(float(cum.iloc[-1]),2)}


# Run
print("\n=== ASRP Strategy (ETF version) ===")

periods = [
    ("Full", "2005-01-01", "2026-04-01"),
    ("Train", "2010-01-01", "2019-12-31"),
    ("Validation", "2020-04-01", "2022-12-31"),
    ("Test", "2023-04-01", "2026-04-01"),
]

results = {}
for name, start, end in periods:
    rets, trades = backtest(start, end)
    m = metrics(rets)
    m["trades"] = trades
    m["period"] = f"{start} to {end}"
    results[name] = m
    print(f"\n  {name} ({start} to {end}):")
    print(f"    Sharpe={m['sharpe']}  CAGR={m['cagr']}%  MaxDD={m['max_dd']}%  "
          f"Sortino={m['sortino']}  Vol={m['vol']}%  Trades={trades}")

# Walk-forward
print("\n--- Walk-Forward ---")
wf_results = []
fold_defs = [
    ("2005-01-01","2010-12-31","2011-01-01","2013-12-31"),
    ("2005-01-01","2013-12-31","2014-01-01","2016-12-31"),
    ("2005-01-01","2016-12-31","2017-01-01","2019-12-31"),
    ("2005-01-01","2019-12-31","2020-01-01","2022-12-31"),
    ("2005-01-01","2022-12-31","2023-01-01","2026-04-01"),
]
for i, (ts, te, vs, ve) in enumerate(fold_defs):
    rets, _ = backtest(vs, ve)
    m = metrics(rets)
    wf_results.append(m)
    print(f"  Fold {i+1} ({vs}-{ve}): Sharpe={m['sharpe']} CAGR={m['cagr']}%")

wf_sharpes = [r['sharpe'] for r in wf_results if 'sharpe' in r]
print(f"  WF Mean Sharpe: {np.mean(wf_sharpes):.3f}")

# Compare to our carry strategy
print("\n--- vs Carry Strategy ---")
carry = pd.read_csv(DATA_DIR/"results"/"dichs_returns.csv", parse_dates=[0])
carry.columns=["Date","return"]; carry=carry.set_index("Date")["return"]
carry_m = metrics(carry)
print(f"  Carry (daily scaling): Sharpe={carry_m['sharpe']} CAGR={carry_m['cagr']}%")
print(f"  ASRP (monthly, T+1):   Sharpe={results['Full']['sharpe']} CAGR={results['Full']['cagr']}%")

# SPY benchmark
spy_rets = spy_close.pct_change().loc["2005-01-01":"2026-04-01"]
spy_m = metrics(spy_rets)
print(f"  SPY buy-and-hold:      Sharpe={spy_m['sharpe']} CAGR={spy_m['cagr']}%")

# Save
all_results = {"asrp": results, "walk_forward": wf_results,
               "wf_mean_sharpe": round(np.mean(wf_sharpes),3),
               "carry_comparison": carry_m, "spy_comparison": spy_m}
with open(RESULTS_DIR/"asrp_results.json","w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {RESULTS_DIR}/asrp_results.json")
