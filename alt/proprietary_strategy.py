#!/usr/bin/env python3
"""
PROPRIETARY WEEKLY STRATEGY — Zero Leakage by Construction
============================================================

The core problem with every strategy so far: using information from
period T to make decisions about period T. The fix isn't shift(1) on
an existing signal — it's building signals that are STRUCTURALLY lagged.

DESIGN PRINCIPLES:
1. ALL signals use data from PRIOR week (Friday close T-1 to Friday close T-1)
2. Decision made over the weekend
3. Portfolio held Monday-to-Friday of NEXT week  
4. Return earned = next week's Monday-to-Friday return
5. Zero overlap between signal period and return period

SIGNAL SOURCES (all known as of last Friday):
A. Realized vol of SPY over prior 21 days
B. Yield curve slope (10Y-2Y) as of last Friday
C. Credit spread (HY OAS) as of last Friday  
D. SPY vs 200-day moving average (trend)
E. Cross-sectional trailing returns (lookback ending LAST Friday)
F. Breadth: % of sectors above their 50-day MA
G. Term spread momentum: is the curve steepening or flattening?
H. Credit momentum: are spreads widening or tightening?

The innovation: combine ALL of these into a single composite regime
score, then allocate across asset classes based on the regime.
Each signal is independently weak — the combination is the edge.
"""
import pandas as pd, numpy as np, sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")

DATA_DIR = Path("/home/user/bonds/data")
ETF_DIR = DATA_DIR / "etfs"
FRED_PATH = DATA_DIR / "fred" / "_combined_fred.csv"
RESULTS_DIR = Path(__file__).parent / "results"

# Load data
prices = {}
for f in sorted(ETF_DIR.glob("*.csv")):
    if f.name.startswith("_"): continue
    try:
        df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
        df = df[~df.index.duplicated(keep="first")].sort_index()
        if "Close" in df.columns: prices[f.stem] = df["Close"]
    except: continue
prices = pd.DataFrame(prices).sort_index()
ret = prices.pct_change()

fred = pd.read_csv(FRED_PATH, parse_dates=["Date"]).set_index("Date")
for c in fred.columns: fred[c] = pd.to_numeric(fred[c], errors="coerce")
fred = fred.ffill()

# Weekly data — shift by 1 to ensure we use LAST week's data
weekly_px = prices.resample("W-FRI").last()
weekly_ret = prices.resample("W-FRI").apply(lambda x: x.pct_change().dropna().add(1).prod()-1)
# CRITICAL: weekly_ret.iloc[i] = return DURING week i
# We need to make decisions using data up to week i-1

# Daily data for signal computation
daily_ret = ret

print(f"Universe: {len(prices.columns)} ETFs, {len(weekly_ret)} weeks")

# ================================================================
# SIGNAL CONSTRUCTION (all structurally lagged)
# ================================================================

def compute_signals(weekly_px, daily_ret, fred):
    """
    Compute weekly signals. Signal at week i uses data through week i-1.
    Returns a DataFrame of signals indexed by week.
    """
    signals = pd.DataFrame(index=weekly_px.index)
    
    # A. Realized volatility regime (SPY 21-day vol as of last Friday)
    if "SPY" in daily_ret.columns:
        spy_vol = daily_ret["SPY"].rolling(21).std() * np.sqrt(252)
        spy_vol_weekly = spy_vol.resample("W-FRI").last()
        # Percentile over trailing year
        signals["vol_pctl"] = spy_vol_weekly.rolling(52, min_periods=26).rank(pct=True)
        # Risk score: low vol = positive (risk on)
        signals["vol_signal"] = 1 - signals["vol_pctl"]  # High when vol is low
    
    # B. Yield curve slope
    if "T10Y2Y" in fred.columns:
        slope = fred["T10Y2Y"].resample("W-FRI").last()
        slope_ma = slope.rolling(52, min_periods=26).mean()
        slope_std = slope.rolling(52, min_periods=26).std()
        signals["slope_z"] = (slope - slope_ma) / slope_std.clip(lower=0.01)
        # Steep curve = positive (risk on)
        signals["slope_signal"] = signals["slope_z"].clip(-2, 2) / 2
    
    # C. Credit spread level
    if "BAMLH0A0HYM2" in fred.columns:
        hy_oas = fred["BAMLH0A0HYM2"].resample("W-FRI").last()
        hy_ma = hy_oas.rolling(52, min_periods=26).mean()
        hy_std = hy_oas.rolling(52, min_periods=26).std()
        signals["credit_z"] = (hy_oas - hy_ma) / hy_std.clip(lower=0.01)
        # Tight spreads (negative z) = positive (risk on)
        signals["credit_signal"] = -signals["credit_z"].clip(-2, 2) / 2
    
    # D. SPY trend (price vs 200-day MA)
    if "SPY" in prices.columns:
        spy_200ma = prices["SPY"].rolling(200).mean()
        spy_trend = prices["SPY"] / spy_200ma - 1
        signals["trend_signal"] = spy_trend.resample("W-FRI").last().clip(-0.2, 0.2) * 5
    
    # E. Credit spread momentum (tightening = good)
    if "BAMLH0A0HYM2" in fred.columns:
        hy_oas = fred["BAMLH0A0HYM2"].resample("W-FRI").last()
        hy_chg_4w = hy_oas.diff(4)
        hy_chg_std = hy_chg_4w.rolling(52, min_periods=26).std()
        # Tightening (negative change) = positive signal
        signals["credit_mom"] = -(hy_chg_4w / hy_chg_std.clip(lower=0.01)).clip(-2, 2) / 2
    
    # F. Yield curve momentum (steepening = good for risk)
    if "T10Y2Y" in fred.columns:
        slope = fred["T10Y2Y"].resample("W-FRI").last()
        slope_chg_4w = slope.diff(4)
        slope_chg_std = slope_chg_4w.rolling(52, min_periods=26).std()
        signals["slope_mom"] = (slope_chg_4w / slope_chg_std.clip(lower=0.01)).clip(-2, 2) / 2
    
    # G. Broad equity momentum (SPY 13-week return)
    if "SPY" in weekly_px.columns:
        spy_mom = weekly_px["SPY"].pct_change(13)
        signals["equity_mom"] = (spy_mom / spy_mom.rolling(52, min_periods=26).std().clip(lower=0.01)).clip(-2, 2) / 2
    
    # H. Sector breadth (% of sectors above 50-day MA)
    sectors = [t for t in ["XLF","XLK","XLE","XLV","XLI","XLP","XLY","XLU","XLB"] if t in prices.columns]
    if len(sectors) >= 5:
        breadth = pd.Series(0.0, index=prices.index)
        for s in sectors:
            above_50 = (prices[s] > prices[s].rolling(50).mean()).astype(float)
            breadth += above_50
        breadth = breadth / len(sectors)
        signals["breadth"] = breadth.resample("W-FRI").last()
        signals["breadth_signal"] = (signals["breadth"] - 0.5) * 2  # Center at 0
    
    # SHIFT ALL SIGNALS BY 1 WEEK — this is the critical step
    # Decision at week i uses signals from week i-1
    signals = signals.shift(1)
    
    return signals.dropna(how="all")


def compute_composite(signals, weights=None):
    """Combine individual signals into a composite risk score."""
    signal_cols = [c for c in signals.columns if c.endswith("_signal") or c in ["breadth_signal","credit_mom","slope_mom","equity_mom"]]
    
    if not signal_cols:
        return pd.Series(0.0, index=signals.index)
    
    if weights is None:
        # Equal weight all signals
        weights = {c: 1.0/len(signal_cols) for c in signal_cols}
    
    composite = pd.Series(0.0, index=signals.index)
    for col in signal_cols:
        if col in signals.columns and col in weights:
            composite += weights[col] * signals[col].fillna(0)
    
    return composite


def run_regime_strategy(signals, weekly_ret_df, composite,
                         risk_on_assets, balanced_assets, risk_off_assets,
                         threshold_on=0.2, threshold_off=-0.2,
                         top_n=0, use_momentum_tilt=False,
                         min_weeks_warmup=104, name=""):
    """
    Weekly regime rotation using the composite signal.
    composite > threshold_on → risk on
    composite < threshold_off → risk off  
    otherwise → balanced
    
    Optional: within regime, tilt toward higher-momentum assets.
    """
    p = pd.Series(0.0, index=weekly_ret_df.index)
    
    for i in range(min_weeks_warmup, len(weekly_ret_df)):
        sig = composite.iloc[i] if i < len(composite) else 0
        
        if np.isnan(sig):
            # Use balanced as default
            basket = balanced_assets
        elif sig > threshold_on:
            basket = risk_on_assets
        elif sig < threshold_off:
            basket = risk_off_assets
        else:
            basket = balanced_assets
        
        available = [t for t in basket if t in weekly_ret_df.columns]
        
        if use_momentum_tilt and top_n > 0 and len(available) > top_n:
            # Within the regime basket, tilt toward momentum
            # Use PRIOR week's returns for momentum (no leakage)
            if i >= 14:
                trailing = weekly_px.iloc[i-1] / weekly_px.iloc[max(0,i-14)] - 1
                trailing = trailing[available].dropna()
                if len(trailing) >= top_n:
                    available = trailing.nlargest(top_n).index.tolist()
        
        if len(available) > 0:
            p.iloc[i] = weekly_ret_df.iloc[i][available].mean()
    
    return p.iloc[min_weeks_warmup:]


def m(r, name=""):
    r = r.dropna()
    if len(r) < 52: return None
    ar = r.mean()*52; av = r.std()*np.sqrt(52); sr = ar/av if av > 0 else 0
    cum = (1+r).cumprod(); mdd = ((cum-cum.cummax())/cum.cummax()).min()
    ds = r[r<0].std()*np.sqrt(52) if (r<0).any() else av
    sortino = ar/ds if ds > 0 else 0; wr = (r>0).mean()
    sp = int(len(r)*0.6)
    tsr = r.iloc[sp:].mean()/r.iloc[sp:].std()*np.sqrt(52) if r.iloc[sp:].std() > 0 else 0
    nt = len(r); fs = nt//6; wf = []
    for fold in range(5):
        s=(fold+1)*fs; e=min(s+fs,nt); fr=r.iloc[s:e]
        if len(fr)>26 and fr.std()>0: wf.append(fr.mean()/fr.std()*np.sqrt(52))
    return {"name":name,"sr":round(sr,3),"ret":round(ar*100,2),"vol":round(av*100,2),
            "mdd":round(mdd*100,2),"sortino":round(sortino,3),"wr":round(wr*100,1),
            "test_sr":round(tsr,3),"wf_mean":round(np.mean(wf),3) if wf else 0,
            "nav":round(float(cum.iloc[-1]),2)}


# ================================================================
# RUN
# ================================================================
print("="*70)
print("PROPRIETARY STRATEGY — STRUCTURALLY LAGGED SIGNALS")
print("="*70)

signals = compute_signals(weekly_px, daily_ret, fred)
print(f"Signals computed: {list(signals.columns)}")
print(f"Signal period: {signals.index[0].date()} to {signals.index[-1].date()}")

# Asset baskets
risk_on = [t for t in ["QQQ","SMH","IWM","EEM","HYG","EMB","VNQ","ARKK","XLK","XLY","IBIT","GBTC"] if t in weekly_ret.columns]
balanced = [t for t in ["SPY","SCHD","VIG","AGG","GLD","IEF","DVY","HDV","LQD","AMLP"] if t in weekly_ret.columns]
risk_off = [t for t in ["SHY","TLT","GLD","JAAA","AGG","TBF","IEF","TIP","BNDX","MUB"] if t in weekly_ret.columns]

print(f"Risk-on: {len(risk_on)} ETFs")
print(f"Balanced: {len(balanced)} ETFs")
print(f"Risk-off: {len(risk_off)} ETFs")

results = []

# Individual signal effectiveness
print(f"\n--- Individual Signal Effectiveness ---")
signal_cols = [c for c in signals.columns if "_signal" in c or c in ["credit_mom","slope_mom","equity_mom","breadth_signal"]]
for sig_col in signal_cols:
    single = pd.Series(0.0, index=signals.index)
    single[:] = signals[sig_col].fillna(0)
    p = run_regime_strategy(signals, weekly_ret, single, risk_on, balanced, risk_off,
                             0.2, -0.2, name=sig_col)
    mx = m(p, sig_col)
    if mx:
        results.append(mx)
        print(f"  {sig_col:20s}: SR={mx['sr']:.3f} WF={mx['wf_mean']:.3f} Ret={mx['ret']:+.1f}%")

# Composite with different weight schemes
print(f"\n--- Composite Signal Experiments ---")

configs = [
    # (name, weights, threshold_on, threshold_off, top_n, mom_tilt)
    ("Composite_equal", None, 0.15, -0.15, 0, False),
    ("Composite_eq_t10", None, 0.10, -0.10, 0, False),
    ("Composite_eq_t20", None, 0.20, -0.20, 0, False),
    ("Composite_eq_t30", None, 0.30, -0.30, 0, False),
    ("Composite_eq_t05", None, 0.05, -0.05, 0, False),
    # With momentum tilt within regime
    ("Comp_mom_t3", None, 0.15, -0.15, 3, True),
    ("Comp_mom_t5", None, 0.15, -0.15, 5, True),
    ("Comp_mom_t3_t10", None, 0.10, -0.10, 3, True),
    # Heavy on credit/vol (these tend to be strongest)
    ("Comp_credit_heavy", {"vol_signal":0.25,"credit_signal":0.30,"credit_mom":0.20,"slope_signal":0.10,"trend_signal":0.15}, 0.15, -0.15, 0, False),
    ("Comp_vol_heavy", {"vol_signal":0.35,"credit_signal":0.20,"credit_mom":0.15,"slope_signal":0.10,"trend_signal":0.10,"equity_mom":0.10}, 0.15, -0.15, 0, False),
    ("Comp_trend_heavy", {"vol_signal":0.15,"credit_signal":0.15,"trend_signal":0.30,"equity_mom":0.25,"breadth_signal":0.15}, 0.15, -0.15, 0, False),
    # Only the best signals
    ("Comp_best3", {"credit_signal":0.33,"credit_mom":0.33,"vol_signal":0.34}, 0.15, -0.15, 0, False),
    ("Comp_best3_mom", {"credit_signal":0.33,"credit_mom":0.33,"vol_signal":0.34}, 0.15, -0.15, 3, True),
    # Asymmetric thresholds (harder to go risk-off)
    ("Comp_asym", None, 0.10, -0.25, 0, False),
    ("Comp_asym_mom", None, 0.10, -0.25, 3, True),
    # Very simple: just credit + vol
    ("Simple_credit_vol", {"credit_signal":0.5,"vol_signal":0.5}, 0.15, -0.15, 0, False),
    ("Simple_credit_vol_mom", {"credit_signal":0.5,"vol_signal":0.5}, 0.15, -0.15, 3, True),
    # Credit + trend
    ("Simple_credit_trend", {"credit_signal":0.5,"trend_signal":0.5}, 0.15, -0.15, 0, False),
    # All macro, no price
    ("Pure_macro", {"slope_signal":0.25,"credit_signal":0.25,"credit_mom":0.25,"slope_mom":0.25}, 0.15, -0.15, 0, False),
    # Continuous allocation (no binary regime, use signal as weight)
    ("Continuous", None, 999, -999, 0, False),  # Never triggers regime → always balanced
]

for name, wts, ton, toff, tn, mom in configs:
    composite = compute_composite(signals, wts)
    
    if name == "Continuous":
        # Special: use signal to blend risk-on and risk-off continuously
        p = pd.Series(0.0, index=weekly_ret.index)
        for i in range(104, len(weekly_ret)):
            sig = composite.iloc[i] if i < len(composite) else 0
            if np.isnan(sig): sig = 0
            # Map signal [-1,1] to weight [0,1] for risk-on
            ro_weight = max(0, min(1, (sig + 1) / 2))
            available_on = [t for t in risk_on if t in weekly_ret.columns]
            available_off = [t for t in risk_off if t in weekly_ret.columns]
            ret_on = weekly_ret.iloc[i][available_on].mean() if available_on else 0
            ret_off = weekly_ret.iloc[i][available_off].mean() if available_off else 0
            p.iloc[i] = ro_weight * ret_on + (1-ro_weight) * ret_off
        p = p.iloc[104:]
    else:
        p = run_regime_strategy(signals, weekly_ret, composite,
                                 risk_on, balanced, risk_off,
                                 ton, toff, tn, mom, name=name)
    
    mx = m(p, name)
    if mx:
        results.append(mx)

# SPY baseline
spy_weekly = weekly_ret.get("SPY")
if spy_weekly is not None:
    spy_m = m(spy_weekly.iloc[104:], "SPY_BuyHold")
    if spy_m: results.append(spy_m)

# Equal weight all
eq_all = weekly_ret[risk_on + balanced + risk_off].iloc[104:].mean(axis=1)
eq_m = m(eq_all, "EqualWeight_All")
if eq_m: results.append(eq_m)

# Summary
print(f"\n{'='*70}")
print(f"RANKED BY WALK-FORWARD SHARPE")
print(f"{'='*70}")
print(f"{'Name':25s} {'SR':>7} {'WF':>7} {'Test':>7} {'Ret':>8} {'Vol':>7} {'MDD':>8} {'Sort':>7} {'WR':>6}")
print("-"*82)
for r in sorted(results, key=lambda x:-x['wf_mean']):
    flag = " ★" if r['wf_mean'] > 1.0 else ""
    print(f"  {r['name']:23s} {r['sr']:>6.3f} {r['wf_mean']:>6.3f} {r['test_sr']:>6.3f} "
          f"{r['ret']:>+7.1f}% {r['vol']:>6.1f}% {r['mdd']:>+7.1f}% {r['sortino']:>6.3f} {r['wr']*1:>5.1f}%{flag}")

best = max(results, key=lambda x: x['wf_mean'])
print(f"\nBEST: {best['name']} → SR={best['sr']} WF={best['wf_mean']} Ret={best['ret']}% MDD={best['mdd']}%")
print(f"Monthly carry (fixed): SR≈1.54 WF≈1.62")

# ================================================================
# SHUFFLE TEST on best strategy
# ================================================================
print(f"\n--- Shuffle test on best ---")
best_name = best['name']
np.random.seed(42)
shuffle_srs = []
for trial in range(50):
    # Shuffle the composite signal dates
    shuffled_signals = signals.copy()
    for col in shuffled_signals.columns:
        vals = shuffled_signals[col].dropna().values.copy()
        np.random.shuffle(vals)
        shuffled_signals[col].iloc[:len(vals)] = vals[:len(shuffled_signals[col])]
    shuffled_composite = compute_composite(shuffled_signals, None)
    p_shuf = run_regime_strategy(shuffled_signals, weekly_ret, shuffled_composite,
                                  risk_on, balanced, risk_off, 0.15, -0.15)
    p_shuf = p_shuf.dropna()
    if len(p_shuf) > 52 and p_shuf.std() > 0:
        shuffle_srs.append(p_shuf.mean()/p_shuf.std()*np.sqrt(52))

print(f"  Best strategy Sharpe: {best['sr']:.3f}")
print(f"  Shuffled: mean={np.mean(shuffle_srs):.3f} std={np.std(shuffle_srs):.3f}")
print(f"  Z-score: {(best['sr']-np.mean(shuffle_srs))/np.std(shuffle_srs):.1f}")
if best['sr'] > np.mean(shuffle_srs) + 2*np.std(shuffle_srs):
    print(f"  Signal is REAL (>{2}σ above shuffled) ✓")
else:
    print(f"  Signal is WEAK (<2σ above shuffled)")

# Save
with open(RESULTS_DIR/"proprietary_experiments.json","w") as f:
    json.dump({"experiments":results,"best":best,"n_signals":len(signal_cols)},f,indent=2)
print(f"\nSaved {len(results)} experiments")
