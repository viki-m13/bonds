#!/usr/bin/env python3
"""
ALTERNATIVE STRATEGY: Non-carry, weekly rebalance, no daily scaling.
====================================================================

Everything we've done so far is carry-based. What about completely
different alpha sources that don't need daily vol scaling?

IDEAS FOR WEEKLY-ONLY STRATEGIES:

1. CROSS-SECTIONAL MOMENTUM ROTATION
   Each week, rank ALL ETFs by trailing return. Buy top quintile.
   This is Jegadeesh & Titman (1993) — the most documented anomaly.

2. MEAN REVERSION (weekly)
   Buy ETFs that fell the most this week, sell those that rose most.
   Contrarian weekly reversal effect.

3. VOLATILITY RISK PREMIUM
   Each week, allocate inversely to realized vol. Low-vol ETFs get
   more weight. This is the "betting against beta" factor.

4. QUALITY/DEFENSIVE ROTATION
   Rotate between defensive (XLU, XLP, HDV, TLT) and aggressive
   (QQQ, TQQQ, EEM) based on macro signals (VIX, yield curve).

5. CROSS-ASSET MOMENTUM-REVERSAL BLEND
   Combine 12-month momentum with 1-week reversal.
   Well-documented in academic literature.

6. FACTOR TIMING
   Each week, estimate which factor (value, momentum, quality, size)
   is strongest, and rotate into the corresponding ETF.

7. REGIME DETECTION + ROTATION
   Use yield curve slope + VIX + credit spreads to detect regime.
   Different allocation per regime.

ALL WEEKLY: Compute everything on Friday close, trade Monday open.
NO DAILY SCALING. Pure weekly decisions, frozen for 5 days.
"""
import pandas as pd, numpy as np, sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")

DATA_DIR = Path("/home/user/bonds/data")
ETF_DIR = DATA_DIR / "etfs"
FRED_PATH = DATA_DIR / "fred" / "_combined_fred.csv"
RESULTS_DIR = Path(__file__).parent / "results"

def load():
    prices = {}
    for f in sorted(ETF_DIR.glob("*.csv")):
        if f.name.startswith("_"): continue
        try:
            df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
            df = df[~df.index.duplicated(keep="first")].sort_index()
            if "Close" in df.columns: prices[f.stem] = df["Close"]
        except: continue
    prices = pd.DataFrame(prices).sort_index()
    fred = pd.read_csv(FRED_PATH, parse_dates=["Date"]).set_index("Date")
    fred = fred[~fred.index.duplicated(keep="first")].sort_index()
    for c in fred.columns: fred[c] = pd.to_numeric(fred[c], errors="coerce")
    fred = fred.ffill()
    return prices, fred

prices, fred = load()

# Focus on liquid, tradeable ETFs with good history
universe = [t for t in [
    # Equity
    "SPY","QQQ","IWM","DIA","MDY",
    # Sectors
    "XLF","XLK","XLE","XLV","XLI","XLP","XLY","XLU","XLB","XLRE",
    # International
    "EFA","EEM","EWJ","EWZ","FXI","EWG","EWU","EWY","INDA",
    # Bonds
    "TLT","IEF","SHY","LQD","HYG","JNK","AGG","TIP","EMB","MUB","BNDX",
    # Commodities
    "GLD","SLV","DBC","USO","URA",
    # REITs
    "VNQ","IYR",
    # Alternatives
    "AMLP","PFF","BKLN",
    # Dividend
    "SCHD","HDV","DVY","VIG",
    # Thematic
    "SMH","XBI","ARKK","KWEB",
    # Crypto
    "IBIT","GBTC","BITO",
    # Inverse (for hedging)
    "SH","TBF","PSQ",
    # CLO
    "JAAA",
] if t in prices.columns]

print(f"Universe: {len(universe)} ETFs")
px = prices[universe].dropna(how="all")
ret = px.pct_change()

# Weekly returns
weekly_ret = ret.resample("W-FRI").apply(lambda x: (1+x).prod()-1)
weekly_px = px.resample("W-FRI").last()

min_weeks = 104  # 2 years warmup

def m(r, name=""):
    r=r.dropna()
    if len(r)<52: return None
    # Convert weekly returns to annualized
    ar=r.mean()*52; av=r.std()*np.sqrt(52); sr=ar/av if av>0 else 0
    cum=(1+r).cumprod(); mdd=((cum-cum.cummax())/cum.cummax()).min()
    ds=r[r<0].std()*np.sqrt(52) if (r<0).any() else av
    sortino=ar/ds if ds>0 else 0; wr=(r>0).mean()
    sp=int(len(r)*0.6)
    test_sr=r.iloc[sp:].mean()/r.iloc[sp:].std()*np.sqrt(52) if r.iloc[sp:].std()>0 else 0
    nt=len(r);fs=nt//6;wf=[]
    for fold in range(5):
        s=(fold+1)*fs;e=min(s+fs,nt);fr=r.iloc[s:e]
        if len(fr)>26 and fr.std()>0:wf.append(fr.mean()/fr.std()*np.sqrt(52))
    return {"name":name,"sr":round(sr,3),"ret":round(ar*100,2),"vol":round(av*100,2),
            "mdd":round(mdd*100,2),"sortino":round(sortino,3),"wr":round(wr*100,1),
            "test_sr":round(test_sr,3),"wf_mean":round(np.mean(wf),3) if wf else 0,
            "nav":round(float(cum.iloc[-1]),2)}


# ================================================================
# STRATEGY 1: CROSS-SECTIONAL MOMENTUM ROTATION
# ================================================================
def strat_momentum(lookback_weeks=26, holding_weeks=1, top_n=5, 
                    bottom_n=0, skip_recent=1, name=""):
    """
    Each week: rank ETFs by trailing return over lookback_weeks.
    Buy top_n, optionally short bottom_n.
    skip_recent: skip the most recent week (reversal effect).
    """
    p = pd.Series(0.0, index=weekly_ret.index)
    
    for i in range(min_weeks, len(weekly_ret)):
        if i % holding_weeks != 0: 
            p.iloc[i] = p.iloc[i-1] if i > 0 else 0
            continue
        
        # Trailing return (skip recent weeks if specified)
        end = i - skip_recent
        start = max(0, end - lookback_weeks)
        if end <= start: continue
        
        trailing = weekly_px.iloc[end] / weekly_px.iloc[start] - 1
        trailing = trailing.dropna()
        if len(trailing) < top_n: continue
        
        # Rank and pick
        ranked = trailing.sort_values(ascending=False)
        longs = ranked.head(top_n).index
        
        # Equal weight the top N
        long_ret = weekly_ret.iloc[i][longs].mean() if len(longs) > 0 else 0
        
        if bottom_n > 0:
            shorts = ranked.tail(bottom_n).index
            short_ret = -weekly_ret.iloc[i][shorts].mean() if len(shorts) > 0 else 0
            p.iloc[i] = (long_ret + short_ret) / 2
        else:
            p.iloc[i] = long_ret
    
    return p.iloc[min_weeks:]


# ================================================================
# STRATEGY 2: WEEKLY MEAN REVERSION
# ================================================================
def strat_reversal(lookback_weeks=1, top_n=5, name=""):
    """
    Each week: buy the WORST performers from last week.
    Classic short-term reversal.
    """
    p = pd.Series(0.0, index=weekly_ret.index)
    
    for i in range(min_weeks, len(weekly_ret)):
        # Last week's returns
        last_week = weekly_ret.iloc[i-lookback_weeks:i].mean()
        last_week = last_week.dropna()
        if len(last_week) < top_n: continue
        
        # Buy the WORST performers (they should revert)
        worst = last_week.sort_values().head(top_n).index
        p.iloc[i] = weekly_ret.iloc[i][worst].mean()
    
    return p.iloc[min_weeks:]


# ================================================================
# STRATEGY 3: LOW VOLATILITY (Betting Against Beta)
# ================================================================
def strat_low_vol(lookback_weeks=52, top_n=10, name=""):
    """
    Each week: rank ETFs by realized vol. Buy the LOWEST vol ones.
    """
    p = pd.Series(0.0, index=weekly_ret.index)
    
    for i in range(min_weeks, len(weekly_ret)):
        window = weekly_ret.iloc[max(0,i-lookback_weeks):i]
        vols = window.std() * np.sqrt(52)
        vols = vols.dropna()
        vols = vols[vols > 0]
        if len(vols) < top_n: continue
        
        # Lowest vol
        lowest = vols.sort_values().head(top_n).index
        p.iloc[i] = weekly_ret.iloc[i][lowest].mean()
    
    return p.iloc[min_weeks:]


# ================================================================
# STRATEGY 4: MOMENTUM + REVERSAL BLEND
# ================================================================
def strat_mom_rev_blend(mom_lookback=26, rev_lookback=1, top_n=5,
                         mom_weight=0.7, skip_recent=1, name=""):
    """
    Blend 6-month momentum (skip recent week) with 1-week reversal.
    """
    p = pd.Series(0.0, index=weekly_ret.index)
    
    for i in range(min_weeks, len(weekly_ret)):
        # Momentum signal
        end = i - skip_recent
        start = max(0, end - mom_lookback)
        if end <= start: continue
        mom = weekly_px.iloc[end] / weekly_px.iloc[start] - 1
        mom = mom.dropna()
        
        # Reversal signal (negative of last week's return)
        rev = -weekly_ret.iloc[i-rev_lookback:i].mean()
        rev = rev.dropna()
        
        # Combine
        common = mom.index.intersection(rev.index)
        if len(common) < top_n: continue
        
        # Z-score both
        mom_z = (mom[common] - mom[common].mean()) / mom[common].std()
        rev_z = (rev[common] - rev[common].mean()) / rev[common].std()
        
        combined = mom_weight * mom_z + (1-mom_weight) * rev_z
        top = combined.nlargest(top_n).index
        
        p.iloc[i] = weekly_ret.iloc[i][top].mean()
    
    return p.iloc[min_weeks:]


# ================================================================
# STRATEGY 5: VIX REGIME ROTATION
# ================================================================
def strat_regime_rotation(name=""):
    """
    Use VIX regime to rotate between risk-on and risk-off.
    Low VIX: QQQ, SMH, ARKK, EEM, HYG (risk-on)
    Mid VIX: SPY, SCHD, VIG, AGG, GLD (balanced)
    High VIX: SHY, TLT, GLD, JAAA, TBF (risk-off)
    """
    vix = fred.get("VIXCLS")
    if vix is None: return pd.Series(dtype=float)
    
    risk_on = [t for t in ["QQQ","SMH","TQQQ","EEM","HYG","VNQ","ARKK"] if t in weekly_ret.columns]
    balanced = [t for t in ["SPY","SCHD","VIG","AGG","GLD","IEF"] if t in weekly_ret.columns]
    risk_off = [t for t in ["SHY","TLT","GLD","JAAA","TBF","AGG"] if t in weekly_ret.columns]
    
    p = pd.Series(0.0, index=weekly_ret.index)
    
    for i in range(min_weeks, len(weekly_ret)):
        dt = weekly_ret.index[i]
        # Get VIX as of this date
        vix_val = vix.loc[:dt].dropna()
        if len(vix_val) < 252: continue
        
        current_vix = vix_val.iloc[-1]
        vix_pctl = vix_val.iloc[-252:].rank(pct=True).iloc[-1]
        
        if vix_pctl < 0.3:  # Low VIX → risk on
            basket = risk_on
        elif vix_pctl < 0.7:  # Mid → balanced
            basket = balanced
        else:  # High VIX → risk off
            basket = risk_off
        
        if len(basket) > 0:
            p.iloc[i] = weekly_ret.iloc[i][basket].mean()
    
    return p.iloc[min_weeks:]


# ================================================================
# STRATEGY 6: DUAL MOMENTUM (absolute + relative)
# ================================================================
def strat_dual_momentum(lookback=26, risk_free="SHY", top_n=3, name=""):
    """
    Antonacci's dual momentum:
    1. Relative momentum: rank all ETFs by trailing return
    2. Absolute momentum: only buy if return > risk-free rate
    If nothing beats risk-free, go to cash (SHY).
    """
    p = pd.Series(0.0, index=weekly_ret.index)
    
    for i in range(min_weeks, len(weekly_ret)):
        trailing = weekly_px.iloc[i] / weekly_px.iloc[max(0,i-lookback)] - 1
        trailing = trailing.dropna()
        
        # Risk-free return
        rf_ret = trailing.get(risk_free, 0)
        
        # Only keep those beating risk-free (absolute momentum filter)
        above_rf = trailing[trailing > rf_ret]
        if risk_free in above_rf.index:
            above_rf = above_rf.drop(risk_free)
        
        if len(above_rf) >= top_n:
            top = above_rf.nlargest(top_n).index
            p.iloc[i] = weekly_ret.iloc[i][top].mean()
        elif len(above_rf) > 0:
            p.iloc[i] = weekly_ret.iloc[i][above_rf.index].mean()
        else:
            # Go to cash
            if risk_free in weekly_ret.columns:
                p.iloc[i] = weekly_ret.iloc[i].get(risk_free, 0)
    
    return p.iloc[min_weeks:]


# ================================================================
# STRATEGY 7: YIELD CURVE + CREDIT SPREAD SIGNAL
# ================================================================
def strat_macro_signal(name=""):
    """
    Combine yield curve slope + credit spread for macro signal.
    Steep curve + tight spreads = risk-on
    Flat/inverted curve + wide spreads = risk-off
    """
    slope = fred.get("T10Y2Y")
    hy_oas = fred.get("BAMLH0A0HYM2")
    if slope is None or hy_oas is None: return pd.Series(dtype=float)
    
    risk_on = [t for t in ["QQQ","SPY","IWM","HYG","EMB","VNQ","SMH","EEM"] if t in weekly_ret.columns]
    risk_off = [t for t in ["TLT","IEF","GLD","SHY","JAAA","AGG","TBF"] if t in weekly_ret.columns]
    
    p = pd.Series(0.0, index=weekly_ret.index)
    
    for i in range(min_weeks, len(weekly_ret)):
        dt = weekly_ret.index[i]
        
        sl = slope.loc[:dt].dropna()
        hy = hy_oas.loc[:dt].dropna()
        if len(sl) < 252 or len(hy) < 252: continue
        
        # Z-score both
        sl_z = (sl.iloc[-1] - sl.iloc[-252:].mean()) / sl.iloc[-252:].std()
        hy_z = (hy.iloc[-1] - hy.iloc[-252:].mean()) / hy.iloc[-252:].std()
        
        # Combined: positive = risk-on, negative = risk-off
        # Steep curve (positive) + tight spreads (negative hy_z) = risk-on
        signal = sl_z - hy_z
        
        if signal > 0.5:
            basket = risk_on
        elif signal < -0.5:
            basket = risk_off
        else:
            # Blend
            ro_wt = max(0, min(1, (signal + 0.5)))
            basket_ret = 0
            if risk_on: basket_ret += ro_wt * weekly_ret.iloc[i][risk_on].mean()
            if risk_off: basket_ret += (1-ro_wt) * weekly_ret.iloc[i][risk_off].mean()
            p.iloc[i] = basket_ret
            continue
        
        if len(basket) > 0:
            p.iloc[i] = weekly_ret.iloc[i][basket].mean()
    
    return p.iloc[min_weeks:]


# ================================================================
# STRATEGY 8: COMBINED MULTI-FACTOR
# ================================================================
def strat_multifactor(mom_lb=26, rev_lb=1, vol_lb=52, top_n=5,
                       mom_wt=0.4, rev_wt=0.2, vol_wt=0.2, quality_wt=0.2,
                       skip_recent=1, name=""):
    """
    Combine multiple factors:
    1. Momentum (6-month, skip recent week)
    2. Short-term reversal (1-week)
    3. Low volatility
    4. Quality (Sharpe ratio as proxy)
    """
    p = pd.Series(0.0, index=weekly_ret.index)
    
    for i in range(min_weeks, len(weekly_ret)):
        # Available ETFs this week
        avail = weekly_ret.iloc[i].dropna().index
        if len(avail) < top_n: continue
        
        scores = pd.Series(0.0, index=avail)
        
        # 1. Momentum
        end = i - skip_recent
        start = max(0, end - mom_lb)
        if end > start and end < len(weekly_px):
            mom = weekly_px.iloc[end] / weekly_px.iloc[start] - 1
            mom = mom.reindex(avail).dropna()
            if len(mom) > 3:
                mom_z = (mom - mom.mean()) / mom.std()
                scores = scores.add(mom_wt * mom_z, fill_value=0)
        
        # 2. Reversal
        rev = -weekly_ret.iloc[max(0,i-rev_lb):i].mean()
        rev = rev.reindex(avail).dropna()
        if len(rev) > 3:
            rev_z = (rev - rev.mean()) / rev.std()
            scores = scores.add(rev_wt * rev_z, fill_value=0)
        
        # 3. Low vol
        vol_window = weekly_ret.iloc[max(0,i-vol_lb):i]
        vols = vol_window[avail].std()
        vols = vols.dropna()
        if len(vols) > 3:
            vol_z = -(vols - vols.mean()) / vols.std()  # negative: lower vol = higher score
            scores = scores.add(vol_wt * vol_z, fill_value=0)
        
        # 4. Quality (trailing Sharpe)
        quality_window = weekly_ret.iloc[max(0,i-52):i]
        sharpes = quality_window[avail].mean() / quality_window[avail].std()
        sharpes = sharpes.dropna()
        if len(sharpes) > 3:
            q_z = (sharpes - sharpes.mean()) / sharpes.std()
            scores = scores.add(quality_wt * q_z, fill_value=0)
        
        # Pick top N
        scores = scores.dropna()
        if len(scores) >= top_n:
            top = scores.nlargest(top_n).index
            p.iloc[i] = weekly_ret.iloc[i][top].mean()
    
    return p.iloc[min_weeks:]


# ================================================================
# RUN ALL STRATEGIES
# ================================================================
print("="*80)
print("ALTERNATIVE STRATEGIES — WEEKLY REBALANCE, NO DAILY SCALING")
print("="*80)

results = []

# Momentum variations
for lb, skip, tn, name in [
    (13, 1, 5, "Mom_13w_skip1_t5"), (26, 1, 5, "Mom_26w_skip1_t5"),
    (52, 1, 5, "Mom_52w_skip1_t5"), (26, 0, 5, "Mom_26w_skip0_t5"),
    (26, 1, 3, "Mom_26w_skip1_t3"), (26, 1, 10, "Mom_26w_skip1_t10"),
    (13, 1, 3, "Mom_13w_skip1_t3"), (52, 1, 3, "Mom_52w_skip1_t3"),
    (26, 2, 5, "Mom_26w_skip2_t5"),
]:
    p = strat_momentum(lb, 1, tn, 0, skip, name)
    mx = m(p, name)
    if mx: results.append(mx)

# Reversal
for lb, tn, name in [(1,5,"Rev_1w_t5"),(1,10,"Rev_1w_t10"),(2,5,"Rev_2w_t5"),(1,3,"Rev_1w_t3")]:
    p = strat_reversal(lb, tn, name)
    mx = m(p, name)
    if mx: results.append(mx)

# Low vol
for lb, tn, name in [(26,5,"LowVol_26w_t5"),(52,5,"LowVol_52w_t5"),
                       (52,10,"LowVol_52w_t10"),(26,10,"LowVol_26w_t10")]:
    p = strat_low_vol(lb, tn, name)
    mx = m(p, name)
    if mx: results.append(mx)

# Momentum + Reversal blend
for mw, name in [(0.5,"MomRev_50_50"),(0.7,"MomRev_70_30"),(0.3,"MomRev_30_70"),
                   (0.8,"MomRev_80_20"),(0.6,"MomRev_60_40")]:
    p = strat_mom_rev_blend(26, 1, 5, mw, 1, name)
    mx = m(p, name)
    if mx: results.append(mx)

# VIX regime
p = strat_regime_rotation("VIX_Regime")
mx = m(p, "VIX_Regime")
if mx: results.append(mx)

# Dual momentum
for lb, tn, name in [(26,3,"DualMom_26w_t3"),(26,5,"DualMom_26w_t5"),
                       (13,3,"DualMom_13w_t3"),(52,3,"DualMom_52w_t3"),
                       (26,1,"DualMom_26w_t1")]:
    p = strat_dual_momentum(lb, "SHY", tn, name)
    mx = m(p, name)
    if mx: results.append(mx)

# Macro signal
p = strat_macro_signal("Macro_Signal")
mx = m(p, "Macro_Signal")
if mx: results.append(mx)

# Multi-factor
for mom_w, rev_w, vol_w, q_w, tn, name in [
    (0.4, 0.2, 0.2, 0.2, 5, "MF_balanced_t5"),
    (0.6, 0.1, 0.2, 0.1, 5, "MF_mom_heavy_t5"),
    (0.2, 0.3, 0.3, 0.2, 5, "MF_rev_vol_t5"),
    (0.4, 0.2, 0.2, 0.2, 3, "MF_balanced_t3"),
    (0.4, 0.2, 0.2, 0.2, 10, "MF_balanced_t10"),
    (0.5, 0.0, 0.3, 0.2, 5, "MF_no_rev_t5"),
    (0.3, 0.3, 0.2, 0.2, 5, "MF_equal_t5"),
    (0.0, 0.0, 0.5, 0.5, 5, "MF_vol_quality_t5"),
]:
    p = strat_multifactor(26, 1, 52, tn, mom_w, rev_w, vol_w, q_w, 1, name)
    mx = m(p, name)
    if mx: results.append(mx)

# SPY baseline
spy_weekly = weekly_ret.get("SPY")
if spy_weekly is not None:
    spy_m = m(spy_weekly.iloc[min_weeks:], "SPY_BuyHold")
    if spy_m: results.append(spy_m)

# Summary
print(f"\n{'='*80}")
print(f"RANKED BY WALK-FORWARD SHARPE")
print(f"{'='*80}")
print(f"{'Name':22s} {'SR':>7} {'WF':>7} {'Test':>7} {'Ret':>8} {'Vol':>7} {'MDD':>8} {'Sort':>7} {'WR':>6}")
print("-"*82)
for r in sorted(results, key=lambda x:-x['wf_mean']):
    flag = " ★" if r['wf_mean'] > 0.8 else ""
    print(f"  {r['name']:20s} {r['sr']:>6.3f} {r['wf_mean']:>6.3f} {r['test_sr']:>6.3f} "
          f"{r['ret']:>+7.1f}% {r['vol']:>6.1f}% {r['mdd']:>+7.1f}% {r['sortino']:>6.3f} {r['wr']*1:>5.1f}%{flag}")

best = max(results, key=lambda x: x['wf_mean'])
print(f"\nBEST: {best['name']} → SR={best['sr']} WF={best['wf_mean']} Ret={best['ret']}% MDD={best['mdd']}%")

# Compare to carry strategies
print(f"\nFor reference:")
print(f"  Sharpe carry (monthly): SR≈1.54, Ret≈+16.6%")
print(f"  Pure monthly carry:     SR≈1.39, Ret≈+10.9%")

with open(RESULTS_DIR/"alt_experiments.json","w") as f:
    json.dump({"experiments":results,"best":best,"n_configs":len(results)},f,indent=2)
print(f"\nSaved {len(results)} experiments")
