#!/usr/bin/env python3
"""
PROPRIETARY ALPHA V2 — Thinking completely differently.
=========================================================

Everything so far failed because:
1. Price-based signals have look-ahead bias when ranking
2. Macro signals (VIX, credit) are too slow for weekly timing
3. Shuffle tests show most signals ≈ random

What if we stop trying to TIME the market and instead 
exploit STRUCTURAL mispricings between related assets?

NEW IDEAS:

A. CROSS-ASSET DISPERSION HARVESTING
   When the spread between related assets (e.g., HYG-AGG, QQQ-SPY,
   EEM-EFA) deviates from normal, bet on convergence.
   This is pairs trading across asset classes — NOT timing.

B. VOLATILITY TERM STRUCTURE 
   When short-term vol > long-term vol (inverted), markets are
   stressed and about to recover. When normal, risk is fine.
   Use VIX futures term structure proxy: VIXY vs SPY realized vol.

C. PUT-CALL PARITY VIOLATIONS (via ETF premiums)
   Bond ETFs trade at premiums/discounts to NAV. When HYG trades
   at a discount, it's cheap — buy it. When at premium, sell it.
   Proxy: ETF return vs underlying index return divergence.

D. CALENDAR EFFECTS (day-of-week, month-of-year, turn-of-month)
   Well-documented: stocks return more Mon-Tue, month-end/beginning.
   Monthly rebalance at the optimal calendar point.

E. RELATIVE STRENGTH RANKING WITH PROPER LAG
   The key insight: instead of ranking by RETURN (which has leakage),
   rank by RISK-ADJUSTED RETURN computed from data ending 2 WEEKS ago.
   The 2-week gap eliminates any possibility of leakage.

F. MEAN-VARIANCE OPTIMIZATION (weekly)
   Each week, compute the efficient frontier from trailing data.
   Allocate to the tangency portfolio. No ranking needed.

G. INTERMARKET DIVERGENCE
   When bonds rally but stocks don't follow (or vice versa),
   there's a divergence that tends to resolve. Trade the resolution.

H. CARRY ACROSS ASSET CLASSES (weekly version)
   Instead of fixed carry pairs, dynamically pick the highest-carry
   asset class each week. Carry = trailing yield proxy.
"""
import pandas as pd, numpy as np, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")

DATA_DIR = Path("/home/user/bonds/data")
ETF_DIR = DATA_DIR / "etfs"
FRED_PATH = DATA_DIR / "fred" / "_combined_fred.csv"
RESULTS_DIR = Path(__file__).parent / "results"

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

weekly_ret = prices.resample("W-FRI").apply(lambda x: x.pct_change().dropna().add(1).prod()-1)
weekly_px = prices.resample("W-FRI").last()
min_w = 104

def m(r, name=""):
    r=r.dropna()
    if len(r)<52: return None
    ar=r.mean()*52; av=r.std()*np.sqrt(52); sr=ar/av if av>0 else 0
    cum=(1+r).cumprod(); mdd=((cum-cum.cummax())/cum.cummax()).min()
    ds=r[r<0].std()*np.sqrt(52) if (r<0).any() else av
    sortino=ar/ds if ds>0 else 0; wr=(r>0).mean()
    sp=int(len(r)*0.6); tsr=r.iloc[sp:].mean()/r.iloc[sp:].std()*np.sqrt(52) if r.iloc[sp:].std()>0 else 0
    nt=len(r);fs=nt//6;wf=[]
    for fold in range(5):
        s=(fold+1)*fs;e=min(s+fs,nt);fr=r.iloc[s:e]
        if len(fr)>26 and fr.std()>0: wf.append(fr.mean()/fr.std()*np.sqrt(52))
    return {"name":name,"sr":round(sr,3),"ret":round(ar*100,2),"vol":round(av*100,2),
            "mdd":round(mdd*100,2),"sortino":round(sortino,3),"wr":round(wr*100,1),
            "test_sr":round(tsr,3),"wf_mean":round(np.mean(wf),3) if wf else 0}

results = []

# ================================================================
# A. CROSS-ASSET SPREAD MEAN REVERSION
# ================================================================
print("=== A: Cross-Asset Spread Mean Reversion ===")

def strat_spread_mr(long_etf, short_etf, lookback=52, entry_z=1.5, exit_z=0.3, name=""):
    """
    Trade the spread between two related assets.
    When spread is unusually wide → buy the cheap one, sell the rich one.
    Signal uses data through week i-1 (shift built in).
    """
    if long_etf not in weekly_ret.columns or short_etf not in weekly_ret.columns:
        return pd.Series(dtype=float)
    
    # Log price ratio
    ratio = np.log(weekly_px[long_etf] / weekly_px[short_etf])
    ratio_ma = ratio.rolling(lookback, min_periods=26).mean()
    ratio_std = ratio.rolling(lookback, min_periods=26).std()
    z = (ratio - ratio_ma) / ratio_std.clip(lower=0.001)
    z = z.shift(1)  # USE LAST WEEK'S Z-SCORE
    
    p = pd.Series(0.0, index=weekly_ret.index)
    pos = 0  # -1, 0, or 1
    
    for i in range(min_w, len(weekly_ret)):
        zval = z.iloc[i] if i < len(z) and not np.isnan(z.iloc[i]) else 0
        
        if pos == 0:
            if zval > entry_z:
                pos = -1  # Spread too wide → short long, buy short
            elif zval < -entry_z:
                pos = 1   # Spread too narrow → buy long, short short
        else:
            if abs(zval) < exit_z:
                pos = 0
        
        if pos == 1:
            p.iloc[i] = weekly_ret.iloc[i].get(long_etf, 0) - weekly_ret.iloc[i].get(short_etf, 0)
        elif pos == -1:
            p.iloc[i] = -(weekly_ret.iloc[i].get(long_etf, 0) - weekly_ret.iloc[i].get(short_etf, 0))
    
    return p.iloc[min_w:]

# For LONG-ONLY version: when spread is wide, buy the cheap asset
def strat_spread_longonly(asset_a, asset_b, lookback=52, name=""):
    """
    Each week: compute z-score of log ratio A/B.
    If A is cheap (z < -1): overweight A
    If B is cheap (z > 1): overweight B
    Otherwise: equal weight both
    All using LAST WEEK's data (shift(1)).
    """
    if asset_a not in weekly_ret.columns or asset_b not in weekly_ret.columns:
        return pd.Series(dtype=float)
    
    ratio = np.log(weekly_px[asset_a] / weekly_px[asset_b])
    ratio_ma = ratio.rolling(lookback, min_periods=26).mean()
    ratio_std = ratio.rolling(lookback, min_periods=26).std()
    z = (ratio - ratio_ma) / ratio_std.clip(lower=0.001)
    z = z.shift(1)
    
    p = pd.Series(0.0, index=weekly_ret.index)
    for i in range(min_w, len(weekly_ret)):
        zval = z.iloc[i] if i < len(z) and not np.isnan(z.iloc[i]) else 0
        ra = weekly_ret.iloc[i].get(asset_a, 0)
        rb = weekly_ret.iloc[i].get(asset_b, 0)
        
        if zval < -1:    # A is cheap
            p.iloc[i] = 0.7*ra + 0.3*rb
        elif zval > 1:   # B is cheap
            p.iloc[i] = 0.3*ra + 0.7*rb
        else:
            p.iloc[i] = 0.5*ra + 0.5*rb
    
    return p.iloc[min_w:]

# Test many pairs
pairs = [
    ("HYG","AGG","Credit_vs_Govt"), ("QQQ","SPY","Growth_vs_Value"),
    ("EEM","EFA","EM_vs_DM"), ("IWM","SPY","Small_vs_Large"),
    ("TLT","SHY","Long_vs_Short_Treas"), ("GLD","TLT","Gold_vs_Bonds"),
    ("VNQ","SPY","REIT_vs_Equity"), ("XLE","XLK","Energy_vs_Tech"),
    ("HYG","LQD","HY_vs_IG"), ("EMB","AGG","EM_vs_US_Bond"),
    ("SCHD","QQQ","Div_vs_Growth"), ("XLU","XLY","Defensive_vs_Cyclical"),
    ("GLD","SPY","Gold_vs_Equity"), ("EEM","SPY","EM_vs_US"),
    ("BTC_USD","GLD","BTC_vs_Gold"), ("SMH","SPY","Semi_vs_Broad"),
]

for a, b, name in pairs:
    p = strat_spread_longonly(a, b, 52, name)
    mx = m(p, f"SpreadLO_{name}")
    if mx: results.append(mx)

# ================================================================
# B. MULTI-PAIR SPREAD PORTFOLIO
# ================================================================
print("=== B: Multi-Pair Spread Portfolio ===")

def strat_multi_spread(pairs_list, lookback=52, name=""):
    """Combine multiple spread trades into one portfolio."""
    all_rets = []
    for a, b, _ in pairs_list:
        p = strat_spread_longonly(a, b, lookback)
        if len(p.dropna()) > 52:
            all_rets.append(p)
    
    if not all_rets:
        return pd.Series(dtype=float)
    
    combined = pd.concat(all_rets, axis=1).mean(axis=1)
    return combined

# All pairs
p = strat_multi_spread(pairs, 52, "AllPairs")
mx = m(p, "MultiSpread_All")
if mx: results.append(mx)

# Best pairs (credit-related)
credit_pairs = [p for p in pairs if any(x in p[2] for x in ["Credit","HY","IG","EM_vs_US_Bond"])]
p = strat_multi_spread(credit_pairs, 52, "CreditPairs")
mx = m(p, "MultiSpread_Credit")
if mx: results.append(mx)

# Equity style pairs
equity_pairs = [p for p in pairs if any(x in p[2] for x in ["Growth","Small","Div","Defensive","Semi"])]
p = strat_multi_spread(equity_pairs, 52, "EquityPairs")
mx = m(p, "MultiSpread_Equity")
if mx: results.append(mx)

# ================================================================
# C. RELATIVE STRENGTH WITH 2-WEEK GAP (zero leakage guaranteed)
# ================================================================
print("=== C: Relative Strength (2-week gap) ===")

def strat_rel_strength_gapped(assets, lookback=26, gap=2, top_n=5, name=""):
    """
    Rank by trailing return ending GAP weeks ago. 
    2-week gap = impossible to have any leakage.
    """
    p = pd.Series(0.0, index=weekly_ret.index)
    for i in range(min_w, len(weekly_ret)):
        end = i - gap  # 2 weeks ago
        start = max(0, end - lookback)
        if end <= start or end < 0: continue
        
        trailing = weekly_px.iloc[end] / weekly_px.iloc[start] - 1
        trailing = trailing[assets].dropna()
        if len(trailing) < top_n: continue
        
        top = trailing.nlargest(top_n).index
        avail_ret = weekly_ret.iloc[i][top].dropna()
        if len(avail_ret) > 0:
            p.iloc[i] = avail_ret.mean()
    
    return p.iloc[min_w:]

# Different asset pools
broad = [t for t in ["SPY","QQQ","IWM","EFA","EEM","TLT","GLD","VNQ","HYG","AGG",
                      "DBC","SLV","EMB","SCHD","SMH","XLE","XLK","XLU","XLP","AMLP",
                      "BTC_USD","ETH_USD","GBTC","JAAA","BKLN","SHY"] if t in weekly_px.columns]

equity_only = [t for t in ["SPY","QQQ","IWM","EFA","EEM","VNQ","SCHD","HDV","DVY",
                            "SMH","XLK","XLE","XLU","XLP","XLY","XLI","XLF","XLB",
                            "EWJ","EWZ","FXI","KWEB","ARKK","INDA"] if t in weekly_px.columns]

with_lev = [t for t in ["TQQQ","UPRO","SOXL","TECL","SSO","QLD","SPY","QQQ","IWM",
                          "EEM","GLD","TLT","BTC_USD","ETH_USD","GBTC","JAAA","SHY"] if t in weekly_px.columns]

for assets, pool_name in [(broad,"Broad"),(equity_only,"Equity"),(with_lev,"WithLev")]:
    for lb, gap, tn in [(13,2,5),(26,2,5),(52,2,5),(26,2,3),(26,2,10),(13,2,3),(26,3,5),(13,1,5)]:
        name = f"RS_{pool_name}_{lb}w_gap{gap}_t{tn}"
        p = strat_rel_strength_gapped(assets, lb, gap, tn, name)
        mx = m(p, name)
        if mx: results.append(mx)

# ================================================================
# D. DYNAMIC CARRY ROTATION (weekly)
# ================================================================
print("=== D: Dynamic Carry Rotation ===")

def strat_carry_rotation(name=""):
    """
    Each week, estimate "carry" for each asset class using
    trailing 52-week return as proxy. Allocate to top 5 by carry.
    Use 2-week-old data to avoid leakage.
    """
    carry_assets = [t for t in ["HYG","JNK","EMB","EMLC","BKLN","PFF","AMLP","SCHD","HDV",
                                 "DVY","VIG","VNQ","IYR","GLD","TLT","AGG","IEF","SHY",
                                 "JAAA","LQD","MUB","TIP","BNDX","XLU","XLP",
                                 "BTC_USD","GBTC","EEM","EFA","SPY"] if t in weekly_px.columns]
    
    p = pd.Series(0.0, index=weekly_ret.index)
    for i in range(min_w, len(weekly_ret)):
        end = i - 2  # 2-week gap
        start = max(0, end - 52)
        if end <= start: continue
        
        carry = weekly_px.iloc[end] / weekly_px.iloc[start] - 1
        carry = carry[carry_assets].dropna()
        if len(carry) < 5: continue
        
        top5 = carry.nlargest(5).index
        avail = weekly_ret.iloc[i][top5].dropna()
        if len(avail) > 0:
            p.iloc[i] = avail.mean()
    
    return p.iloc[min_w:]

p = strat_carry_rotation("CarryRotation")
mx = m(p, "CarryRotation_52w")
if mx: results.append(mx)

# ================================================================
# E. INTERMARKET DIVERGENCE
# ================================================================
print("=== E: Intermarket Divergence ===")

def strat_divergence(name=""):
    """
    When stocks and bonds diverge (one up, other down for 4+ weeks),
    bet on convergence. Use 2-week-old data.
    """
    if "SPY" not in weekly_ret.columns or "TLT" not in weekly_ret.columns:
        return pd.Series(dtype=float)
    
    spy_4w = weekly_px["SPY"].pct_change(4).shift(2)
    tlt_4w = weekly_px["TLT"].pct_change(4).shift(2)
    
    p = pd.Series(0.0, index=weekly_ret.index)
    for i in range(min_w, len(weekly_ret)):
        if i >= len(spy_4w) or i >= len(tlt_4w): continue
        spy_r = spy_4w.iloc[i]; tlt_r = tlt_4w.iloc[i]
        if np.isnan(spy_r) or np.isnan(tlt_r): continue
        
        # Divergence: one up, other down
        if spy_r > 0.02 and tlt_r < -0.02:
            # Stocks up, bonds down → bonds should catch up
            p.iloc[i] = 0.6*weekly_ret.iloc[i].get("TLT",0) + 0.4*weekly_ret.iloc[i].get("SPY",0)
        elif spy_r < -0.02 and tlt_r > 0.02:
            # Stocks down, bonds up → stocks should catch up
            p.iloc[i] = 0.6*weekly_ret.iloc[i].get("SPY",0) + 0.4*weekly_ret.iloc[i].get("TLT",0)
        else:
            # No divergence → balanced
            p.iloc[i] = 0.5*weekly_ret.iloc[i].get("SPY",0) + 0.5*weekly_ret.iloc[i].get("TLT",0)
    
    return p.iloc[min_w:]

p = strat_divergence("StockBondDiv")
mx = m(p, "StockBond_Divergence")
if mx: results.append(mx)

# ================================================================
# F. COMBINED: Best of everything
# ================================================================
print("=== F: Combined Strategies ===")

# Combine spread + gapped momentum + carry rotation
all_strats = {}

# Best spread pairs
for a, b, name in [("HYG","AGG","cr"),("QQQ","SPY","gr"),("EEM","EFA","em"),("GLD","TLT","gt")]:
    p = strat_spread_longonly(a, b, 52)
    if len(p.dropna()) > 52: all_strats[f"spread_{name}"] = p

# Best gapped momentum
for assets, lb, gap, tn, name in [
    (broad, 26, 2, 5, "broad_26"), (equity_only, 13, 2, 5, "eq_13"),
    (with_lev, 26, 2, 3, "lev_26")]:
    p = strat_rel_strength_gapped(assets, lb, gap, tn)
    if len(p.dropna()) > 52: all_strats[f"gapRS_{name}"] = p

# Carry rotation
p = strat_carry_rotation()
if len(p.dropna()) > 52: all_strats["carry_rot"] = p

# Divergence
p = strat_divergence()
if len(p.dropna()) > 52: all_strats["divergence"] = p

# Equal weight combine
if all_strats:
    combined = pd.DataFrame(all_strats).mean(axis=1).dropna()
    mx = m(combined, "Combined_All")
    if mx: results.append(mx)

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*70}")
print(f"RANKED BY WALK-FORWARD SHARPE — ALL ZERO-LEAKAGE")
print(f"{'='*70}")
print(f"{'Name':35s} {'SR':>7} {'WF':>7} {'Test':>7} {'Ret':>8} {'Vol':>7} {'MDD':>8} {'Sort':>7}")
print("-"*85)
for r in sorted(results, key=lambda x:-x['wf_mean'])[:30]:
    flag = " ★" if r['wf_mean'] > 0.9 else ""
    print(f"  {r['name']:33s} {r['sr']:>6.3f} {r['wf_mean']:>6.3f} {r['test_sr']:>6.3f} "
          f"{r['ret']:>+7.1f}% {r['vol']:>6.1f}% {r['mdd']:>+7.1f}% {r['sortino']:>6.3f}{flag}")

best = max(results, key=lambda x: x['wf_mean'])
print(f"\nBEST: {best['name']} → SR={best['sr']} WF={best['wf_mean']} Ret={best['ret']}% MDD={best['mdd']}%")
print(f"Monthly carry (fixed): SR≈1.54 WF≈1.62")

with open(RESULTS_DIR/"proprietary_v2.json","w") as f:
    json.dump({"experiments":results,"best":best,"n_total":len(results)},f,indent=2)
print(f"\nSaved {len(results)} experiments")
