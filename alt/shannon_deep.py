#!/usr/bin/env python3
"""
SHANNON'S DEMON — DEEP EXPLORATION
====================================

Questions to answer:
1. Leveraged ETFs? (more vol = more rebalancing premium?)
2. Crypto? (BTC/ETH have low correlation to everything)
3. Inverse ETFs? (negative correlation by design)
4. Is the non-correlation structural or does it drift?
5. Should we adapt correlations weekly?
6. What's the return with optimal asset selection?
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
weekly_ret = prices.resample("W-FRI").apply(lambda x: x.pct_change().dropna().add(1).prod()-1)
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
            "test_sr":round(tsr,3),"wf_mean":round(np.mean(wf),3) if wf else 0,
            "nav":round(float(cum.iloc[-1]),2)}

results = []

print("="*70)
print("SHANNON'S DEMON — DEEP EXPLORATION")
print("="*70)

# ================================================================
# FIRST: Understand the math
# ================================================================
print("""
SHANNON'S DEMON MATH:
  Two assets with returns r1, r2, volatilities σ1, σ2, correlation ρ.
  Rebalanced to 50/50 weekly.
  
  Expected rebalancing bonus ≈ 0.5 × w1 × w2 × (σ1² + σ2² - 2ρσ1σ2) × rebal_freq
  
  Maximized when:
  - σ1 and σ2 are HIGH (more vol = more bonus) → LEVERAGED ETFs
  - ρ is LOW or NEGATIVE (less correlation = more bonus) → UNCORRELATED/INVERSE
  - Rebalancing is FREQUENT → DAILY better than weekly
  
  KEY INSIGHT: Shannon's demon LOVES high-vol, low-correlation pairs.
  Leveraged ETFs + crypto + inverse ETFs = MAXIMUM rebalancing premium.
""")

# ================================================================
# 1. STATIC BASKETS — every combination
# ================================================================
print("\n=== STATIC BASKETS ===")

baskets = {
    # Original (baseline)
    "Original9": ["JAAA","SCHD","GLD","TLT","HYG","EMB","BKLN","MUB","TIP"],
    
    # With crypto
    "Crypto3": ["BTC_USD","GLD","TLT"],
    "Crypto5": ["BTC_USD","ETH_USD","GLD","TLT","JAAA"],
    "Crypto7": ["BTC_USD","ETH_USD","GLD","TLT","JAAA","SCHD","SHY"],
    "CryptoWide": ["BTC_USD","ETH_USD","SOL_USD","GLD","TLT","JAAA","SCHD","AGG","SHY"],
    
    # With leveraged
    "Lev3": ["TQQQ","TMF","GLD"],
    "Lev5": ["TQQQ","TMF","GLD","SOXL","UGL"],
    "Lev_Mixed": ["TQQQ","TMF","GLD","JAAA","SCHD"],
    "Lev_Crypto": ["TQQQ","BTC_USD","GLD","TMF","JAAA"],
    
    # With inverse (negative correlation by design)
    "Inv3": ["SPY","SH","GLD"],
    "Inv5": ["QQQ","PSQ","TLT","TBF","GLD"],
    "Inv_Carry": ["HYG","TBF","GLD","JAAA","SCHD"],
    "Inv_Wide": ["SPY","SH","TLT","TBF","GLD","GLL","QQQ","PSQ"],
    
    # Max diversification (one from each asset class)
    "MaxDiv": ["SPY","TLT","GLD","HYG","JAAA","SCHD","EMB","VNQ","DBC","EEM","BKLN","MUB"],
    "MaxDiv_Crypto": ["SPY","TLT","GLD","HYG","JAAA","SCHD","BTC_USD","EMB","VNQ","DBC"],
    "MaxDiv_Lev": ["TQQQ","TMF","GLD","HYG","JAAA","SCHD","BTC_USD","SOXL"],
    
    # Leveraged + Crypto + Inverse (maximum vol/decorrelation)
    "UltraShannon": ["TQQQ","TMF","BTC_USD","GLD","SOXL","SH","TBF","JAAA"],
    "UltraShannon2": ["TQQQ","TMF","BTC_USD","ETH_USD","GLD","SH","TBF","JAAA","SCHD"],
    "MaxVol": ["TQQQ","SOXL","BTC_USD","ETH_USD","SOL_USD","TMF","GLD","ERX"],
    
    # Carry-focused Shannon
    "CarryShannon": ["JAAA","BKLN","HYG","EMB","MUB","SCHD","HDV","PFF","MBB","TIP"],
    "CarryShannon_Gold": ["JAAA","BKLN","HYG","SCHD","GLD","TLT","MUB","TIP"],
    
    # Minimum vol Shannon
    "MinVol": ["SHY","JAAA","BKLN","AGG","MUB","TIP","BNDX","GLD"],
    
    # Pure non-correlation
    "PureAntiCorr": ["SPY","TLT","GLD","UUP","VNQ"],
    "PureAntiCorr_Lev": ["TQQQ","TMF","GLD","UGL","DRN"],
}

for name, assets in baskets.items():
    avail = [a for a in assets if a in weekly_ret.columns]
    if len(avail) < 2: continue
    p = weekly_ret[avail].mean(axis=1)
    mx = m(p.iloc[min_w:], f"Static_{name}")
    if mx: results.append(mx)

# ================================================================
# 2. ADAPTIVE SHANNON — update weights by inverse correlation
# ================================================================
print("\n=== ADAPTIVE SHANNON (weekly correlation update) ===")

def adaptive_shannon(universe, lookback=52, gap=2, method="equal", min_assets=3, name=""):
    """
    Each week:
    - Compute trailing correlation matrix (from 2 weeks ago)
    - Weight assets to MINIMIZE portfolio correlation
    - Options: equal weight, inverse-vol, min-correlation, max-diversification
    """
    avail = [a for a in universe if a in weekly_ret.columns]
    if len(avail) < min_assets: return pd.Series(dtype=float)
    
    p = pd.Series(0.0, index=weekly_ret.index)
    
    for i in range(min_w, len(weekly_ret)):
        end = i - gap
        if end < lookback: continue
        
        window = weekly_ret[avail].iloc[end-lookback:end].dropna(axis=1, how="all")
        cols = [c for c in window.columns if window[c].std() > 0]
        if len(cols) < min_assets: continue
        window = window[cols]
        
        if method == "equal":
            w = pd.Series(1.0/len(cols), index=cols)
        
        elif method == "inverse_vol":
            vols = window.std() * np.sqrt(52)
            inv = 1.0/vols.clip(lower=0.001)
            w = inv/inv.sum()
        
        elif method == "min_corr":
            # Select the N least-correlated assets
            corr = window.corr()
            avg_corr = corr.mean()
            # Pick assets with lowest average correlation
            n_pick = min(8, len(cols))
            lowest_corr = avg_corr.nsmallest(n_pick).index
            w = pd.Series(1.0/n_pick, index=lowest_corr)
        
        elif method == "max_div":
            # Maximum diversification: weight by vol / (vol contribution)
            vols = window.std() * np.sqrt(52)
            corr = window.corr()
            # Simple approximation: weight by vol * (1 - avg_corr)
            avg_corr = corr.mean()
            div_score = vols * (1 - avg_corr.clip(lower=-0.5))
            w = div_score / div_score.sum()
        
        elif method == "risk_parity":
            vols = window.std() * np.sqrt(52)
            inv_vol = 1.0/vols.clip(lower=0.001)
            w = inv_vol/inv_vol.sum()
        
        else:
            w = pd.Series(1.0/len(cols), index=cols)
        
        ret_week = weekly_ret.iloc[i][w.index].dropna()
        if len(ret_week) > 0:
            p.iloc[i] = sum(w.get(k,0)*ret_week.get(k,0) for k in w.index if k in ret_week.index)
    
    return p.iloc[min_w:]

# Test adaptive methods on different universes
universes = {
    "Broad": ["SPY","TLT","GLD","HYG","JAAA","SCHD","EMB","VNQ","DBC","EEM","BKLN","MUB","BTC_USD","ETH_USD"],
    "LevCrypto": ["TQQQ","TMF","BTC_USD","ETH_USD","GLD","SOXL","JAAA","SCHD","SH","TBF"],
    "CarryPlus": ["JAAA","BKLN","HYG","EMB","SCHD","HDV","MUB","GLD","TLT","BTC_USD"],
    "UltraBroad": ["SPY","QQQ","TLT","IEF","GLD","SLV","HYG","LQD","JAAA","SCHD","HDV","EMB","VNQ","DBC","EEM","EFA","BTC_USD","ETH_USD","BKLN","MUB","AMLP","TIP"],
}

for uni_name, uni_assets in universes.items():
    for method in ["equal","inverse_vol","min_corr","max_div","risk_parity"]:
        p = adaptive_shannon(uni_assets, 52, 2, method, 3, f"{uni_name}_{method}")
        mx = m(p, f"Adapt_{uni_name}_{method}")
        if mx: results.append(mx)

# ================================================================
# 3. CORRELATION STABILITY ANALYSIS
# ================================================================
print("\n=== CORRELATION STABILITY ===")
# Check if key correlations are stable or drifting
key_pairs = [("SPY","TLT"),("GLD","SPY"),("GLD","TLT"),("BTC_USD","SPY"),
             ("BTC_USD","GLD"),("HYG","TLT"),("JAAA","SPY"),("SCHD","TLT")]

print(f"{'Pair':15s} {'Full':>7} {'2005-10':>7} {'2011-15':>7} {'2016-20':>7} {'2021-26':>7} {'Stable?':>8}")
for a, b in key_pairs:
    if a not in ret.columns or b not in ret.columns: continue
    r_ab = ret[[a,b]].dropna()
    full_corr = r_ab[a].corr(r_ab[b])
    periods = [("2005","2010"),("2011","2015"),("2016","2020"),("2021","2026")]
    period_corrs = []
    for start, end in periods:
        sub = r_ab[(r_ab.index>=start)&(r_ab.index<=end)]
        if len(sub) > 100:
            period_corrs.append(round(sub[a].corr(sub[b]),2))
        else:
            period_corrs.append(None)
    
    corr_vals = [c for c in period_corrs if c is not None]
    stable = "YES" if corr_vals and (max(corr_vals)-min(corr_vals)) < 0.4 else "DRIFTS"
    print(f"  {a}/{b:8s} {full_corr:>6.2f} {'':>3}{'  '.join(str(c) if c else '  -  ' for c in period_corrs)}  {stable}")

# ================================================================
# 4. DAILY REBALANCED SHANNON (more frequent = more premium)
# ================================================================
print("\n=== DAILY vs WEEKLY REBALANCED SHANNON ===")
# The rebalancing premium is proportional to rebalance frequency
# Daily should beat weekly which beats monthly

for name, assets in [
    ("UltraShannon", ["TQQQ","TMF","BTC_USD","GLD","SOXL","SH","TBF","JAAA"]),
    ("CryptoWide", ["BTC_USD","ETH_USD","SOL_USD","GLD","TLT","JAAA","SCHD","AGG","SHY"]),
    ("MaxDiv_Crypto", ["SPY","TLT","GLD","HYG","JAAA","SCHD","BTC_USD","EMB","VNQ","DBC"]),
]:
    avail = [a for a in assets if a in ret.columns]
    if len(avail) < 3: continue
    
    # Daily rebalanced
    daily_p = ret[avail].mean(axis=1)
    # Weekly rebalanced (already have)
    weekly_p = weekly_ret[avail].mean(axis=1)
    
    mx_d = m(daily_p.resample("W-FRI").apply(lambda x:(1+x).prod()-1).iloc[min_w:], f"DailyRebal_{name}")
    mx_w = m(weekly_p.iloc[min_w:], f"WeeklyRebal_{name}")
    
    if mx_d and mx_w:
        results.append(mx_d)
        results.append(mx_w)
        print(f"  {name}:")
        print(f"    Daily:  SR={mx_d['sr']} Ret={mx_d['ret']}% MDD={mx_d['mdd']}%")
        print(f"    Weekly: SR={mx_w['sr']} Ret={mx_w['ret']}% MDD={mx_w['mdd']}%")

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*70}")
print(f"ALL SHANNON EXPERIMENTS — RANKED BY WF SHARPE")
print(f"{'='*70}")
print(f"{'Name':40s} {'SR':>7} {'WF':>7} {'Test':>7} {'Ret':>8} {'Vol':>7} {'MDD':>8}")
print("-"*80)
for r in sorted(results, key=lambda x:-x['wf_mean'])[:30]:
    flag = " ★" if r['wf_mean'] > 1.0 else ""
    print(f"  {r['name']:38s} {r['sr']:>6.3f} {r['wf_mean']:>6.3f} {r['test_sr']:>6.3f} "
          f"{r['ret']:>+7.1f}% {r['vol']:>6.1f}% {r['mdd']:>+7.1f}%{flag}")

best = max(results, key=lambda x: x['wf_mean'])
best_sr = max(results, key=lambda x: x['sr'])
best_ret = max(results, key=lambda x: x['ret'])
print(f"\nBest WF:     {best['name']} → SR={best['sr']} WF={best['wf_mean']} Ret={best['ret']}% MDD={best['mdd']}%")
print(f"Best SR:     {best_sr['name']} → SR={best_sr['sr']} Ret={best_sr['ret']}% MDD={best_sr['mdd']}%")
print(f"Best Return: {best_ret['name']} → SR={best_ret['sr']} Ret={best_ret['ret']}% MDD={best_ret['mdd']}%")

with open(RESULTS_DIR/"shannon_deep.json","w") as f:
    json.dump({"experiments":results,"best_wf":best,"best_sr":best_sr,"best_ret":best_ret,"n":len(results)},f,indent=2)
print(f"\nSaved {len(results)} experiments")
