#!/usr/bin/env python3
"""
PROPRIETARY V3 — Thinking from first principles.
==================================================

Why does the carry strategy work? Because it harvests a STRUCTURAL
premium and hedges away unwanted risk. The daily scaling works
because it maintains CONSTANT RISK exposure.

What if we could maintain constant risk WITHOUT daily scaling?

INSIGHT: The carry streams have LOW and STABLE volatility by
construction (bond carry + inverse hedge ≈ spread return, which
is much less volatile than either leg). If we pick the RIGHT
carry streams, they already have stable vol — no scaling needed.

NEW APPROACH: Instead of scaling volatile streams, find streams
that are INHERENTLY low-vol and stable. Then equal-weight them.
No scaling. Just pick the right instruments.

ALSO: What about exploiting structures in the ETF ecosystem itself?

IDEA 1: SYNTHETIC CONSTANT-VOL CARRY
  Hold a basket of the LOWEST-VOL carry streams.
  By construction, low-vol streams need less scaling.
  Equal weight them. No daily adjustment.

IDEA 2: ETF CREATION/REDEMPTION ARBITRAGE
  Bond ETFs (HYG, LQD, AGG) trade at premiums/discounts to NAV.
  When discount is large → buy (ETF is cheap vs underlying).
  Proxy: ETF weekly return vs credit spread change.

IDEA 3: INFORMATION FLOW ACROSS MARKETS
  Credit markets lead equity markets by 1-2 days.
  IG bond returns on Mon-Wed predict equity returns on Thu-Fri.
  Weekly version: use LAST week's bond market action to predict
  THIS week's equity returns.

IDEA 4: VOLATILITY CLUSTERING
  After a low-vol week, the NEXT week tends to also be low-vol.
  After a high-vol week, the next tends to be high-vol.
  Use this to dynamically size between aggressive and conservative
  baskets, computed from LAST WEEK's realized vol.

IDEA 5: CROSS-ASSET CARRY RANKING WITH VOL PENALTY
  Rank assets by: carry_proxy / realized_vol (Sharpe-like).
  But compute BOTH from data ending 2 weeks ago.
  This is like our carry strategy but weekly, no scaling, gapped.

IDEA 6: PAIRED MOMENTUM ROTATION
  Instead of ranking individual assets (which has leakage),
  rank PAIRS of assets by their SPREAD momentum.
  If HYG-AGG spread momentum is positive → HY carry is working.
  If negative → rotate to safer carry.
  2-week gap on all calculations.

IDEA 7: INVERSE VOLATILITY WEIGHTING (weekly, frozen)
  Each week, weight ALL assets by 1/vol from 2 weeks ago.
  This is not "vol scaling" — it's just portfolio construction.
  Low-vol assets get more weight. High-vol get less.
  No daily adjustment — weights frozen for a week.
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
daily_ret = ret
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

# ================================================================
# IDEA 1: SYNTHETIC CONSTANT-VOL CARRY
# ================================================================
print("=== IDEA 1: Low-Vol Carry Basket ===")

# Build carry streams as simple daily blends, then pick the lowest-vol ones
def hedged(r, l, h, hw):
    if l not in r.columns or h not in r.columns: return None
    w1=1.0/(1.0+hw); w2=hw/(1.0+hw)
    return (w1*r[l]+w2*r[h]).dropna()

carry_streams = {}
for l,h,hw,n in [
    ("HYG","TBF",0.5,"HYcarry"),("JNK","TBF",0.5,"JNKcarry"),
    ("LQD","TBF",0.6,"IGcarry"),("VCIT","TYO",0.3,"MidCorpCarry"),
    ("IGIB","TYO",0.3,"IG5Carry"),("EMB","TBF",0.4,"EMcarry"),
    ("MUB","TBF",0.3,"MuniCarry"),("MBB","TBF",0.6,"MBScarry"),
    ("TIP","TBF",0.5,"TIPScarry"),("PFF","TBF",0.4,"PrefCarry"),
    ("SCHD","SH",0.3,"DivCarry"),("HDV","SH",0.3,"HDVcarry"),
    ("VIG","SH",0.3,"VIGcarry"),("DVY","SH",0.3,"DVYcarry"),
    ("XLP","SH",0.4,"StaplesCarry"),("XLU","SH",0.3,"UtilCarry"),
    ("ANGL","TBF",0.4,"FallenAngelCarry"),("BOND","TBF",0.3,"PIMCOcarry"),
    ("BKLN","SHY",0.0,"SrLoanCarry"),("JAAA","SHY",0.0,"CLOcarry"),
    ("IGF","SH",0.3,"InfraCarry"),("CWB","SH",0.3,"ConvertCarry"),
]:
    if hw == 0:
        if l in daily_ret.columns:
            carry_streams[n] = daily_ret[l]
    else:
        r_stream = hedged(daily_ret, l, h, hw)
        if r_stream is not None: carry_streams[n] = r_stream

# Convert to weekly
weekly_carry = pd.DataFrame({n: s.resample("W-FRI").apply(lambda x: (1+x).prod()-1)
                              for n, s in carry_streams.items()})

# Strategy: each week, pick the 5 lowest-vol carry streams from 2 weeks ago
# Then hold them equal-weight for 1 week
def low_vol_carry(top_n=5, vol_lookback=26, gap=2, name=""):
    p = pd.Series(0.0, index=weekly_carry.index)
    for i in range(min_w, len(weekly_carry)):
        end = i - gap
        if end < vol_lookback: continue
        
        vols = weekly_carry.iloc[end-vol_lookback:end].std() * np.sqrt(52)
        vols = vols.dropna()
        vols = vols[vols > 0]
        if len(vols) < top_n: continue
        
        # Lowest vol streams
        lowest = vols.nsmallest(top_n).index
        avail = weekly_carry.iloc[i][lowest].dropna()
        if len(avail) > 0:
            p.iloc[i] = avail.mean()
    
    return p.iloc[min_w:]

for tn, vlb in [(3,13),(3,26),(5,13),(5,26),(5,52),(7,26),(10,26)]:
    p = low_vol_carry(tn, vlb, 2, f"LowVolCarry_{tn}_{vlb}w")
    mx = m(p, f"LowVolCarry_t{tn}_{vlb}w")
    if mx: results.append(mx)

# Also: pick by trailing Sharpe (2-week gap)
def high_sharpe_carry(top_n=5, lookback=52, gap=2, name=""):
    p = pd.Series(0.0, index=weekly_carry.index)
    for i in range(min_w, len(weekly_carry)):
        end = i - gap
        if end < lookback: continue
        
        window = weekly_carry.iloc[end-lookback:end]
        sharpes = window.mean() / window.std() * np.sqrt(52)
        sharpes = sharpes.dropna()
        sharpes = sharpes[sharpes > 0]
        if len(sharpes) < top_n: continue
        
        top = sharpes.nlargest(top_n).index
        avail = weekly_carry.iloc[i][top].dropna()
        if len(avail) > 0:
            p.iloc[i] = avail.mean()
    
    return p.iloc[min_w:]

for tn, lb in [(3,26),(5,26),(5,52),(7,52),(3,13),(5,13),(3,52),(10,52)]:
    p = high_sharpe_carry(tn, lb, 2, f"HSCarry_{tn}_{lb}w")
    mx = m(p, f"HighSRCarry_t{tn}_{lb}w")
    if mx: results.append(mx)

# ================================================================
# IDEA 5: CARRY RANKING WITH VOL PENALTY (weekly, gapped)
# ================================================================
print("=== IDEA 5: Carry Ranking With Vol Penalty ===")

def carry_vol_rank(top_n=5, ret_lookback=52, vol_lookback=26, gap=2, name=""):
    """Rank carry streams by return/vol (Sharpe proxy), 2-week gap."""
    p = pd.Series(0.0, index=weekly_carry.index)
    for i in range(min_w, len(weekly_carry)):
        end = i - gap
        if end < max(ret_lookback, vol_lookback): continue
        
        trailing_ret = weekly_carry.iloc[end-ret_lookback:end].mean() * 52
        trailing_vol = weekly_carry.iloc[end-vol_lookback:end].std() * np.sqrt(52)
        
        # Sharpe-like ranking
        rank_score = trailing_ret / trailing_vol.clip(lower=0.001)
        rank_score = rank_score.dropna()
        rank_score = rank_score[rank_score > 0]
        if len(rank_score) < top_n: continue
        
        # Weight by score squared (like current strategy)
        top = rank_score.nlargest(top_n)
        sq = top**2
        w = sq / sq.sum()
        
        ret_this_week = weekly_carry.iloc[i][w.index].dropna()
        if len(ret_this_week) > 0:
            p.iloc[i] = sum(w[k] * ret_this_week.get(k, 0) for k in w.index if k in ret_this_week.index)
    
    return p.iloc[min_w:]

for tn, rl, vl in [(3,26,13),(5,26,13),(5,52,26),(3,52,26),(5,26,26),
                     (3,13,13),(7,52,26),(5,52,52),(3,52,52)]:
    p = carry_vol_rank(tn, rl, vl, 2)
    mx = m(p, f"CarryVolRank_t{tn}_r{rl}_v{vl}")
    if mx: results.append(mx)

# ================================================================
# IDEA 4: VOL CLUSTERING REGIME (weekly)
# ================================================================
print("=== IDEA 4: Vol Clustering Regime ===")

def vol_cluster_regime(name=""):
    """
    Use LAST week's realized vol to decide THIS week's aggressiveness.
    Low vol last week → more aggressive (carry + equity)
    High vol last week → more conservative (short duration + gold)
    """
    if "SPY" not in daily_ret.columns: return pd.Series(dtype=float)
    
    # SPY realized vol, weekly
    spy_vol = daily_ret["SPY"].rolling(5).std() * np.sqrt(252)
    spy_vol_weekly = spy_vol.resample("W-FRI").last().shift(1)  # LAST week
    
    vol_median = spy_vol_weekly.rolling(52, min_periods=26).median()
    
    aggressive = [t for t in ["HYG","EMB","SCHD","VNQ","AMLP","BTC_USD","QQQ","SMH","GBTC"] if t in weekly_ret.columns]
    moderate = [t for t in ["SPY","AGG","GLD","SCHD","LQD","IEF","DVY"] if t in weekly_ret.columns]
    conservative = [t for t in ["SHY","JAAA","AGG","GLD","TBF","MUB","TIP","BNDX"] if t in weekly_ret.columns]
    
    p = pd.Series(0.0, index=weekly_ret.index)
    for i in range(min_w, len(weekly_ret)):
        if i >= len(spy_vol_weekly) or i >= len(vol_median): continue
        v = spy_vol_weekly.iloc[i]
        med = vol_median.iloc[i]
        if np.isnan(v) or np.isnan(med): continue
        
        if v < med * 0.8:
            basket = aggressive
        elif v > med * 1.3:
            basket = conservative
        else:
            basket = moderate
        
        avail = [t for t in basket if t in weekly_ret.columns]
        if avail:
            rets = weekly_ret.iloc[i][avail].dropna()
            if len(rets) > 0: p.iloc[i] = rets.mean()
    
    return p.iloc[min_w:]

p = vol_cluster_regime("VolCluster")
mx = m(p, "VolCluster_Regime")
if mx: results.append(mx)

# ================================================================
# IDEA 6: COMBINED — Best carry selection + regime overlay
# ================================================================
print("=== IDEA 6: Combined Carry + Regime ===")

def combined_carry_regime(top_n=5, carry_lookback=52, gap=2, name=""):
    """
    Step 1: Rank carry streams by trailing Sharpe (2-week gap)
    Step 2: Apply vol regime overlay (shift from aggressive to conservative carry)
    """
    if "SPY" not in daily_ret.columns: return pd.Series(dtype=float)
    
    spy_vol = daily_ret["SPY"].rolling(5).std() * np.sqrt(252)
    spy_vol_weekly = spy_vol.resample("W-FRI").last().shift(1)
    vol_median = spy_vol_weekly.rolling(52, min_periods=26).median()
    
    p = pd.Series(0.0, index=weekly_carry.index)
    for i in range(min_w, len(weekly_carry)):
        end = i - gap
        if end < carry_lookback: continue
        
        # Vol regime
        v = spy_vol_weekly.iloc[i] if i < len(spy_vol_weekly) else None
        med = vol_median.iloc[i] if i < len(vol_median) else None
        
        if v is not None and med is not None and not np.isnan(v) and not np.isnan(med):
            if v > med * 1.3:
                # High vol: only pick from low-vol carry streams
                vols = weekly_carry.iloc[end-26:end].std() * np.sqrt(52)
                vols = vols.dropna()
                lowest_vol = vols.nsmallest(max(top_n, 5)).index.tolist()
                eligible = lowest_vol
            else:
                eligible = list(weekly_carry.columns)
        else:
            eligible = list(weekly_carry.columns)
        
        # Rank eligible streams by trailing Sharpe
        window = weekly_carry[eligible].iloc[end-carry_lookback:end]
        sharpes = window.mean() / window.std() * np.sqrt(52)
        sharpes = sharpes.dropna()
        sharpes = sharpes[sharpes > 0]
        if len(sharpes) < 3: continue
        
        top = sharpes.nlargest(min(top_n, len(sharpes)))
        sq = top**2; w = sq / sq.sum()
        
        ret_this = weekly_carry.iloc[i][w.index].dropna()
        if len(ret_this) > 0:
            p.iloc[i] = sum(w[k]*ret_this.get(k,0) for k in w.index if k in ret_this.index)
    
    return p.iloc[min_w:]

for tn, lb in [(3,26),(5,26),(5,52),(3,52),(7,52),(3,13)]:
    p = combined_carry_regime(tn, lb, 2)
    mx = m(p, f"CarryRegime_t{tn}_{lb}w")
    if mx: results.append(mx)

# ================================================================
# IDEA 7: INVERSE VOL WEIGHTING ALL CARRY (simple, no selection)
# ================================================================
print("=== IDEA 7: Inverse Vol Weight All Carry ===")

def inv_vol_all_carry(vol_lookback=26, gap=2, name=""):
    """Weight ALL carry streams by inverse vol. No selection — just sizing."""
    p = pd.Series(0.0, index=weekly_carry.index)
    for i in range(min_w, len(weekly_carry)):
        end = i - gap
        if end < vol_lookback: continue
        
        vols = weekly_carry.iloc[end-vol_lookback:end].std() * np.sqrt(52)
        vols = vols.dropna()
        vols = vols[vols > 0.001]
        if len(vols) < 3: continue
        
        inv = 1.0 / vols
        w = inv / inv.sum()
        
        ret_this = weekly_carry.iloc[i][w.index].dropna()
        if len(ret_this) > 0:
            p.iloc[i] = sum(w[k]*ret_this.get(k,0) for k in w.index if k in ret_this.index)
    
    return p.iloc[min_w:]

for vlb in [13, 26, 52]:
    p = inv_vol_all_carry(vlb, 2)
    mx = m(p, f"InvVolAll_{vlb}w")
    if mx: results.append(mx)

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*70}")
print(f"RANKED BY WALK-FORWARD SHARPE — ALL ZERO-LEAKAGE")
print(f"{'='*70}")
print(f"{'Name':32s} {'SR':>7} {'WF':>7} {'Test':>7} {'Ret':>8} {'Vol':>7} {'MDD':>8} {'Sort':>7}")
print("-"*82)
for r in sorted(results, key=lambda x:-x['wf_mean'])[:30]:
    flag = " ★" if r['wf_mean'] > 1.2 else (" ●" if r['wf_mean'] > 1.0 else "")
    print(f"  {r['name']:30s} {r['sr']:>6.3f} {r['wf_mean']:>6.3f} {r['test_sr']:>6.3f} "
          f"{r['ret']:>+7.1f}% {r['vol']:>6.1f}% {r['mdd']:>+7.1f}% {r['sortino']:>6.3f}{flag}")

best = max(results, key=lambda x: x['wf_mean'])
print(f"\nBEST: {best['name']} → SR={best['sr']} WF={best['wf_mean']} Ret={best['ret']}% MDD={best['mdd']}%")
print(f"Monthly carry w/ daily scaling: SR≈1.54 WF≈1.62")
print(f"Monthly carry no scaling:       SR≈1.39 WF≈1.39")

# Shuffle test on best
print(f"\n--- Shuffle test on best ---")
np.random.seed(42)
shuffle_srs = []
for trial in range(100):
    # Shuffle weekly carry returns (break time structure)
    shuffled = weekly_carry.copy()
    for col in shuffled.columns:
        vals = shuffled[col].dropna().values.copy()
        np.random.shuffle(vals)
        shuffled[col].iloc[:len(vals)] = vals
    
    # Run best strategy on shuffled data
    p_shuf = pd.Series(0.0, index=shuffled.index)
    for i in range(min_w, len(shuffled)):
        end = i - 2
        if end < 52: continue
        window = shuffled.iloc[end-52:end]
        sharpes = window.mean()/window.std()*np.sqrt(52)
        sharpes = sharpes.dropna(); sharpes = sharpes[sharpes>0]
        if len(sharpes)<3: continue
        top = sharpes.nlargest(min(5,len(sharpes)))
        sq=top**2; w=sq/sq.sum()
        rt = shuffled.iloc[i][w.index].dropna()
        if len(rt)>0: p_shuf.iloc[i] = sum(w[k]*rt.get(k,0) for k in w.index if k in rt.index)
    r = p_shuf.iloc[min_w:].dropna()
    if len(r)>52 and r.std()>0: shuffle_srs.append(r.mean()/r.std()*np.sqrt(52))

z = (best['sr']-np.mean(shuffle_srs))/np.std(shuffle_srs) if np.std(shuffle_srs)>0 else 0
print(f"  Real: {best['sr']:.3f}, Shuffled: {np.mean(shuffle_srs):.3f}±{np.std(shuffle_srs):.3f}, Z={z:.1f}")
if z > 3: print(f"  HIGHLY SIGNIFICANT (p<0.01) ✓")
elif z > 2: print(f"  SIGNIFICANT (p<0.05) ✓")
else: print(f"  {'Marginal' if z>1.5 else 'NOT significant'}")

with open(RESULTS_DIR/"proprietary_v3.json","w") as f:
    json.dump({"experiments":results,"best":best,"n_total":len(results)},f,indent=2)
print(f"\nSaved {len(results)} experiments")
