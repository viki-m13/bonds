#!/usr/bin/env python3
"""
INVENT. No limitations. Think like no one has thought before.
============================================================

Forget carry. Forget momentum. Forget factors. 

What STRUCTURAL features of ETFs can be exploited?

1. LEVERAGED ETF DECAY IS MATHEMATICAL
   TQQQ decays vs 3×QQQ every single day. This is Jensen's inequality.
   Not a signal — a MATHEMATICAL CERTAINTY.
   Can we harvest this WITHOUT shorting? YES:
   Buy QQQ + buy SQQQ in the right ratio = harvest the mutual decay.

2. VOLATILITY IS MEAN-REVERTING
   After a vol spike, vol almost always comes back down.
   The SPEED of mean reversion is predictable.
   Buy assets AFTER vol spikes, sell BEFORE.
   Use realized vol from 2 weeks ago (lagged, no leakage).

3. CALENDAR EFFECTS IN BONDS
   Month-end bond index rebalancing creates predictable flows.
   Pension funds rebalance quarterly.
   These flows are KNOWN in advance — not a prediction, a schedule.

4. CROSS-MARKET LEAD-LAG
   Credit markets (HY OAS, IG spreads) lead equity by 1-3 DAYS.
   But we need WEEKLY lag for no leakage.
   What if we use the VOLATILITY of credit spreads (not level)?
   Spread vol from 2 weeks ago predicts next week's equity vol.

5. THE CARRY-MOMENTUM INTERACTION
   When carry AND momentum agree → very strong signal.
   When they disagree → go to cash.
   This INTERACTION is the alpha, not either signal alone.

6. RELATIVE VALUE ACROSS THE TERM STRUCTURE
   When 2Y-5Y spread is different from 5Y-10Y spread,
   the curve is "kinked." This kink mean-reverts.
   Trade the belly vs wings using bond ETFs.

7. PAIRS OF PAIRS
   Not individual pairs, but pairs OF pairs.
   (HYG-TBF) vs (LQD-TBF) = pure credit quality spread.
   When HY carry outperforms IG carry → credit is strong.
   Use this to time allocation between HY and IG.

8. ENTROPY-BASED REGIME
   Compute the entropy (randomness) of recent returns.
   High entropy = random walk = trend-following works.
   Low entropy = ordered/trending = mean reversion works.
   Switch strategy based on entropy, not vol.

9. OPTIONS-LIKE PAYOFFS WITHOUT OPTIONS
   Combination of ETFs that REPLICATES a covered call:
   Long SPY + Long SH×0.1 = dampened equity with income
   Long TLT + Long TMV×0.05 = dampened bonds with kicker
   These are STATIC positions that produce option-like payoffs.

10. THE ANTI-CORRELATION PREMIUM
    Some ETF pairs are NEGATIVELY correlated.
    Holding both = natural hedge.
    The REBALANCING between them generates return (Shannon's demon).
    Long GLD + Long TLT: negatively correlated in crises.
    Weekly rebalance to 50/50 generates rebalancing premium.
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

weekly_px = prices.resample("W-FRI").last()
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
print("INVENTING — NO LIMITATIONS")
print("="*70)

# ================================================================
# 1. LEVERAGED DECAY HARVESTING (long-only, mathematical edge)
# ================================================================
print("\n=== 1. Leveraged Decay Harvesting ===")
# Buy QQQ + buy SQQQ → when QQQ flat, BOTH decay. Net positive from QQQ.
# When QQQ up, QQQ profit > SQQQ loss. When QQQ down, SQQQ profit > QQQ loss.
# The KEY: the rebalancing to maintain fixed ratio generates the return.

for underlying, inverse_lev, ratio, name in [
    ("QQQ","SQQQ",0.15,"QQQ_SQQQ"), ("SPY","SPXU",0.15,"SPY_SPXU"),
    ("TLT","TMV",0.15,"TLT_TMV"), ("GLD","GLL",0.20,"GLD_GLL"),
    ("QQQ","SQQQ",0.10,"QQQ_SQQQ10"), ("QQQ","SQQQ",0.20,"QQQ_SQQQ20"),
    ("SPY","SPXU",0.10,"SPY_SPXU10"), ("SPY","SPXU",0.20,"SPY_SPXU20"),
]:
    if underlying not in weekly_ret.columns or inverse_lev not in weekly_ret.columns: continue
    w1 = 1.0/(1.0+ratio); w2 = ratio/(1.0+ratio)
    p = w1*weekly_ret[underlying] + w2*weekly_ret[inverse_lev]
    mx = m(p.iloc[min_w:], f"Decay_{name}")
    if mx: results.append(mx); print(f"  {name}: SR={mx['sr']} Ret={mx['ret']}% MDD={mx['mdd']}%")

# ================================================================
# 10. SHANNON'S DEMON (rebalancing premium)
# ================================================================
print("\n=== 10. Shannon's Demon — Rebalancing Premium ===")
# Two uncorrelated or negatively correlated assets, rebalanced weekly to 50/50.
# The rebalancing GENERATES return beyond the average of the two assets.

anti_corr_pairs = [
    (["GLD","TLT"], "GLD_TLT"),
    (["GLD","SHY"], "GLD_SHY"),
    (["TLT","SPY"], "TLT_SPY"),
    (["GLD","SPY"], "GLD_SPY"),
    (["TLT","HYG"], "TLT_HYG"),
    (["GLD","HYG"], "GLD_HYG"),
    (["JAAA","GLD"], "CLO_GLD"),
    (["JAAA","TLT"], "CLO_TLT"),
    (["SHY","QQQ"], "SHY_QQQ"),
    (["GLD","EEM"], "GLD_EEM"),
    (["TBF","HYG"], "TBF_HYG"),
    (["TBF","SCHD"], "TBF_SCHD"),
    (["GLD","SCHD"], "GLD_SCHD"),
    (["JAAA","SCHD"], "CLO_SCHD"),
]

for assets, name in anti_corr_pairs:
    avail = [a for a in assets if a in weekly_ret.columns]
    if len(avail) != len(assets): continue
    p = weekly_ret[avail].mean(axis=1)
    mx = m(p.iloc[min_w:], f"Shannon_{name}")
    if mx: results.append(mx)

# More complex Shannon: 3+ assets, weekly rebalance
print("\n  Multi-asset Shannon:")
multi_shannon = [
    (["GLD","TLT","SHY","JAAA"], "Safe4"),
    (["GLD","TLT","HYG","SPY"], "Mixed4"),
    (["JAAA","GLD","SCHD","AGG"], "Income4"),
    (["GLD","TLT","SCHD","JAAA","AGG"], "Balanced5"),
    (["GLD","TLT","HYG","SPY","EEM","SCHD"], "Global6"),
    (["JAAA","SCHD","GLD","TLT","HYG","EMB","BKLN"], "MaxDiv7"),
    (["GLD","TLT","SHY","JAAA","SCHD","HDV","MUB","AGG","TIP"], "UltraDiv9"),
]

for assets, name in multi_shannon:
    avail = [a for a in assets if a in weekly_ret.columns]
    if len(avail) < 3: continue
    p = weekly_ret[avail].mean(axis=1)
    mx = m(p.iloc[min_w:], f"Shannon_{name}")
    if mx: results.append(mx)

# ================================================================
# 5. CARRY-MOMENTUM INTERACTION
# ================================================================
print("\n=== 5. Carry-Momentum Interaction ===")
# When carry (hedged bond income) AND momentum (equity trend) agree → full risk
# When they disagree → go to cash

def carry_mom_interaction(carry_assets, mom_asset, cash_asset, carry_lookback=26, mom_lookback=13, gap=2, name=""):
    """
    Carry signal: are carry assets (hedged) returning positive? (2-week gap)
    Momentum signal: is the equity trend positive? (2-week gap)
    Both positive → hold carry + equity mix
    Only carry → hold carry only
    Only momentum → hold equity only
    Neither → hold cash
    """
    if mom_asset not in weekly_px.columns or cash_asset not in weekly_ret.columns: 
        return pd.Series(dtype=float)
    
    carry_avail = [a for a in carry_assets if a in weekly_ret.columns]
    if len(carry_avail) < 2: return pd.Series(dtype=float)
    
    p = pd.Series(0.0, index=weekly_ret.index)
    
    for i in range(min_w, len(weekly_ret)):
        end = i - gap
        if end < max(carry_lookback, mom_lookback): continue
        
        # Carry signal: average carry stream return over lookback
        carry_ret_trailing = weekly_ret[carry_avail].iloc[end-carry_lookback:end].mean().mean() * 52
        carry_positive = carry_ret_trailing > 0.01  # > 1% annualized
        
        # Momentum signal: is mom_asset above its moving average?
        if end >= mom_lookback:
            mom_current = weekly_px[mom_asset].iloc[end]
            mom_avg = weekly_px[mom_asset].iloc[end-mom_lookback:end].mean()
            mom_positive = mom_current > mom_avg
        else:
            mom_positive = False
        
        if carry_positive and mom_positive:
            # Both agree → aggressive: 60% carry, 40% equity
            cr = weekly_ret.iloc[i][carry_avail].mean() * 0.6
            mr = weekly_ret.iloc[i].get(mom_asset, 0) * 0.4
            p.iloc[i] = cr + mr
        elif carry_positive:
            # Only carry → conservative carry
            p.iloc[i] = weekly_ret.iloc[i][carry_avail].mean()
        elif mom_positive:
            # Only momentum → equity with hedge
            p.iloc[i] = weekly_ret.iloc[i].get(mom_asset, 0) * 0.5 + weekly_ret.iloc[i].get(cash_asset, 0) * 0.5
        else:
            # Neither → cash
            p.iloc[i] = weekly_ret.iloc[i].get(cash_asset, 0)
    
    return p.iloc[min_w:]

carry_assets = [a for a in ["HYG","LQD","EMB","BKLN","JAAA","SCHD","HDV","MUB","PFF","MBB"] if a in weekly_ret.columns]

for mom, cash, name in [
    ("SPY","SHY","SPY_SHY"), ("QQQ","SHY","QQQ_SHY"),
    ("SPY","JAAA","SPY_CLO"), ("QQQ","JAAA","QQQ_CLO"),
    ("SPY","GLD","SPY_GLD"), ("EEM","SHY","EEM_SHY"),
]:
    p = carry_mom_interaction(carry_assets, mom, cash, 26, 13, 2, name)
    if len(p.dropna()) > 52:
        mx = m(p, f"CarryMom_{name}")
        if mx: results.append(mx)

# ================================================================
# 9. OPTIONS-LIKE PAYOFFS (static positions)
# ================================================================
print("\n=== 9. Options-Like Payoffs ===")
# Covered call analog: Long SPY + small Long SH = dampened equity
# Put-write analog: Long SHY + small Long SSO = cash + upside capture

for assets, weights, name in [
    (["SPY","SH"], [0.9, 0.1], "CoveredCall_SPY"),
    (["QQQ","PSQ"], [0.9, 0.1], "CoveredCall_QQQ"),
    (["SPY","SH"], [0.8, 0.2], "Collar_SPY"),
    (["SHY","SSO"], [0.8, 0.2], "PutWrite_SHY"),
    (["SHY","QLD"], [0.8, 0.2], "PutWrite_QQQ"),
    (["AGG","SSO"], [0.7, 0.3], "BondPlus_SPY"),
    (["JAAA","SSO"], [0.7, 0.3], "CLOPlus_SPY"),
    (["JAAA","QLD"], [0.7, 0.3], "CLOPlus_QQQ"),
    (["GLD","SSO"], [0.6, 0.4], "GoldPlus_SPY"),
    (["SCHD","TBF"], [0.85, 0.15], "DivHedged"),
    (["HDV","TBF"], [0.85, 0.15], "HDVHedged"),
    # Triple combo
    (["JAAA","SCHD","GLD"], [0.4, 0.4, 0.2], "CLO_Div_Gold"),
    (["JAAA","SCHD","TBF"], [0.4, 0.4, 0.2], "CLO_Div_Hedge"),
    (["JAAA","HDV","GLD"], [0.4, 0.4, 0.2], "CLO_HDV_Gold"),
    (["JAAA","QQQ","SH"], [0.3, 0.5, 0.2], "CLO_QQQ_Hedge"),
    (["JAAA","SPY","SH"], [0.3, 0.5, 0.2], "CLO_SPY_Hedge"),
    (["JAAA","SCHD","SH","GLD"], [0.3, 0.3, 0.2, 0.2], "Ultimate4"),
    (["JAAA","SCHD","HDV","GLD","TBF"], [0.25, 0.2, 0.2, 0.2, 0.15], "Ultimate5"),
    (["JAAA","SCHD","HDV","GLD","TBF","BKLN","MUB"], [0.15,0.15,0.15,0.15,0.15,0.15,0.10], "Ultimate7"),
]:
    avail = [(a,w) for a,w in zip(assets,weights) if a in weekly_ret.columns]
    if len(avail) != len(assets): continue
    w_total = sum(w for _,w in avail)
    p = sum(w/w_total * weekly_ret[a] for a,w in avail)
    mx = m(p.iloc[min_w:], f"OptLike_{name}")
    if mx: results.append(mx)

# ================================================================
# COMBINATIONS: Best from each category
# ================================================================
print("\n=== Combinations ===")

# Take the best leveraged decay pair + best Shannon pair + best options-like
# and blend them

# Top carry-mom + top Shannon + top options-like
# Simple: just average the best weekly returns from each category

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*70}")
print(f"ALL INVENTIONS — RANKED BY WF SHARPE")
print(f"{'='*70}")
print(f"{'Name':35s} {'SR':>7} {'WF':>7} {'Test':>7} {'Ret':>8} {'Vol':>7} {'MDD':>8} {'Sort':>7}")
print("-"*85)
for r in sorted(results, key=lambda x:-x['wf_mean'])[:30]:
    flag = " ★" if r['wf_mean'] > 1.0 else (" ●" if r['wf_mean'] > 0.8 else "")
    print(f"  {r['name']:33s} {r['sr']:>6.3f} {r['wf_mean']:>6.3f} {r['test_sr']:>6.3f} "
          f"{r['ret']:>+7.1f}% {r['vol']:>6.1f}% {r['mdd']:>+7.1f}% {r['sortino']:>6.3f}{flag}")

best = max(results, key=lambda x: x['wf_mean'])
best_sr = max(results, key=lambda x: x['sr'])
print(f"\nBest WF: {best['name']} → SR={best['sr']} WF={best['wf_mean']} Ret={best['ret']}% MDD={best['mdd']}%")
print(f"Best SR: {best_sr['name']} → SR={best_sr['sr']} Ret={best_sr['ret']}% MDD={best_sr['mdd']}%")

# Save
with open(RESULTS_DIR/"inventions.json","w") as f:
    json.dump({"experiments":results,"best_wf":best,"best_sr":best_sr,"n":len(results)},f,indent=2)
print(f"\nSaved {len(results)} inventions")
