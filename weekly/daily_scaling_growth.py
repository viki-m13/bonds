#!/usr/bin/env python3
"""
DAILY SCALING GROWTH STRATEGY
===============================
No fixed rebalance schedule. Every single day:
1. Evaluate all streams
2. Pick the best
3. Size by vol
4. Hold until tomorrow, repeat

The key insight: daily scaling IS the alpha. So lean into it fully.
No artificial "hold for 21 days" — continuously ride the best stream.

Variations to test:
A. Continuous top-1 (always in the single best stream)
B. Continuous top-3 with return-squared weighting
C. Daily scaling + momentum filter (only hold when trending)
D. Regime-adaptive: more streams in calm, fewer in crisis
E. Pure vol-targeting: hold everything, just scale by vol
"""
import pandas as pd, numpy as np, sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

from strategy_v11 import load_all_data, generate_all_streams

prices, fred = load_all_data()
ret = prices.pct_change()

# Generate ALL streams (Sharpe universe for broader selection)
streams = generate_all_streams(ret, fred)

# Also add unhedged high-return streams
for etf in ["QQQ","SPY","IWM","TQQQ","UPRO","SOXL","TECL","SMH","ARKK",
            "BTC_USD","ETH_USD","SOL_USD","GLD","VNQ","AMLP",
            "SCHD","HDV","HYG","EMB","JAAA","BKLN","EEM","EFA"]:
    if etf in ret.columns:
        r = ret[etf].dropna()
        if len(r) >= 252: streams[f"unhedged_{etf}"] = r

raw_df = pd.DataFrame(streams).dropna(how="all").dropna(thresh=5).fillna(0)
print(f"Total streams: {len(streams)}")

min_warmup = 504

def m(r, name=""):
    r=r.dropna()
    if len(r)<60: return None
    ar=r.mean()*252; av=r.std()*np.sqrt(252); sr=ar/av if av>0 else 0
    cum=(1+r).cumprod(); mdd=((cum-cum.cummax())/cum.cummax()).min()
    ds=r[r<0].std()*np.sqrt(252) if (r<0).any() else av
    sortino=ar/ds if ds>0 else 0
    sp=int(len(r)*0.6)
    test_sr = r.iloc[sp:].mean()/r.iloc[sp:].std()*np.sqrt(252) if r.iloc[sp:].std()>0 else 0
    nt=len(r); fs=nt//6; wf=[]
    for fold in range(5):
        s=(fold+1)*fs; e=min(s+fs,nt); fr=r.iloc[s:e]
        if len(fr)>60 and fr.std()>0: wf.append(fr.mean()/fr.std()*np.sqrt(252))
    return {"name":name,"sr":round(sr,3),"ret":round(ar*100,2),"vol":round(av*100,2),
            "mdd":round(mdd*100,2),"sortino":round(sortino,3),
            "test_sr":round(test_sr,3),"wf_mean":round(np.mean(wf),3) if wf else 0,
            "nav":round(float(cum.iloc[-1]),2),"ac1":round(r.autocorr(1),4)}


def run_daily_growth(top_n=3, target_vol=0.30, weighting="return_sq",
                      selection="return",  # "return" or "sharpe" or "sortino"
                      eval_window=252, stream_vol=0.05,
                      min_threshold=0.0,
                      vix_scale=False, dd_control=False, dd_strength=3,
                      momentum_filter=False, mom_lookback=63,
                      name=""):
    """
    DAILY decision-making. Every day:
    1. Compute trailing metric for each stream
    2. Pick top N
    3. Weight and scale
    No holding period — can change every day.
    """
    # Vol-target streams
    vol_t = pd.DataFrame(index=raw_df.index)
    for col in raw_df.columns:
        rv = raw_df[col].rolling(63,min_periods=21).std()*np.sqrt(252)
        sc = (stream_vol/rv.clip(lower=0.003)).clip(0.1,10.0)
        vol_t[col] = raw_df[col]*sc.shift(1)
    vol_t = vol_t.fillna(0)
    
    p = pd.Series(0.0, index=vol_t.index)
    start = min_warmup
    
    for i in range(start, len(vol_t)):
        ev = vol_t.iloc[max(0,i-eval_window):i]
        
        scores = {}
        for col in vol_t.columns:
            s = ev[col]
            if s.count() < 63: continue
            
            if selection == "return":
                score = s.mean() * 252
            elif selection == "sharpe":
                score = s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0
            elif selection == "sortino":
                ds = s[s<0].std()*np.sqrt(252) if (s<0).any() else s.std()*np.sqrt(252)
                score = s.mean()*252/ds if ds>0 else 0
            else:
                score = s.mean()*252
            
            if score > min_threshold:
                # Optional momentum filter
                if momentum_filter:
                    recent = s.iloc[-mom_lookback:] if len(s) >= mom_lookback else s
                    if recent.mean() <= 0:
                        continue  # Skip streams with negative recent momentum
                scores[col] = score
        
        if scores:
            sv = pd.Series(scores).nlargest(top_n)
            sv = sv.clip(upper=sv.quantile(0.9) if len(sv)>3 else 99)
            if weighting == "return_sq":
                sq = sv**2; w = sq/sq.sum()
            elif weighting == "sharpe_sq":
                sq = sv**2; w = sq/sq.sum()
            else:
                w = sv/sv.sum()
            p.iloc[i] = sum(w[k]*vol_t.iloc[i][k] for k in w.index)
        
    p = p.iloc[start:]
    
    if vix_scale:
        vix = fred.get("VIXCLS")
        if vix is not None:
            va = vix.reindex(p.index).ffill()
            vp = va.rolling(252,min_periods=126).rank(pct=True)
            p = p*(1.1-0.3*vp).clip(0.7,1.1).shift(1)
    
    if dd_control:
        cum = (1+p).cumprod(); dd = (cum-cum.cummax())/cum.cummax()
        p = p*np.exp(dd*dd_strength).clip(0.3,1.0).shift(1)
    
    pv = p.rolling(63,min_periods=21).std()*np.sqrt(252)
    ps = (target_vol/pv.clip(lower=0.005)).clip(0.2,8.0)
    p = p*ps.shift(1)
    
    return p.dropna()


print("="*80)
print("DAILY SCALING GROWTH — NO FIXED REBALANCE SCHEDULE")
print("="*80)

results = []

configs = [
    # (name, top_n, tv, wt, sel, ew, sv, min_t, vix, dd, dds, mom_filter, mom_lb)
    # Core variations
    ("D_top1_ret", 1, 0.30, "return_sq", "return", 252, 0.05, 0, False, False, 0, False, 0),
    ("D_top3_ret", 3, 0.30, "return_sq", "return", 252, 0.05, 0, False, False, 0, False, 0),
    ("D_top5_ret", 5, 0.30, "return_sq", "return", 252, 0.05, 0, False, False, 0, False, 0),
    ("D_top3_sr", 3, 0.30, "sharpe_sq", "sharpe", 252, 0.05, 0, False, False, 0, False, 0),
    ("D_top5_sr", 5, 0.30, "sharpe_sq", "sharpe", 252, 0.05, 0, False, False, 0, False, 0),
    
    # Vol target sweep
    ("D_top3_tv20", 3, 0.20, "return_sq", "return", 252, 0.05, 0, False, False, 0, False, 0),
    ("D_top3_tv40", 3, 0.40, "return_sq", "return", 252, 0.05, 0, False, False, 0, False, 0),
    ("D_top3_tv50", 3, 0.50, "return_sq", "return", 252, 0.05, 0, False, False, 0, False, 0),
    
    # Eval window
    ("D_top3_ew126", 3, 0.30, "return_sq", "return", 126, 0.05, 0, False, False, 0, False, 0),
    ("D_top3_ew63", 3, 0.30, "return_sq", "return", 63, 0.05, 0, False, False, 0, False, 0),
    
    # Momentum filter
    ("D_top3_mom21", 3, 0.30, "return_sq", "return", 252, 0.05, 0, False, False, 0, True, 21),
    ("D_top3_mom63", 3, 0.30, "return_sq", "return", 252, 0.05, 0, False, False, 0, True, 63),
    
    # With light risk controls
    ("D_top3_dd", 3, 0.30, "return_sq", "return", 252, 0.05, 0, False, True, 3, False, 0),
    ("D_top3_vix", 3, 0.30, "return_sq", "return", 252, 0.05, 0, True, False, 0, False, 0),
    ("D_top3_both", 3, 0.30, "return_sq", "return", 252, 0.05, 0, True, True, 3, False, 0),
    
    # Sharpe-optimized daily (for comparison)
    ("D_top5_sr10", 5, 0.10, "sharpe_sq", "sharpe", 252, 0.03, 0.75, True, True, 5, False, 0),
    
    # High stream vol (less scaling needed)
    ("D_top3_sv8", 3, 0.30, "return_sq", "return", 252, 0.08, 0, False, False, 0, False, 0),
    ("D_top3_sv3", 3, 0.30, "return_sq", "return", 252, 0.03, 0, False, False, 0, False, 0),
    
    # Sortino selection
    ("D_top3_sortino", 3, 0.30, "return_sq", "sortino", 252, 0.05, 0, False, False, 0, False, 0),
    
    # Ultra aggressive
    ("D_top1_tv50", 1, 0.50, "return_sq", "return", 126, 0.08, 0, False, False, 0, False, 0),
    ("D_top1_tv50_mom", 1, 0.50, "return_sq", "return", 126, 0.08, 0, False, False, 0, True, 21),
    
    # Best combos
    ("D_best1", 3, 0.30, "return_sq", "return", 252, 0.05, 0, False, True, 3, True, 21),
    ("D_best2", 3, 0.30, "return_sq", "return", 126, 0.05, 0, True, True, 3, True, 21),
    ("D_best3", 5, 0.30, "sharpe_sq", "sharpe", 252, 0.05, 0.5, True, True, 3, False, 0),
    ("D_best4", 3, 0.40, "return_sq", "return", 252, 0.05, 0, False, True, 3, False, 0),
    ("D_best5", 1, 0.30, "return_sq", "return", 252, 0.05, 0, False, True, 3, True, 63),
]

for name, tn, tv, wt, sel, ew, sv, mt, vix, dd, dds, mf, ml in configs:
    p = run_daily_growth(top_n=tn, target_vol=tv, weighting=wt, selection=sel,
                          eval_window=ew, stream_vol=sv, min_threshold=mt,
                          vix_scale=vix, dd_control=dd, dd_strength=dds,
                          momentum_filter=mf, mom_lookback=ml, name=name)
    mx = m(p, name)
    if mx: results.append(mx)

# Baselines
daily_ret = pd.read_csv(DATA_DIR/"results"/"dichs_returns.csv", parse_dates=[0])
daily_ret.columns=["Date","return"]; daily_ret=daily_ret.set_index("Date")["return"]
daily_m = m(daily_ret, "Sharpe Daily")

growth_ret = pd.read_csv(Path(__file__).parent/"results"/"growth_best_returns.csv", parse_dates=[0])
growth_ret.columns=["Date","return"]; growth_ret=growth_ret.set_index("Date")["return"]
growth_m = m(growth_ret, "Growth V1")

spy = pd.read_csv(DATA_DIR/"etfs"/"SPY.csv", parse_dates=["Date"]).set_index("Date")["Close"].pct_change().dropna()
spy_m = m(spy, "SPY")

# Summary
print(f"\n{'='*80}")
print(f"RANKED BY ANNUALIZED RETURN")
print(f"{'='*80}")
print(f"{'Name':20s} {'Ret':>8} {'SR':>6} {'WF':>6} {'Vol':>6} {'MDD':>7} {'Sort':>6} {'NAV':>8} {'AC1':>6}")
print("-"*80)
print(f"  {'SPY':18s} {spy_m['ret']:>+7.1f}% {spy_m['sr']:>5.3f} {spy_m['wf_mean']:>5.3f} {spy_m['vol']:>5.1f}% {spy_m['mdd']:>+6.1f}% {spy_m['sortino']:>5.3f} {spy_m['nav']:>7.1f}x {spy_m['ac1']:>5.3f}")
print(f"  {'Sharpe Daily':18s} {daily_m['ret']:>+7.1f}% {daily_m['sr']:>5.3f} {daily_m['wf_mean']:>5.3f} {daily_m['vol']:>5.1f}% {daily_m['mdd']:>+6.1f}% {daily_m['sortino']:>5.3f} {daily_m['nav']:>7.1f}x {daily_m['ac1']:>5.3f}")
print(f"  {'Growth V1':18s} {growth_m['ret']:>+7.1f}% {growth_m['sr']:>5.3f} {growth_m['wf_mean']:>5.3f} {growth_m['vol']:>5.1f}% {growth_m['mdd']:>+6.1f}% {growth_m['sortino']:>5.3f} {growth_m['nav']:>7.1f}x {growth_m['ac1']:>5.3f}")
print("-"*80)

for r in sorted(results, key=lambda x:-x['ret'])[:20]:
    flag = " ★" if r['ret'] > growth_m['ret'] else ""
    print(f"  {r['name']:18s} {r['ret']:>+7.1f}% {r['sr']:>5.3f} {r['wf_mean']:>5.3f} {r['vol']:>5.1f}% {r['mdd']:>+6.1f}% {r['sortino']:>5.3f} {r['nav']:>7.1f}x {r['ac1']:>5.3f}{flag}")

best_ret = max(results, key=lambda x: x['ret'])
best_wf = max(results, key=lambda x: x['wf_mean'])
print(f"\nBest by RETURN: {best_ret['name']} → {best_ret['ret']:+.1f}% (NAV {best_ret['nav']}x, MDD {best_ret['mdd']}%)")
print(f"Best by WF:     {best_wf['name']} → WF={best_wf['wf_mean']} Ret={best_wf['ret']:+.1f}%")

# Save best
for name, tn, tv, wt, sel, ew, sv, mt, vix, dd, dds, mf, ml in configs:
    if name == best_ret['name']:
        p = run_daily_growth(top_n=tn, target_vol=tv, weighting=wt, selection=sel,
                              eval_window=ew, stream_vol=sv, min_threshold=mt,
                              vix_scale=vix, dd_control=dd, dd_strength=dds,
                              momentum_filter=mf, mom_lookback=ml, name=name)
        p.to_csv(RESULTS_DIR/"daily_growth_best_returns.csv", header=["return"])
        
        # Yearly
        common = p.index.intersection(spy.index)
        print(f"\n  Yearly:")
        print(f"  {'Year':>6} {'DailyGrowth':>12} {'GrowthV1':>12} {'SPY':>10}")
        for yr in sorted(set(common.year)):
            mask = common.year == yr
            if mask.sum() < 20: continue
            dg = ((1+p.loc[common[mask]]).prod()-1)*100
            gv = ((1+growth_ret.reindex(common[mask]).dropna()).prod()-1)*100 if len(growth_ret.reindex(common[mask]).dropna())>0 else 0
            sr = ((1+spy.loc[common[mask]]).prod()-1)*100
            print(f"  {yr:>6} {dg:>+11.1f}% {gv:>+11.1f}% {sr:>+9.1f}%")
        break

with open(RESULTS_DIR/"daily_growth_experiments.json","w") as f:
    json.dump({"experiments":results,"baselines":{"sharpe_daily":daily_m,"growth_v1":growth_m,"spy":spy_m},
               "best_return":best_ret,"best_wf":best_wf},f,indent=2)

print(f"\nSaved {len(results)} experiments")
