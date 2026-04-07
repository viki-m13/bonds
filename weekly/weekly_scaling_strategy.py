#!/usr/bin/env python3
"""
WEEKLY SCALING STRATEGY
========================
Core insight: daily vol scaling is the main alpha driver.
What if we build a strategy that explicitly rebalances weekly
with vol scaling as the PRIMARY mechanism, not an afterthought?

Design principles:
- Rebalance every 5 trading days (weekly)
- Vol-target each stream at rebalance, freeze for 5 days
- Vol-target the portfolio at rebalance, freeze for 5 days
- VIX check at rebalance (not daily)
- Drawdown check at rebalance (not daily)
- Stream selection also weekly (not monthly)
- Everything computed once per week, zero daily adjustments

The key difference from our "pure monthly" test:
- 4x more frequent scaling updates (weekly vs monthly)
- Stream selection also weekly (can rotate faster)
- But still zero intra-week adjustments
"""
import pandas as pd, numpy as np, sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

DATA_DIR = Path(__file__).parent.parent / "data"
ETF_DIR = DATA_DIR / "etfs"
FRED_PATH = DATA_DIR / "fred" / "_combined_fred.csv"
RESULTS_DIR = Path(__file__).parent / "results"

from strategy_v11 import load_all_data, generate_all_streams, EVAL_WINDOW

prices, fred = load_all_data()
ret = prices.pct_change()
streams = generate_all_streams(ret, fred)

# Raw stream returns
raw_df = pd.DataFrame(streams).dropna(how="all").dropna(thresh=5).fillna(0)

# Vol-targeted (for evaluation/ranking)
vol_t = pd.DataFrame(index=raw_df.index)
for col in raw_df.columns:
    rv = raw_df[col].rolling(63,min_periods=21).std()*np.sqrt(252)
    sc = (0.03/rv.clip(lower=0.003)).clip(0.1,8.0)
    vol_t[col] = raw_df[col]*sc.shift(1)
vol_t = vol_t.fillna(0)

min_warmup = 504

def m(r, name=""):
    r=r.dropna()
    if len(r)<60: return None
    ar=r.mean()*252; av=r.std()*np.sqrt(252); sr=ar/av if av>0 else 0
    cum=(1+r).cumprod(); mdd=((cum-cum.cummax())/cum.cummax()).min()
    cal=ar/abs(mdd) if mdd!=0 else 0; wr=(r>0).mean()
    ds=r[r<0].std()*np.sqrt(252) if (r<0).any() else av
    sortino=ar/ds if ds>0 else 0
    total = cum.iloc[-1]-1
    sp=int(len(r)*0.6)
    test_sr = r.iloc[sp:].mean()/r.iloc[sp:].std()*np.sqrt(252) if r.iloc[sp:].std()>0 else 0
    nt=len(r); fs=nt//6; wf=[]
    for fold in range(5):
        s=(fold+1)*fs; e=min(s+fs,nt); fr=r.iloc[s:e]
        if len(fr)>60 and fr.std()>0: wf.append(fr.mean()/fr.std()*np.sqrt(252))
    return {"name":name,"sr":round(sr,3),"sortino":round(sortino,3),
            "ret":round(ar*100,2),"vol":round(av*100,2),
            "mdd":round(mdd*100,2),"calmar":round(cal,3),"wr":round(wr*100,1),
            "test_sr":round(test_sr,3),"wf_mean":round(np.mean(wf),3) if wf else 0,
            "ac1":round(r.autocorr(1),4),"nav":round(float(cum.iloc[-1]),2),
            "total_ret":round(total*100,1)}


def run_weekly(top_n=5, target_vol=0.10, weighting="sharpe_sq",
               min_sharpe=0.75, vol_lookback=63, stream_target=0.03,
               vix_scale=True, dd_control=True, dd_strength=5,
               eval_window=252, name=""):
    """
    Pure weekly rebalance. EVERYTHING happens every 5 trading days.
    Zero adjustments between weekly rebalances.
    """
    p = pd.Series(0.0, index=raw_df.index)
    start = min_warmup
    
    cw = pd.Series(0.0, index=raw_df.columns)
    frozen_scalers = pd.Series(1.0, index=raw_df.columns)
    total_mult = 1.0
    
    for i in range(start, len(raw_df)):
        if (i-start) % 5 == 0:  # WEEKLY
            # 1. SELECT STREAMS
            ev = vol_t.iloc[max(0,i-eval_window):i]
            scores = {}
            for col in vol_t.columns:
                s = ev[col]
                if s.std()>0 and s.count()>=63:
                    scores[col] = s.mean()/s.std()*np.sqrt(252)
            sel = {k:v for k,v in scores.items() if v>min_sharpe}
            
            if sel:
                sv = pd.Series(sel).nlargest(top_n)
                sv = sv.clip(upper=sv.quantile(0.9) if len(sv)>3 else 99)
                if weighting == "sharpe_sq":
                    sq = sv**2; w = sq/sq.sum()
                elif weighting == "sharpe":
                    w = sv/sv.sum()
                elif weighting == "equal":
                    w = pd.Series(1.0/len(sv), index=sv.index)
                elif weighting == "inverse_vol":
                    vols = raw_df[sv.index].iloc[max(0,i-vol_lookback):i].std()*np.sqrt(252)
                    inv = 1.0/vols.clip(lower=0.001)
                    w = inv/inv.sum()
                else:
                    w = sv/sv.sum()
                cw = pd.Series(0.0, index=raw_df.columns)
                for k,v in w.items(): cw[k] = v
            else:
                cw = pd.Series(0.0, index=raw_df.columns)
            
            # 2. VOL SCALERS (frozen for 5 days)
            for col in cw.index:
                if cw[col] > 0:
                    rv = raw_df[col].iloc[max(0,i-vol_lookback):i].std()*np.sqrt(252)
                    if rv > 0.003:
                        frozen_scalers[col] = min(8.0, max(0.1, stream_target/rv))
                    else:
                        frozen_scalers[col] = 1.0
            
            # 3. PORTFOLIO VOL SCALE (frozen for 5 days)
            port_scale = 1.0
            recent = p.iloc[max(0,i-63):i]
            if len(recent) > 21 and recent.std() > 0:
                recent_vol = recent.std() * np.sqrt(252)
                port_scale = min(5.0, max(0.2, target_vol / recent_vol))
            
            # 4. VIX SCALE (frozen for 5 days)
            vix_mult = 1.0
            if vix_scale:
                vix = fred.get("VIXCLS")
                if vix is not None:
                    vd = vix.iloc[max(0,i-252):i].dropna()
                    if len(vd) > 126:
                        vix_pctl = vd.rank(pct=True).iloc[-1]
                        vix_mult = max(0.5, min(1.2, 1.2 - 0.6*vix_pctl))
            
            # 5. DRAWDOWN SCALE (frozen for 5 days)
            dd_mult = 1.0
            if dd_control:
                cum = (1+p.iloc[start:i]).cumprod()
                if len(cum) > 0:
                    dd = (cum.iloc[-1] - cum.max()) / cum.max() if cum.max() > 0 else 0
                    dd_mult = max(0.2, min(1.0, np.exp(dd * dd_strength)))
            
            total_mult = port_scale * vix_mult * dd_mult
        
        # BETWEEN REBALANCES: everything frozen
        p.iloc[i] = (cw * raw_df.iloc[i] * frozen_scalers).sum() * total_mult
    
    return p.iloc[start:].dropna()


print("="*80)
print("WEEKLY SCALING STRATEGY — EXTENSIVE EXPERIMENTS")
print("="*80)
print(f"Streams: {len(streams)}, ETFs: {len(prices.columns)}")

# Baselines
daily_ret = pd.read_csv(DATA_DIR/"results"/"dichs_returns.csv", parse_dates=[0])
daily_ret.columns=["Date","return"]; daily_ret=daily_ret.set_index("Date")["return"]
daily_m = m(daily_ret, "DAILY (current)")

results = []

# ================================================================
# MASSIVE EXPERIMENT GRID
# ================================================================
configs = [
    # Basic weekly variants
    ("W_base", 5, 0.10, "sharpe_sq", 0.75, 63, 0.03, True, True, 5, 252),
    ("W_126d", 5, 0.10, "sharpe_sq", 0.75, 126, 0.03, True, True, 5, 252),
    ("W_252d", 5, 0.10, "sharpe_sq", 0.75, 252, 0.03, True, True, 5, 252),
    
    # Vol lookback sweep
    ("W_vl21", 5, 0.10, "sharpe_sq", 0.75, 21, 0.03, True, True, 5, 252),
    ("W_vl42", 5, 0.10, "sharpe_sq", 0.75, 42, 0.03, True, True, 5, 252),
    
    # Stream vol target sweep
    ("W_sv2", 5, 0.10, "sharpe_sq", 0.75, 63, 0.02, True, True, 5, 252),
    ("W_sv4", 5, 0.10, "sharpe_sq", 0.75, 63, 0.04, True, True, 5, 252),
    ("W_sv5", 5, 0.10, "sharpe_sq", 0.75, 63, 0.05, True, True, 5, 252),
    
    # Portfolio vol target
    ("W_tv8", 5, 0.08, "sharpe_sq", 0.75, 63, 0.03, True, True, 5, 252),
    ("W_tv12", 5, 0.12, "sharpe_sq", 0.75, 63, 0.03, True, True, 5, 252),
    ("W_tv15", 5, 0.15, "sharpe_sq", 0.75, 63, 0.03, True, True, 5, 252),
    
    # Min Sharpe
    ("W_ms0.5", 5, 0.10, "sharpe_sq", 0.5, 63, 0.03, True, True, 5, 252),
    ("W_ms1.0", 5, 0.10, "sharpe_sq", 1.0, 63, 0.03, True, True, 5, 252),
    ("W_ms1.5", 5, 0.10, "sharpe_sq", 1.5, 63, 0.03, True, True, 5, 252),
    
    # Top N
    ("W_top3", 3, 0.10, "sharpe_sq", 0.75, 63, 0.03, True, True, 5, 252),
    ("W_top4", 4, 0.10, "sharpe_sq", 0.75, 63, 0.03, True, True, 5, 252),
    ("W_top7", 7, 0.10, "sharpe_sq", 0.75, 63, 0.03, True, True, 5, 252),
    
    # Weighting
    ("W_eq", 5, 0.10, "equal", 0.75, 63, 0.03, True, True, 5, 252),
    ("W_lin", 5, 0.10, "sharpe", 0.75, 63, 0.03, True, True, 5, 252),
    ("W_invvol", 5, 0.10, "inverse_vol", 0.75, 63, 0.03, True, True, 5, 252),
    
    # No VIX / no DD
    ("W_noVIX", 5, 0.10, "sharpe_sq", 0.75, 63, 0.03, False, True, 5, 252),
    ("W_noDD", 5, 0.10, "sharpe_sq", 0.75, 63, 0.03, True, False, 0, 252),
    ("W_noVIX_noDD", 5, 0.10, "sharpe_sq", 0.75, 63, 0.03, False, False, 0, 252),
    
    # DD strength
    ("W_dd3", 5, 0.10, "sharpe_sq", 0.75, 63, 0.03, True, True, 3, 252),
    ("W_dd8", 5, 0.10, "sharpe_sq", 0.75, 63, 0.03, True, True, 8, 252),
    ("W_dd10", 5, 0.10, "sharpe_sq", 0.75, 63, 0.03, True, True, 10, 252),
    
    # Eval window
    ("W_ew126", 5, 0.10, "sharpe_sq", 0.75, 63, 0.03, True, True, 5, 126),
    ("W_ew189", 5, 0.10, "sharpe_sq", 0.75, 63, 0.03, True, True, 5, 189),
    ("W_ew378", 5, 0.10, "sharpe_sq", 0.75, 63, 0.03, True, True, 5, 378),
    
    # Combined best guesses
    ("W_best1", 5, 0.10, "sharpe_sq", 0.75, 63, 0.03, True, True, 5, 252),  # same as base
    ("W_best2", 5, 0.10, "sharpe_sq", 1.0, 126, 0.03, True, True, 5, 252),
    ("W_best3", 4, 0.10, "sharpe_sq", 0.75, 63, 0.04, True, True, 5, 252),
    ("W_best4", 5, 0.12, "sharpe_sq", 0.75, 126, 0.03, True, True, 3, 252),
    ("W_best5", 5, 0.10, "sharpe_sq", 0.75, 42, 0.03, True, True, 5, 252),
    ("W_best6", 5, 0.10, "sharpe_sq", 1.0, 63, 0.03, True, True, 8, 252),
    ("W_best7", 5, 0.10, "sharpe_sq", 0.75, 63, 0.03, False, True, 8, 252),
    ("W_best8", 4, 0.12, "sharpe_sq", 1.0, 63, 0.03, True, True, 5, 252),
    ("W_best9", 5, 0.10, "sharpe_sq", 0.75, 63, 0.03, True, True, 5, 189),
    ("W_best10", 3, 0.10, "sharpe_sq", 1.0, 63, 0.03, True, True, 5, 252),
]

for name, tn, tv, wt, ms, vl, sv, vix, dd, dds, ew in configs:
    p = run_weekly(top_n=tn, target_vol=tv, weighting=wt, min_sharpe=ms,
                   vol_lookback=vl, stream_target=sv, vix_scale=vix,
                   dd_control=dd, dd_strength=dds, eval_window=ew, name=name)
    mx = m(p, name)
    if mx: results.append(mx)

# Summary
print(f"\n{'='*80}")
print(f"RANKED BY WALK-FORWARD SHARPE")
print(f"{'='*80}")
print(f"{'Name':18s} {'SR':>6} {'WF':>6} {'Test':>6} {'Ret':>8} {'Vol':>6} {'MDD':>7} {'Sort':>6} {'NAV':>7} {'AC1':>6}")
print("-"*85)
print(f"  {'DAILY (current)':16s} {daily_m['sr']:>5.3f} {daily_m['wf_mean']:>5.3f} {daily_m['test_sr']:>5.3f} "
      f"{daily_m['ret']:>+7.1f}% {daily_m['vol']:>5.1f}% {daily_m['mdd']:>+6.1f}% {daily_m['sortino']:>5.3f} {daily_m['nav']:>6.1f}x {daily_m['ac1']:>5.3f}")
print("-"*85)

for r in sorted(results, key=lambda x:-x['wf_mean'])[:25]:
    gap = r['wf_mean'] - daily_m['wf_mean']
    flag = " ★" if gap >= -0.1 else (" ●" if gap >= -0.2 else "")
    print(f"  {r['name']:16s} {r['sr']:>5.3f} {r['wf_mean']:>5.3f} {r['test_sr']:>5.3f} "
          f"{r['ret']:>+7.1f}% {r['vol']:>5.1f}% {r['mdd']:>+6.1f}% {r['sortino']:>5.3f} {r['nav']:>6.1f}x {r['ac1']:>5.3f}{flag}")

best = max(results, key=lambda x: x['wf_mean'])
print(f"\nBEST WEEKLY: {best['name']}")
print(f"  Sharpe={best['sr']} WF={best['wf_mean']} Test={best['test_sr']} Ret={best['ret']}% MDD={best['mdd']}%")
print(f"  Gap to daily: WF {best['wf_mean']-daily_m['wf_mean']:+.3f} ({(best['wf_mean']-daily_m['wf_mean'])/daily_m['wf_mean']*100:+.1f}%)")

# Get current positions for best
for name, tn, tv, wt, ms, vl, sv, vix, dd, dds, ew in configs:
    if name == best['name']:
        p = run_weekly(top_n=tn, target_vol=tv, weighting=wt, min_sharpe=ms,
                       vol_lookback=vl, stream_target=sv, vix_scale=vix,
                       dd_control=dd, dd_strength=dds, eval_window=ew, name=name)
        # Save returns
        p.to_csv(RESULTS_DIR/"weekly_best_returns.csv", header=["return"])
        (1+p).cumprod().to_csv(RESULTS_DIR/"weekly_best_cumulative.csv", header=["cumulative"])
        
        # Yearly breakdown
        print(f"\n  Yearly:")
        spy = pd.read_csv(DATA_DIR/"etfs"/"SPY.csv", parse_dates=["Date"]).set_index("Date")["Close"].pct_change()
        common = p.index.intersection(daily_ret.index).intersection(spy.index)
        print(f"  {'Year':>6} {'Weekly':>10} {'Daily':>10} {'SPY':>10}")
        for yr in sorted(set(common.year)):
            mask = common.year == yr
            if mask.sum() < 20: continue
            wr = ((1+p.loc[common[mask]]).prod()-1)*100
            dr = ((1+daily_ret.loc[common[mask]]).prod()-1)*100
            sr = ((1+spy.loc[common[mask]]).prod()-1)*100
            print(f"  {yr:>6} {wr:>+9.1f}% {dr:>+9.1f}% {sr:>+9.1f}%")
        break

with open(RESULTS_DIR/"weekly_experiments.json","w") as f:
    json.dump({"experiments":results,"baseline":daily_m,"best":best,
               "total_configs":len(configs)},f,indent=2)

print(f"\nSaved {len(results)} experiments")
