#!/usr/bin/env python3
"""
EXPERIMENT: Pure monthly rebalance — NO daily adjustments.
Goal: match or beat Sharpe 1.67 with truly monthly execution.

The problem: daily vol targeting adds ~0.5 Sharpe. Without it,
we get Sharpe ~1.16. We need to close that gap.

Ideas:
1. Set vol target at rebalance and FREEZE for 21 days
2. Use longer vol lookback (more stable, less need for daily update)
3. Adjust stream weights to account for vol differences at rebalance
4. Over-allocate to low-vol streams (they need less scaling)
5. Use vol-of-vol to set a buffer (oversize when vol is stable)
6. Target a different portfolio vol level
7. Combine carry streams into one blended position (reduces vol)
8. Use different number of positions
9. Pre-scale by inverse vol AT REBALANCE (one-time calculation)
"""
import pandas as pd, numpy as np, sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

from strategy_v11 import (load_all_data, generate_all_streams,
                           EVAL_WINDOW, MIN_TRAILING_SHARPE)

prices, fred = load_all_data()
ret = prices.pct_change()
streams = generate_all_streams(ret, fred)

# Raw stream returns (no vol targeting)
raw_df = pd.DataFrame(streams).dropna(how="all").dropna(thresh=5).fillna(0)

# Vol-targeted for evaluation (same as current strategy uses for ranking)
vol_t = pd.DataFrame(index=raw_df.index)
for col in raw_df.columns:
    rv = raw_df[col].rolling(63,min_periods=21).std()*np.sqrt(252)
    sc = (0.03/rv.clip(lower=0.003)).clip(0.1,8.0)
    vol_t[col] = raw_df[col]*sc.shift(1)
vol_t = vol_t.fillna(0)

def m(r, name=""):
    r=r.dropna()
    if len(r)<60: return None
    ar=r.mean()*252; av=r.std()*np.sqrt(252); sr=ar/av if av>0 else 0
    cum=(1+r).cumprod(); mdd=((cum-cum.cummax())/cum.cummax()).min()
    wr=(r>0).mean()
    ds=r[r<0].std()*np.sqrt(252) if (r<0).any() else av
    sortino=ar/ds if ds>0 else 0
    sp=int(len(r)*0.6)
    test_sr = r.iloc[sp:].mean()/r.iloc[sp:].std()*np.sqrt(252) if r.iloc[sp:].std()>0 else 0
    nt=len(r); fs=nt//6; wf=[]
    for fold in range(5):
        s=(fold+1)*fs; e=min(s+fs,nt); fr=r.iloc[s:e]
        if len(fr)>60 and fr.std()>0: wf.append(fr.mean()/fr.std()*np.sqrt(252))
    return {"name":name,"sr":round(sr,3),"ret":round(ar*100,2),"vol":round(av*100,2),
            "mdd":round(mdd*100,2),"sortino":round(sortino,3),"wr":round(wr*100,1),
            "test_sr":round(test_sr,3),"wf_mean":round(np.mean(wf),3) if wf else 0,
            "ac1":round(r.autocorr(1),4),"nav":round(float(cum.iloc[-1]),1)}

min_warmup = 504
results = []

def run_monthly_only(top_n=5, target_vol=0.10, weighting="sharpe_sq",
                      rebal_freq=21, min_sharpe=0.75,
                      vol_method="rebal_freeze", vol_lookback=63,
                      stream_target=0.03, vix_at_rebal=False,
                      name=""):
    """
    Pure monthly: ALL decisions made at rebalance, NOTHING changes between.
    
    vol_method options:
      "none" — use raw returns, scale portfolio vol at rebalance only
      "rebal_freeze" — vol-target streams at rebalance, freeze scalers for 21d
      "long_lookback" — use 126 or 252 day vol (changes slowly)
      "inverse_vol_weight" — weight by inverse vol instead of Sharpe
    """
    p = pd.Series(0.0, index=raw_df.index)
    start = min_warmup
    cw = pd.Series(0.0, index=raw_df.columns)
    frozen_scalers = pd.Series(1.0, index=raw_df.columns)
    port_scale = 1.0
    
    for i in range(start, len(raw_df)):
        if (i-start) % rebal_freq == 0:
            # === ALL DECISIONS HAPPEN HERE ===
            
            # 1. Evaluate streams using vol-targeted returns (for fair comparison)
            ev = vol_t.iloc[max(0,i-EVAL_WINDOW):i]
            ts = {}
            for col in vol_t.columns:
                s = ev[col]
                if s.std()>0 and s.count()>=63:
                    ts[col] = s.mean()/s.std()*np.sqrt(252)
            sel = {k:v for k,v in ts.items() if v>min_sharpe}
            
            if sel:
                sv = pd.Series(sel).nlargest(top_n)
                sv = sv.clip(upper=sv.quantile(0.9) if len(sv)>3 else 99)
                
                if weighting == "sharpe_sq":
                    sq = sv**2; w = sq/sq.sum()
                elif weighting == "inverse_vol":
                    # Weight by inverse realized vol of raw streams
                    vols = raw_df[sv.index].iloc[max(0,i-vol_lookback):i].std()*np.sqrt(252)
                    inv = 1.0/vols.clip(lower=0.001)
                    w = inv/inv.sum()
                else:
                    w = sv/sv.sum()
                
                cw = pd.Series(0.0, index=raw_df.columns)
                for k,v in w.items(): cw[k] = v
            else:
                cw = pd.Series(0.0, index=raw_df.columns)
            
            # 2. Compute vol scalers at rebalance (frozen for 21 days)
            if vol_method == "rebal_freeze":
                for col in cw.index:
                    if cw[col] > 0:
                        rv = raw_df[col].iloc[max(0,i-vol_lookback):i].std()*np.sqrt(252)
                        if rv > 0.003:
                            frozen_scalers[col] = min(8.0, max(0.1, stream_target/rv))
                        else:
                            frozen_scalers[col] = 1.0
            elif vol_method == "none":
                frozen_scalers = pd.Series(1.0, index=raw_df.columns)
            
            # 3. Compute portfolio-level vol scale at rebalance
            recent_port = p.iloc[max(0,i-63):i]
            if len(recent_port) > 21 and recent_port.std() > 0:
                recent_vol = recent_port.std() * np.sqrt(252)
                port_scale = min(5.0, max(0.2, target_vol / recent_vol))
            
            # 4. Optional VIX check at rebalance only
            vix_scale = 1.0
            if vix_at_rebal:
                vix = fred.get("VIXCLS")
                if vix is not None:
                    vix_val = vix.iloc[:i].dropna().iloc[-1] if len(vix.iloc[:i].dropna()) > 0 else 20
                    vix_pctl = vix.iloc[max(0,i-252):i].dropna().rank(pct=True).iloc[-1] if len(vix.iloc[max(0,i-252):i].dropna()) > 10 else 0.5
                    vix_scale = max(0.5, min(1.2, 1.2 - 0.6*vix_pctl))
        
        # === BETWEEN REBALANCES: everything is frozen ===
        daily_ret = (cw * raw_df.iloc[i] * frozen_scalers).sum()
        p.iloc[i] = daily_ret * port_scale * vix_scale
    
    return p.iloc[start:].dropna()

print("="*80)
print("PURE MONTHLY EXPERIMENTS")
print("="*80)

# Baseline: current strategy (with daily scaling)
baseline = pd.read_csv(DATA_DIR/"results"/"dichs_returns.csv", parse_dates=[0])
baseline.columns=["Date","return"]; baseline=baseline.set_index("Date")["return"]
bm = m(baseline, "Current (daily scaling)")
print(f"\nBaseline (daily scaling): Sharpe={bm['sr']}  Ret={bm['ret']}%  MDD={bm['mdd']}%  NAV={bm['nav']}x")

configs = [
    # Basic monthly: freeze vol scalers at rebalance
    ("Monthly_freeze_63d", 5, 0.10, "sharpe_sq", 21, 0.75, "rebal_freeze", 63, 0.03, False),
    ("Monthly_freeze_126d", 5, 0.10, "sharpe_sq", 21, 0.75, "rebal_freeze", 126, 0.03, False),
    ("Monthly_freeze_252d", 5, 0.10, "sharpe_sq", 21, 0.75, "rebal_freeze", 252, 0.03, False),
    
    # No stream vol targeting at all — just raw returns with portfolio vol target
    ("Monthly_no_vol", 5, 0.10, "sharpe_sq", 21, 0.75, "none", 63, 0.03, False),
    
    # Inverse-vol weighting (let the weights handle vol differences)
    ("Monthly_invvol_wt", 5, 0.10, "inverse_vol", 21, 0.75, "rebal_freeze", 63, 0.03, False),
    
    # Higher stream vol target (less scaling needed = less daily sensitivity)
    ("Monthly_sv5", 5, 0.10, "sharpe_sq", 21, 0.75, "rebal_freeze", 63, 0.05, False),
    ("Monthly_sv7", 5, 0.10, "sharpe_sq", 21, 0.75, "rebal_freeze", 63, 0.07, False),
    ("Monthly_sv10", 5, 0.10, "sharpe_sq", 21, 0.75, "rebal_freeze", 63, 0.10, False),
    
    # Different portfolio vol targets
    ("Monthly_tv8", 5, 0.08, "sharpe_sq", 21, 0.75, "rebal_freeze", 63, 0.03, False),
    ("Monthly_tv12", 5, 0.12, "sharpe_sq", 21, 0.75, "rebal_freeze", 63, 0.03, False),
    ("Monthly_tv15", 5, 0.15, "sharpe_sq", 21, 0.75, "rebal_freeze", 63, 0.03, False),
    
    # VIX check at rebalance only
    ("Monthly_freeze_vix", 5, 0.10, "sharpe_sq", 21, 0.75, "rebal_freeze", 63, 0.03, True),
    ("Monthly_freeze_126d_vix", 5, 0.10, "sharpe_sq", 21, 0.75, "rebal_freeze", 126, 0.03, True),
    
    # Fewer positions (less rebalancing complexity)
    ("Monthly_top3", 3, 0.10, "sharpe_sq", 21, 0.75, "rebal_freeze", 63, 0.03, False),
    ("Monthly_top4", 4, 0.10, "sharpe_sq", 21, 0.75, "rebal_freeze", 63, 0.03, False),
    ("Monthly_top7", 7, 0.10, "sharpe_sq", 21, 0.75, "rebal_freeze", 63, 0.03, False),
    
    # Min Sharpe thresholds
    ("Monthly_ms0.5", 5, 0.10, "sharpe_sq", 21, 0.50, "rebal_freeze", 63, 0.03, False),
    ("Monthly_ms1.0", 5, 0.10, "sharpe_sq", 21, 1.00, "rebal_freeze", 63, 0.03, False),
    
    # Combined best guesses
    ("Monthly_best_v1", 5, 0.10, "sharpe_sq", 21, 0.75, "rebal_freeze", 126, 0.03, True),
    ("Monthly_best_v2", 5, 0.12, "sharpe_sq", 21, 0.75, "rebal_freeze", 126, 0.04, True),
    ("Monthly_best_v3", 4, 0.10, "sharpe_sq", 21, 1.0, "rebal_freeze", 126, 0.03, True),
    ("Monthly_best_v4", 5, 0.10, "inverse_vol", 21, 0.75, "rebal_freeze", 126, 0.03, True),
    ("Monthly_best_v5", 5, 0.10, "sharpe_sq", 21, 0.75, "rebal_freeze", 126, 0.05, True),
    ("Monthly_best_v6", 3, 0.12, "sharpe_sq", 21, 1.0, "rebal_freeze", 126, 0.04, True),
    ("Monthly_best_v7", 5, 0.10, "sharpe_sq", 21, 0.75, "rebal_freeze", 252, 0.03, True),
    ("Monthly_best_v8", 5, 0.15, "sharpe_sq", 21, 0.75, "rebal_freeze", 252, 0.05, False),
]

for name, tn, tv, wt, rf, ms, vm, vl, sv, vix in configs:
    p = run_monthly_only(top_n=tn, target_vol=tv, weighting=wt, rebal_freq=rf,
                          min_sharpe=ms, vol_method=vm, vol_lookback=vl,
                          stream_target=sv, vix_at_rebal=vix, name=name)
    mx = m(p, name)
    if mx:
        results.append(mx)

# Summary
print(f"\n{'='*80}")
print(f"ALL EXPERIMENTS RANKED BY SHARPE (Walk-Forward)")
print(f"{'='*80}")
print(f"{'Name':25s} {'SR':>7} {'WF SR':>7} {'Test':>7} {'Ret':>8} {'Vol':>7} {'MDD':>8} {'NAV':>7}")
print("-"*80)
for r in sorted(results, key=lambda x:-x['wf_mean']):
    flag = " ★" if r['wf_mean'] >= bm['wf_mean'] * 0.95 else ""
    print(f"  {r['name']:23s} {r['sr']:>6.3f} {r['wf_mean']:>6.3f} {r['test_sr']:>6.3f} "
          f"{r['ret']:>+7.1f}% {r['vol']:>6.1f}% {r['mdd']:>+7.1f}% {r['nav']:>6.1f}x{flag}")

best = max(results, key=lambda x: x['wf_mean'])
print(f"\nBEST MONTHLY: {best['name']} → SR={best['sr']} WF={best['wf_mean']} Ret={best['ret']}%")
print(f"BASELINE:     Current → SR={bm['sr']} WF={bm['wf_mean']} Ret={bm['ret']}%")
gap = best['wf_mean'] - bm['wf_mean']
print(f"GAP: {gap:+.3f} Sharpe ({gap/bm['wf_mean']*100:+.1f}%)")

with open(RESULTS_DIR/"monthly_only_experiments.json","w") as f:
    json.dump({"experiments":results,"baseline":{"name":"Current (daily)","sr":bm['sr'],
               "wf_mean":bm['wf_mean'],"ret":bm['ret']},"best":best},f,indent=2)

print(f"\nSaved {len(results)} experiments")
