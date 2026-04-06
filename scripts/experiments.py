#!/usr/bin/env python3
"""
COMPREHENSIVE STRATEGY IMPROVEMENT EXPERIMENTS
================================================
Goal: Push toward Sharpe 3 while staying long-only, concentrated,
no leakage, no bias, no overfitting.

EXPERIMENT AREAS:
A. Better carry stream construction (dynamic hedge ratios)
B. Better signal for stream selection (multi-factor vs pure Sharpe)
C. Better portfolio construction (risk parity, max diversification)
D. New alpha sources (options-like payoffs, term structure, seasonality)
E. Better risk management (regime-aware sizing, tail hedging)
F. Higher-frequency signals within monthly rebalance
G. Combining carry with short-term mean reversion
"""

import pandas as pd
import numpy as np
import sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path("/home/user/bonds/data")
RESULTS = []

def load():
    from strategy_v11 import load_all_data, generate_all_streams
    prices, fred = load_all_data()
    ret = prices.pct_change()
    streams = generate_all_streams(ret, fred)
    return prices, fred, ret, streams

def vol_target_streams(streams, target=0.03):
    df = pd.DataFrame(streams).dropna(how="all").dropna(thresh=5).fillna(0)
    vol_t = pd.DataFrame(index=df.index)
    for col in df.columns:
        rv = df[col].rolling(63,min_periods=21).std()*np.sqrt(252)
        sc = (target/rv.clip(lower=0.003)).clip(0.1,8.0)
        vol_t[col] = df[col]*sc.shift(1)
    return vol_t.fillna(0)

def run_adaptive(vol_t, fred, top_n=5, rebal_freq=21, eval_window=252,
                  min_warmup=504, target_vol=0.10, weighting="sharpe",
                  min_sharpe=0.0, dd_strength=5, vix_scale=True):
    portfolio_ret = pd.Series(0.0, index=vol_t.index)
    start_idx = min_warmup
    if start_idx >= len(vol_t): return None
    cw = pd.Series(0.0, index=vol_t.columns)
    
    for i in range(start_idx, len(vol_t)):
        if (i-start_idx) % rebal_freq == 0:
            eval_data = vol_t.iloc[max(0,i-eval_window):i]
            ts = {}
            for col in vol_t.columns:
                s = eval_data[col]
                if s.std()>0 and s.count()>=63:
                    ts[col] = s.mean()/s.std()*np.sqrt(252)
            pos = {k:v for k,v in ts.items() if v>min_sharpe}
            if pos:
                sv = pd.Series(pos).nlargest(top_n)
                sv = sv.clip(upper=sv.quantile(0.9) if len(sv)>3 else 99)
                
                if weighting == "equal":
                    w = pd.Series(1.0/len(sv), index=sv.index)
                elif weighting == "sharpe":
                    w = sv/sv.sum()
                elif weighting == "sharpe_sq":
                    sq = sv**2
                    w = sq/sq.sum()
                elif weighting == "inverse_vol":
                    # Weight by inverse realized vol
                    vols = eval_data[sv.index].std()
                    inv = 1.0/vols.clip(lower=1e-6)
                    w = inv/inv.sum()
                elif weighting == "max_div":
                    # Simple max diversification: inverse correlation
                    corr_matrix = eval_data[sv.index].corr()
                    avg_corr = corr_matrix.mean()
                    inv_corr = 1.0/(avg_corr.clip(lower=0.01))
                    w = inv_corr/inv_corr.sum()
                else:
                    w = sv/sv.sum()
                
                cw = pd.Series(0.0, index=vol_t.columns)
                for k,v in w.items(): cw[k] = v
            else:
                cw = pd.Series(0.0, index=vol_t.columns)
        portfolio_ret.iloc[i] = (cw*vol_t.iloc[i]).sum()
    
    p = portfolio_ret.iloc[start_idx:]
    
    if vix_scale:
        vix = fred.get("VIXCLS")
        if vix is not None:
            va = vix.reindex(p.index).ffill()
            vp = va.rolling(252,min_periods=126).rank(pct=True)
            p = p*(1.2-0.6*vp).clip(0.5,1.2).shift(1)
    
    cum = (1+p).cumprod()
    dd = (cum-cum.cummax())/cum.cummax()
    dd_scale = np.exp(dd*dd_strength).clip(0.2,1.0)
    p = p*dd_scale.shift(1)
    
    pv = p.rolling(63,min_periods=21).std()*np.sqrt(252)
    ps = (target_vol/pv.clip(lower=0.005)).clip(0.2,5.0)
    p = p*ps.shift(1)
    
    return p.dropna()

def m(r, name=""):
    r=r.dropna()
    if len(r)<60: return None
    ar=r.mean()*252; av=r.std()*np.sqrt(252); sr=ar/av if av>0 else 0
    cum=(1+r).cumprod(); mdd=((cum-cum.cummax())/cum.cummax()).min()
    wr=(r>0).mean()
    ds=r[r<0].std()*np.sqrt(252) if (r<0).any() else av
    sortino=ar/ds if ds>0 else 0
    sp = int(len(r)*0.6)
    test_sr = r.iloc[sp:].mean()/r.iloc[sp:].std()*np.sqrt(252) if r.iloc[sp:].std()>0 else 0
    # Walk-forward
    nt=len(r); fs=nt//6; wf=[]
    for fold in range(5):
        s=(fold+1)*fs; e=min(s+fs,nt)
        fr=r.iloc[s:e]
        if len(fr)>60 and fr.std()>0:
            wf.append(fr.mean()/fr.std()*np.sqrt(252))
    wf_mean = np.mean(wf) if wf else 0
    return {"name":name,"sr":round(sr,3),"ret":round(ar*100,2),"vol":round(av*100,2),
            "mdd":round(mdd*100,2),"wr":round(wr*100,1),"sortino":round(sortino,3),
            "test_sr":round(test_sr,3),"wf_mean":round(wf_mean,3),
            "ac1":round(r.autocorr(1),4),"n":len(r)}

# ================================================================
print("Loading data...")
prices, fred, ret, streams = load()
vol_t = vol_target_streams(streams)
print(f"Universe: {len(streams)} streams, {len(prices.columns)} ETFs")

# BASELINE
print("\n" + "="*70)
print("BASELINE: Current V11 (Top 5, Sharpe-weighted, monthly)")
print("="*70)
baseline = run_adaptive(vol_t, fred, top_n=5)
bm = m(baseline, "Baseline")
print(f"  Sharpe={bm['sr']}  Test={bm['test_sr']}  WF={bm['wf_mean']}  Ret={bm['ret']}%  MDD={bm['mdd']}%  AC={bm['ac1']}")
RESULTS.append(bm)

# ================================================================
# EXPERIMENT A: TOP N SELECTION
# ================================================================
print("\n" + "="*70)
print("EXPERIMENT A: Optimal number of positions")
print("="*70)
for n in [3,4,5,6,7,8,10,12,15]:
    p = run_adaptive(vol_t, fred, top_n=n)
    if p is not None:
        mx = m(p, f"Top_{n}")
        print(f"  Top {n:2d}: SR={mx['sr']:.3f}  Test={mx['test_sr']:.3f}  WF={mx['wf_mean']:.3f}  Ret={mx['ret']:+.1f}%  MDD={mx['mdd']:.1f}%")
        RESULTS.append(mx)

# ================================================================
# EXPERIMENT B: WEIGHTING SCHEMES
# ================================================================
print("\n" + "="*70)
print("EXPERIMENT B: Weighting schemes (Top 5)")
print("="*70)
for wt in ["equal","sharpe","sharpe_sq","inverse_vol","max_div"]:
    p = run_adaptive(vol_t, fred, top_n=5, weighting=wt)
    if p is not None:
        mx = m(p, f"Wt_{wt}")
        print(f"  {wt:12s}: SR={mx['sr']:.3f}  Test={mx['test_sr']:.3f}  WF={mx['wf_mean']:.3f}  Ret={mx['ret']:+.1f}%")
        RESULTS.append(mx)

# ================================================================
# EXPERIMENT C: EVALUATION WINDOW
# ================================================================
print("\n" + "="*70)
print("EXPERIMENT C: Evaluation window length")
print("="*70)
for ew in [63, 126, 189, 252, 378, 504]:
    p = run_adaptive(vol_t, fred, top_n=5, eval_window=ew)
    if p is not None:
        mx = m(p, f"EvalWin_{ew}")
        print(f"  {ew:3d}d: SR={mx['sr']:.3f}  Test={mx['test_sr']:.3f}  WF={mx['wf_mean']:.3f}  Ret={mx['ret']:+.1f}%")
        RESULTS.append(mx)

# ================================================================
# EXPERIMENT D: MINIMUM SHARPE THRESHOLD
# ================================================================
print("\n" + "="*70)
print("EXPERIMENT D: Minimum trailing Sharpe to include")
print("="*70)
for ms in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
    p = run_adaptive(vol_t, fred, top_n=5, min_sharpe=ms)
    if p is not None:
        mx = m(p, f"MinSR_{ms}")
        print(f"  ≥{ms:.2f}: SR={mx['sr']:.3f}  Test={mx['test_sr']:.3f}  WF={mx['wf_mean']:.3f}  Ret={mx['ret']:+.1f}%")
        RESULTS.append(mx)

# ================================================================
# EXPERIMENT E: VOL TARGET LEVELS
# ================================================================
print("\n" + "="*70)
print("EXPERIMENT E: Portfolio vol target")
print("="*70)
for tv in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    p = run_adaptive(vol_t, fred, top_n=5, target_vol=tv)
    if p is not None:
        mx = m(p, f"VolTgt_{int(tv*100)}")
        print(f"  {tv*100:.0f}%: SR={mx['sr']:.3f}  Test={mx['test_sr']:.3f}  Ret={mx['ret']:+.1f}%  Vol={mx['vol']:.1f}%  MDD={mx['mdd']:.1f}%")
        RESULTS.append(mx)

# ================================================================
# EXPERIMENT F: DRAWDOWN CONTROL STRENGTH
# ================================================================
print("\n" + "="*70)
print("EXPERIMENT F: Drawdown control strength")
print("="*70)
for ds in [0, 2, 3, 5, 8, 10, 15]:
    p = run_adaptive(vol_t, fred, top_n=5, dd_strength=ds)
    if p is not None:
        mx = m(p, f"DD_{ds}")
        print(f"  DD={ds:2d}: SR={mx['sr']:.3f}  Test={mx['test_sr']:.3f}  Ret={mx['ret']:+.1f}%  MDD={mx['mdd']:.1f}%")
        RESULTS.append(mx)

# ================================================================
# EXPERIMENT G: VIX SCALING ON/OFF
# ================================================================
print("\n" + "="*70)
print("EXPERIMENT G: VIX scaling")
print("="*70)
for vs in [True, False]:
    p = run_adaptive(vol_t, fred, top_n=5, vix_scale=vs)
    if p is not None:
        mx = m(p, f"VIX_{'on' if vs else 'off'}")
        vlbl = "on " if vs else "off"
        print(f"  VIX={vlbl}: SR={mx['sr']:.3f}  Test={mx['test_sr']:.3f}  Ret={mx['ret']:+.1f}%  MDD={mx['mdd']:.1f}%")
        RESULTS.append(mx)

# ================================================================
# EXPERIMENT H: STREAM VOL TARGET
# ================================================================
print("\n" + "="*70)
print("EXPERIMENT H: Per-stream vol target")
print("="*70)
for sv in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
    vt2 = vol_target_streams(streams, target=sv)
    p = run_adaptive(vt2, fred, top_n=5)
    if p is not None:
        mx = m(p, f"StreamVol_{int(sv*100)}")
        print(f"  {sv*100:.0f}%: SR={mx['sr']:.3f}  Test={mx['test_sr']:.3f}  Ret={mx['ret']:+.1f}%  Vol={mx['vol']:.1f}%")
        RESULTS.append(mx)

# ================================================================
# EXPERIMENT I: REBALANCE FREQUENCY (already tested, quick confirm)
# ================================================================
print("\n" + "="*70)
print("EXPERIMENT I: Rebalance frequency")
print("="*70)
for rf in [5, 10, 15, 21, 42, 63]:
    p = run_adaptive(vol_t, fred, top_n=5, rebal_freq=rf)
    if p is not None:
        mx = m(p, f"Rebal_{rf}d")
        print(f"  {rf:2d}d: SR={mx['sr']:.3f}  Test={mx['test_sr']:.3f}  WF={mx['wf_mean']:.3f}")
        RESULTS.append(mx)

# ================================================================
# EXPERIMENT J: COMBINED BEST PARAMETERS
# ================================================================
print("\n" + "="*70)
print("EXPERIMENT J: Combined parameter sweeps")
print("="*70)

# Find best individual settings
combos = [
    # (name, top_n, weighting, eval_window, min_sharpe, dd_strength, vix, stream_vol, target_vol, rebal)
    ("Best_v1", 5, "sharpe", 252, 0.0, 5, True, 0.03, 0.10, 21),     # current baseline
    ("Aggressive", 3, "sharpe_sq", 189, 0.25, 3, True, 0.03, 0.15, 21),
    ("Conservative", 7, "equal", 378, 0.0, 8, True, 0.02, 0.08, 21),
    ("HighConviction", 3, "sharpe", 252, 0.5, 5, True, 0.03, 0.12, 21),
    ("FastRotate", 5, "sharpe", 126, 0.0, 5, True, 0.03, 0.10, 10),
    ("MaxSharpe", 4, "sharpe_sq", 189, 0.25, 5, True, 0.03, 0.12, 15),
    ("LowDD", 5, "sharpe", 252, 0.0, 10, True, 0.02, 0.08, 21),
    ("Balanced", 5, "sharpe", 252, 0.25, 5, True, 0.03, 0.12, 21),
    ("Ultra", 3, "sharpe_sq", 126, 0.5, 3, True, 0.05, 0.15, 10),
    ("HighVol", 4, "sharpe", 189, 0.25, 3, False, 0.05, 0.20, 15),
]

for name, tn, wt, ew, ms, dd, vix, sv, tv, rf in combos:
    vt2 = vol_target_streams(streams, target=sv)
    p = run_adaptive(vt2, fred, top_n=tn, weighting=wt, eval_window=ew,
                      min_sharpe=ms, dd_strength=dd, vix_scale=vix,
                      target_vol=tv, rebal_freq=rf)
    if p is not None:
        mx = m(p, name)
        print(f"  {name:16s}: SR={mx['sr']:.3f}  Test={mx['test_sr']:.3f}  WF={mx['wf_mean']:.3f}  "
              f"Ret={mx['ret']:+.1f}%  Vol={mx['vol']:.1f}%  MDD={mx['mdd']:.1f}%  AC={mx['ac1']:.4f}")
        RESULTS.append(mx)

# ================================================================
# EXPERIMENT K: NEW CARRY STREAMS (not in V11)
# ================================================================
print("\n" + "="*70)
print("EXPERIMENT K: Testing new potential carry streams")
print("="*70)

def hedged(ret, l, h, hw):
    if l not in ret.columns or h not in ret.columns: return None
    w1 = 1.0/(1.0+hw); w2 = hw/(1.0+hw)
    return (w1*ret[l]+w2*ret[h]).dropna()

new_streams = {}
# Ultra-short vs inverse (pure carry, minimal duration)
for l,h,hw,n in [("MINT","TBF",0.2,"mint_tbf"),("JPST","TBF",0.2,"jpst_tbf"),
                   ("BIL","TBF",0.3,"bil_tbf"),("SGOV","TBF",0.3,"sgov_tbf"),
                   ("USFR","TBF",0.2,"usfr_tbf"),
                   # Fallen angels
                   ("ANGL","SH",0.3,"angl_sh"),("ANGL","TBF",0.4,"angl_tbf"),
                   ("SHYG","SH",0.3,"shyg_sh"),("SHYG","TBF",0.3,"shyg_tbf"),
                   # More sector hedged
                   ("XLK","SH",0.3,"xlk_sh"),("XLE","SH",0.3,"xle_sh"),
                   ("XLI","SH",0.3,"xli_sh"),("XLY","SH",0.3,"xly_sh"),
                   # Country hedged
                   ("EWJ","SH",0.3,"ewj_sh"),("EFA","SH",0.3,"efa_sh"),
                   ("EEM","SH",0.3,"eem_sh"),
                   # REIT variations
                   ("VNQ","SH",0.3,"vnq_sh"),("IYR","SH",0.3,"iyr_sh"),
                   # Active bond funds
                   ("BOND","TBF",0.3,"bond_tbf"),("TOTL","TBF",0.3,"totl_tbf"),
                   # Convertible + inverse
                   ("CWB","SH",0.3,"cwb_sh_new"),
                   # Uranium + inverse
                   ("URA","SH",0.3,"ura_sh"),
                   # Semiconductor + inverse
                   ("SMH","SH",0.3,"smh_sh"),
                   # Homebuilder + inverse
                   ("ITB","SH",0.3,"itb_sh"),("XHB","SH",0.3,"xhb_sh"),
                   ]:
    r = hedged(ret, l, h, hw)
    if r is not None and len(r) >= 252:
        sr = r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0
        new_streams[f"new_{n}"] = r
        if sr > 0.3:
            print(f"  ★ {n:20s}: Sharpe={sr:+.3f}  Ret={r.mean()*252*100:+.1f}%  (PROMISING)")
        elif sr > 0:
            print(f"    {n:20s}: Sharpe={sr:+.3f}  Ret={r.mean()*252*100:+.1f}%")

# Test adding promising new streams to the portfolio
promising = {k:v for k,v in new_streams.items() 
             if v.mean()/v.std()*np.sqrt(252) > 0.3 and len(v)>=252}
print(f"\n  {len(promising)} promising new streams found")

if promising:
    # Combine with existing streams
    all_streams = {**streams, **promising}
    vt_expanded = vol_target_streams(all_streams)
    p_expanded = run_adaptive(vt_expanded, fred, top_n=5)
    if p_expanded is not None:
        mx = m(p_expanded, "Expanded_Universe")
        print(f"  Expanded (top 5): SR={mx['sr']:.3f}  Test={mx['test_sr']:.3f}  WF={mx['wf_mean']:.3f}")
        RESULTS.append(mx)
    
    # Also try top 5 from expanded with best params
    for tn in [4,5,6]:
        p2 = run_adaptive(vt_expanded, fred, top_n=tn, weighting="sharpe_sq",
                           eval_window=189, min_sharpe=0.25, dd_strength=5,
                           target_vol=0.12, rebal_freq=15)
        if p2 is not None:
            mx = m(p2, f"Expanded_Tuned_{tn}")
            print(f"  Expanded tuned top {tn}: SR={mx['sr']:.3f}  Test={mx['test_sr']:.3f}  WF={mx['wf_mean']:.3f}")
            RESULTS.append(mx)

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "="*70)
print("SUMMARY: ALL EXPERIMENTS RANKED BY WALK-FORWARD SHARPE")
print("="*70)
print(f"{'Name':24s} {'Full SR':>8} {'Test SR':>8} {'WF Mean':>8} {'Return':>8} {'MDD':>8} {'AC1':>8}")
print("-"*76)
for r in sorted(RESULTS, key=lambda x: -x['wf_mean']):
    flag = " ★" if r['wf_mean'] > bm['wf_mean'] + 0.05 else ""
    print(f"  {r['name']:22s} {r['sr']:>7.3f} {r['test_sr']:>7.3f} {r['wf_mean']:>7.3f} {r['ret']:>+7.1f}% {r['mdd']:>7.1f}% {r['ac1']:>7.4f}{flag}")

# Save results
with open(DATA_DIR/"results"/"experiments.json", "w") as f:
    json.dump({"experiments": RESULTS, "baseline": bm}, f, indent=2)

print(f"\nSaved {len(RESULTS)} experiment results to experiments.json")
