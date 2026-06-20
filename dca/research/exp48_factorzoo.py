"""Loop3/exp48 — systematic FACTOR ZOO with multiple-testing discipline.
Generate many distinct cross-sectional signals, test each for OOS sign-stability
(same-sign + |t|>1.5 in BOTH 2010-17 and 2018-25 halves), and compare the
survivor count to the FALSE-DISCOVERY expectation (~chance). The honest way to
run '100 ideas': if survivors <= what data-mining produces by chance, there's no
real signal. Includes a RANDOM-feature null to calibrate."""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
me = pd.read_pickle("/tmp/wave/_ins_px.pkl"); names=[x for x in me.columns if x not in ("SPY","QQQ")]
M = me[names]; q = me["QQQ"]
fwd = {h:(M.shift(-h)/M-1).sub(q.shift(-h)/q-1,axis=0) for h in (3,12)}
def z(df): return df.sub(df.mean(axis=1),axis=0)
SIG={}
for k in (1,3,6,9,12): SIG[f"mom{k}"]=M/M.shift(k)-1
SIG["rev1"]=-(M/M.shift(1)-1); SIG["rev3"]=-(M/M.shift(3)-1)
SIG["mom12skip1"]=M.shift(1)/M.shift(12)-1
for k in (6,12): SIG[f"lowvol{k}"]=-(M.pct_change().rolling(k).std())
SIG["accel"]=(M/M.shift(3)-1)-(M/M.shift(6)-1)
SIG["highprox"]=M/M.rolling(12).max()
SIG["drawdown"]=M/M.rolling(6).max()-1
SIG["madist"]=M/M.rolling(10).mean()-1
SIG["riskadjmom"]=(M/M.shift(6)-1)/(M.pct_change().rolling(6).std()+1e-9)
SIG["consist"]=(M.pct_change()>0).rolling(12).mean()
SIG["maxmo"]=-(M.pct_change().rolling(6).max())          # monthly 'lottery'
SIG["voltrend"]=M.pct_change().rolling(3).std()/(M.pct_change().rolling(12).std()+1e-9)
SIG["longrev"]=-(M.shift(12)/M.shift(36)-1)              # long-term reversal
# random-feature nulls to calibrate false discovery
rng=np.random.default_rng(0)
for j in range(6): SIG[f"RANDnull{j}"]=pd.DataFrame(rng.standard_normal(M.shape),index=M.index,columns=M.columns)

def ic(sig,h,lo,hi):
    f=fwd[h]; ics=[]
    for d in M.index:
        if not (pd.Timestamp(lo)<=d<pd.Timestamp(hi)): continue
        x=sig.loc[d].replace([np.inf,-np.inf],np.nan).dropna()
        c=[t for t in x.index if t in f.columns and np.isfinite(f.loc[d,t])]
        if len(c)<40: continue
        ics.append(spearmanr(x[c],f.loc[d,c]).correlation)
    a=np.array([i for i in ics if np.isfinite(i)])
    return (a.mean(), a.mean()/(a.std()+1e-12)*np.sqrt(len(a))) if len(a)>5 else (np.nan,np.nan)

print(f"{len(names)} names; testing {len(SIG)} signals x 2 horizons = {len(SIG)*2} hypotheses",flush=True)
print(f"{'signal':12s} {'h':>2s} | {'TRAIN t':>8s} {'TEST t':>8s}  survive?",flush=True)
surv=0; survR=0; tot=0
for nm,sig in SIG.items():
    for h in (3,12):
        _,t1=ic(sig,h,"2010-01-01","2018-01-01"); _,t2=ic(sig,h,"2018-01-01","2026-12-31")
        if np.isnan(t1) or np.isnan(t2): continue
        tot+=1
        ok = (np.sign(t1)==np.sign(t2)) and abs(t1)>1.5 and abs(t2)>1.5
        if ok: surv+=1; (survR:=survR+1) if nm.startswith("RAND") else None
        if ok or nm.startswith("RAND"):
            print(f"{nm:12s} {h:>2d} | {t1:8.1f} {t2:8.1f}   {'<<< SURVIVES' if ok else ''}",flush=True)
print(f"\nSURVIVORS (same-sign |t|>1.5 both halves): {surv}/{tot} real-signal tests",flush=True)
print(f"  of which RANDOM-null survivors: {survR} (calibrates pure data-mining/chance rate)",flush=True)
print(f"  -> if real survivors ~= random-null rate, there is NO durable price signal.",flush=True)
print("\nDONE",flush=True)
