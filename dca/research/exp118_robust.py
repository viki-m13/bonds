import time; t0=time.time()
from costlib import *
hdr()
Wl,Ws=build(PROB); base=net(Wl,Ws); row("v3 baseline (base ML)",base)
print("-"*92)
# model-average ensemble (rank-average of non-broken cached models) — robustness play
models={n:pd.read_pickle(f"/tmp/wave/_mlprob{('_'+n) if n else ''}.pkl").reindex(index=M,columns=cols) for n in ["","50feat","ens","selens"]}
# individual through full harness
for n,Pm in models.items():
    if n=="": continue
    Wl,Ws=build(Pm); row(f"alt model {n} (full harness)",net(Wl,Ws))
print("-"*92)
ranks=[Pm.where(LIQ).rank(axis=1,pct=True) for Pm in models.values()]
avg=sum(ranks)/len(ranks)
Wl,Ws=build(avg); row("model-average (base+50f+ens+sel)",net(Wl,Ws))
avg2=(models[""].where(LIQ).rank(axis=1,pct=True)+models["ens"].where(LIQ).rank(axis=1,pct=True))/2
Wl,Ws=build(avg2); row("model-average (base+ens)",net(Wl,Ws))
print("-"*92)
# parameter robustness grid (rebal x buffer) — is 2.4 Sharpe fragile?
print("param robustness — net Sharpe @100M across (rebal,buffer):",flush=True)
print(f"   {'buf=1.5':>9}{'buf=2.0':>9}{'buf=2.5':>9}{'buf=3.0':>9}",flush=True)
for rb in [2,3,4]:
    cells=[]
    for bf in [1.5,2.0,2.5,3.0]:
        Wl,Ws=build(PROB,rebal=rb,buffer=bf); _,s,_=ann(net(Wl,Ws),FULL); cells.append(f"{s:>9.2f}")
    print(f"rb={rb} "+"".join(cells),flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s")
