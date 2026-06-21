import time; t0=time.time()
from costlib import *
hdr()
# baseline v3
Wl,Ws=build(PROB); base=net(Wl,Ws); row("v3 baseline (q-rebal,buf2,decile)",base)
print("-"*92)
# rebalance frequency (now that costs are modeled, find cost-optimal)
for rb in [1,2,3,6]:
    Wl,Ws=build(PROB,rebal=rb); row(f"rebal={rb}mo",net(Wl,Ws))
print("-"*92)
# buffer width
for bf in [1.0,1.5,2.0,3.0]:
    Wl,Ws=build(PROB,buffer=bf); row(f"buffer={bf}x",net(Wl,Ws))
print("-"*92)
# concentration: decile/quintile/ventile/half-decile
for qq in [0.05,0.10,0.20]:
    Wl,Ws=build(PROB,q=qq); row(f"q={qq} ({'ventile' if qq==.05 else 'decile' if qq==.1 else 'quintile'})",net(Wl,Ws))
print("-"*92)
# tranching: overlapping monthly sub-books each held `rebal` months
for nt in [1,3]:
    Wl,Ws=build(PROB,rebal=3,ntranche=nt); row(f"quarterly, {nt}-tranche",net(Wl,Ws))
for nt in [1,6]:
    Wl,Ws=build(PROB,rebal=6,ntranche=nt); row(f"semiannual, {nt}-tranche",net(Wl,Ws))
print(f"\nDONE t={time.time()-t0:.0f}s")
