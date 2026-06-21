import numpy as np, pandas as pd, json, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; ret=(me/me.shift(1)-1).clip(-0.9,2.0)
idx=pd.date_range("2015-01-01","2025-12-01",freq="MS")
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(M)
LIQ=(liq&(me>=3.0)).fillna(False); SHORT=(liq&(me>=10.0)).fillna(False)
lmrank=FEAT["log_mcap"].rank(axis=1,pct=True)
q=qret.reindex(M); qm=q.rolling(12,min_periods=8).mean(); qv=q.rolling(12,min_periods=8).var()
rq=ret.mul(q,axis=0); cov=rq.rolling(12,min_periods=8).mean().sub(ret.rolling(12,min_periods=8).mean().mul(qm,axis=0),axis=0)
BETA=cov.div(qv,axis=0).clip(-3,3).fillna(1.0)
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(index=M,columns=cols)
borrow_rate=pd.DataFrame(0.06,index=M,columns=cols).where(lmrank<0.8,0.02).where(lmrank<0.95,0.01)
rebal,buffer,qq=3,2.0,0.1
rkl=PROB.where(LIQ).rank(axis=1,pct=True); rks=PROB.where(SHORT).rank(axis=1,pct=True)
Wl=pd.DataFrame(0.0,index=M,columns=cols); Ws=pd.DataFrame(0.0,index=M,columns=cols)
hl=set(); hs=set(); cl=pd.Series(0.0,index=cols); cs=pd.Series(0.0,index=cols)
for k,dt in enumerate(M):
    if k%rebal==0:
        rl=rkl.loc[dt]; rs=rks.loc[dt]
        sel_l=[t for t in hl if rl.get(t,0)>=1-qq*buffer]; nt=int((rl>=1-qq).sum())
        for t in rl[rl>=1-qq].sort_values(ascending=False).index:
            if len(sel_l)>=nt: break
            if t not in sel_l: sel_l.append(t)
        sel_s=[t for t in hs if rs.get(t,1)<=qq*buffer]; st=int((rs<=qq).sum())
        for t in rs[rs<=qq].sort_values().index:
            if len(sel_s)>=st: break
            if t not in sel_s: sel_s.append(t)
        hl=set(sel_l); hs=set(sel_s)
        cl=pd.Series(0.0,index=cols); cs=pd.Series(0.0,index=cols)
        if sel_l: cl[sel_l]=1.0/len(sel_l)
        if sel_s: cs[sel_s]=1.0/len(sel_s)
    b=BETA.loc[dt]; bl=(cl*b).sum(); bs=(cs*b).sum(); css=cs*(bl/bs) if bs>0.05 else cs
    Wl.loc[dt]=cl; Ws.loc[dt]=css
g=((Wl.shift(1)*ret).sum(axis=1)-(Ws.shift(1)*ret).sum(axis=1))
turn=(Wl.diff().abs().sum(axis=1)+Ws.diff().abs().sum(axis=1))
bcost=(Ws.shift(1)*borrow_rate/12).sum(axis=1)
net=(g-turn*(10/1e4)-bcost).reindex(idx).fillna(0.0)
def st(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
c,s,d=st(net); p(f"SUMMIT v3 (beta-neutral, tiered borrow): CAGR {c:.1%} Sharpe {s:.2f} maxDD {d:.1%} corrQQQ {net.corr(qret.reindex(idx)):.2f}")
qr=pd.Series(json.load(open('/home/user/_wsdata.json'))['qqq']).values
qser=pd.Series(qr,index=idx)
for nm,r in [('QQQ+1a',qser+net),('halfQ+half+a',0.5*qser+0.5*net),('QQQ+2a',qser+2*net)]:
    cc,ss,dd=st(r); p(f"  {nm:14} CAGR {cc:.1%} Sharpe {ss:.2f} DD {dd:.1%}")
wd=json.load(open("/home/user/_wsdata.json")); assert len(wd["dates"])==len(net)
wd["summit"]=[round(float(x),5) for x in net.values]
json.dump(wd,open("/home/user/_wsdata.json","w"))
p(f"updated _wsdata.json summit (v3) t={time.time()-t0:.0f}s")
