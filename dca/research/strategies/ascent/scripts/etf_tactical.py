"""Tactical ETF DCA — decide WHICH ETF(s) to DCA into monthly (incl gold/oil/tech/
midcap/leveraged) and WHEN to sell, to beat DCA-into-QQQ.

Signal = dual momentum + trend filter:
  eligible = above 200d MA AND positive blended momentum (3/6/12m).
  leveraged ETFs additionally require underlying in a confirmed strong uptrend.
  each month: contribution + any freed cash -> top-K eligible by momentum (equal split).
  hold; SELL when a holding drops below its 200d MA (trend break); redeploy next month.
  risk-off (fewer than K eligible): park remainder in BIL (cash).
Benchmark: DCA into QQQ, identical monthly flows.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))

P = pd.read_pickle(f"{HERE}/_etf_panel.pkl")
close, kind, LEV = P["close"], P["kind"], P["lev"]
close = close.sort_index()
# monthly grid = last trading day of month
mclose = close.resample("ME").last()
ret_d = close.pct_change()
ma200 = close.rolling(200, min_periods=200).mean()
# monthly-sampled signals (use daily-through-month-end, causal)
ma200_m = ma200.reindex(mclose.index, method="ffill")
above_ma = (mclose > ma200_m)
def mom(n):  # n-month total return on monthly close
    return mclose / mclose.shift(n) - 1
mom3, mom6, mom12 = mom(3), mom(6), mom(12)
blend = (mom3.rank(axis=1, pct=True) + mom6.rank(axis=1, pct=True) + mom12.rank(axis=1, pct=True)) / 3
absmom = (mom6 > 0) & (mom12 > 0)

UNDER = {name: u for name, (u, L, e, i) in LEV.items()}
def lev_ok(dt):
    """leveraged ETF eligible only if its underlying is in a strong confirmed uptrend."""
    ok = {}
    for name, u in UNDER.items():
        if u in mclose.columns:
            strong = bool(above_ma.loc[dt, u]) and (mom3.loc[dt, u] > 0.0) and (mom6.loc[dt, u] > 0.0)
            ok[name] = strong
    return ok

def eligible(dt, use_lev=True, lev_only_strong=True):
    el = {}
    row_above = above_ma.loc[dt]; row_abs = absmom.loc[dt]
    lok = lev_ok(dt)
    for t in mclose.columns:
        if t == "BIL":
            continue
        if not np.isfinite(mclose.loc[dt, t]):
            continue
        e = bool(row_above.get(t, False)) and bool(row_abs.get(t, False))
        if kind[t] == "lev":
            if not use_lev:
                e = False
            elif lev_only_strong:
                e = e and lok.get(t, False)
        el[t] = e
    return el

def run(start, end, K=3, contrib=1000.0, cost=0.0010, use_lev=True, cash="BIL",
        exclude=None):
    exclude = set(exclude or [])
    dates = mclose.index[(mclose.index >= start) & (mclose.index <= end)]
    pos = {}  # tk -> shares
    rows = []; contributed = 0.0
    holdlog = []
    for dt in dates:
        px = mclose.loc[dt]
        # sell holdings that broke trend (below 200d MA) or leverage no longer strong
        lok = lev_ok(dt)
        freed = 0.0
        for tk in list(pos.keys()):
            below = not bool(above_ma.loc[dt, tk]) if tk in above_ma.columns else True
            lev_bad = (kind.get(tk) == "lev") and (not lok.get(tk, False))
            if below or lev_bad or not np.isfinite(px.get(tk, np.nan)):
                freed += pos[tk] * px.get(tk, 0) * (1 - cost)
                pos.pop(tk)
        cashamt = contrib + freed
        contributed += contrib
        # choose top-K eligible by momentum blend
        el = eligible(dt, use_lev=use_lev)
        cand = [t for t in mclose.columns if el.get(t) and t not in exclude]
        cand = sorted(cand, key=lambda t: -(blend.loc[dt, t] if np.isfinite(blend.loc[dt, t]) else -1))
        # keep existing eligible holdings; fill up to K with best new
        held = [t for t in pos if el.get(t) and t not in exclude]
        target = held[:]
        for t in cand:
            if len(target) >= K: break
            if t not in target: target.append(t)
        target = target[:K]
        holdlog.append((dt, tuple(target) if target else ("CASH",)))
        if target:
            per = cashamt / len(target)
            for t in target:
                sh = per / px[t]
                pos[t] = pos.get(t, 0) + sh
            cashamt = 0.0
        else:
            # risk-off: buy cash asset
            if cash in px and np.isfinite(px[cash]):
                pos[cash] = pos.get(cash, 0) + cashamt / px[cash]; cashamt = 0.0
        V = sum(sh * px.get(t, 0) for t, sh in pos.items()) + cashamt
        rows.append((dt, V, contributed))
    eq = pd.DataFrame(rows, columns=["date","V","contributed"]).set_index("date")
    return eq, pd.DataFrame(holdlog, columns=["date","held"]).set_index("date")

def dca_bench(tk, start, end, contrib=1000.0, cost=0.0005):
    dates = mclose.index[(mclose.index >= start) & (mclose.index <= end)]
    sh = 0.0; contributed = 0.0; rows = []
    for dt in dates:
        p = mclose.loc[dt, tk]
        if not np.isfinite(p): continue
        sh += contrib * (1 - cost) / p; contributed += contrib
        rows.append((dt, sh * p, contributed))
    return pd.DataFrame(rows, columns=["date","V","contributed"]).set_index("date")

def stats(eq, freq=12):
    V = eq["V"]; C = eq["contributed"].diff().fillna(eq["contributed"])
    r = ((V - C) / V.shift(1) - 1).dropna()
    cagr = (1+r).prod()**(freq/len(r))-1
    sh = r.mean()/r.std()*np.sqrt(freq) if r.std()>0 else np.nan
    c=(1+r).cumprod(); dd=(c/c.cummax()-1).min()
    return dict(moic=V.iloc[-1]/eq["contributed"].iloc[-1], cagr=cagr, sharpe=sh, maxdd=dd, final=V.iloc[-1])

if __name__ == "__main__":
    ERAS = [("2006-01","2009-12"),("2010-01","2014-12"),("2015-01","2019-12"),
            ("2020-01","2026-06"),("2010-01","2026-06"),("2006-01","2026-06")]
    print(f"{'config':34} " + " ".join(f"{a[:7]:>8}" for a,b in ERAS))
    def line(nm, K, use_lev, exclude=None):
        out=[]
        for st,en in ERAS:
            s=pd.Timestamp(st+"-01"); e=pd.Timestamp(en+"-01")
            eq,_=run(s,e,K=K,use_lev=use_lev,exclude=exclude)
            b=dca_bench("QQQ",s,e)
            out.append(eq["V"].iloc[-1]/b["V"].iloc[-1])
        print(f"{nm:34} " + " ".join(f"{v:>8.2f}" for v in out))
    # QQQ-DCA reference row = all 1.0 by construction
    line("dual-mom K1 no-lev", 1, False)
    line("dual-mom K3 no-lev", 3, False)
    line("dual-mom K3 +LEVERAGE", 3, True)
    line("dual-mom K1 +LEVERAGE", 1, True)
    line("dual-mom K5 +LEVERAGE", 5, True)
    # detail on the headline
    s=pd.Timestamp("2010-01-01"); e=pd.Timestamp("2026-06-01")
    eq,hl=run(s,e,K=3,use_lev=True); b=dca_bench("QQQ",s,e)
    S=stats(eq); Sb=stats(b)
    print(f"\nK3+lev 2010-2026: MOIC {S['moic']:.2f}x CAGR {S['cagr']:.1%} Sh {S['sharpe']:.2f} DD {S['maxdd']:.1%}"
          f"  vs QQQ-DCA MOIC {Sb['moic']:.2f}x CAGR {Sb['cagr']:.1%} Sh {Sb['sharpe']:.2f} DD {Sb['maxdd']:.1%}")
    print("recent holdings:", hl.tail(6).held.tolist())
