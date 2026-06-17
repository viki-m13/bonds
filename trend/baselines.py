#!/usr/bin/env python3
"""Vol-targeted cross-asset trend baselines (the honest floor for L2GMOM).

Replicates the *reference* strategies from the network-momentum literature
(Moskowitz-Ooi-Pedersen TSMOM, Baz et al. vol-normalised MACD, linear-trend)
on this repo's ETF cross-asset universe, with strict next-day execution,
ex-ante vol scaling, a 15% portfolio vol target, transaction costs, and
sub-period Sharpe so trend-decay is visible.

Question being answered: do simple, free trend signals clear ~1.0 Sharpe net of
costs on the assets we actually have? If not, a learned-graph GNN won't rescue it.
"""
import numpy as np, pandas as pd, glob, os, json

DATA = "/home/user/bonds/data/etfs"
OUT  = "/home/user/bonds/trend"
ANN  = 252
VOL_TGT = 0.15          # annualised, per-asset and portfolio
COST_BPS = [0.0, 1.0, 3.0]   # bps on |Δ position weight| (notional turnover)

# Curated diversified cross-asset panel (liquid, distinct, deep history,
# NO leveraged/inverse, NO target-maturity ladders, NO crypto).
UNIVERSE = {
 "Equity-US":   ["SPY","IWM","MDY","DIA"],
 "Equity-Intl": ["EFA","EEM","EWJ","EWZ","EWG","EWU","EWA","EWC","FXI"],
 "Equity-Sector":["XLE","XLF","XLK","XLI","XLP","XLU","XLV","XLY","XLB"],
 "Rates":       ["SHY","IEI","IEF","TLT","TLH"],
 "Credit":      ["LQD","HYG","AGG","TIP"],
 "Commodity":   ["GLD","SLV","DBC","USO","DBA"],
 "FX":          ["UUP","FXE","FXY","FXB","FXA","FXF"],
 "Real-assets": ["VNQ","IYR"],
}
TICKERS = [t for v in UNIVERSE.values() for t in v]
CLASS = {t:k for k,v in UNIVERSE.items() for t in v}

def load_panel():
    cols={}
    for t in TICKERS:
        f=f"{DATA}/{t}.csv"
        if not os.path.exists(f): print("MISSING",t); continue
        df=pd.read_csv(f, usecols=["Date","Close"]).dropna()
        df["Date"]=pd.to_datetime(df["Date"]); df=df.set_index("Date")["Close"].astype(float)
        cols[t]=df
    px=pd.DataFrame(cols).sort_index()
    px=px[~px.index.duplicated()]
    return px

def ewstd(x, span):  # exponentially weighted std of daily returns
    return x.ewm(span=span, min_periods=span//2).std()

def macd_baz(price):
    """Baz et al. (2015) vol-normalised MACD, averaged over 3 timescale pairs."""
    def hl(n): return np.log(0.5)/np.log(1-1/n)
    sig=pd.Series(0.0, index=price.index); cnt=0
    pstd=price.rolling(63, min_periods=30).std()
    for S,L in [(8,24),(16,48),(32,96)]:
        m = price.ewm(halflife=hl(S)).mean() - price.ewm(halflife=hl(L)).mean()
        q = m/pstd
        y = q/q.rolling(252, min_periods=120).std()
        sig = sig.add(y*np.exp(-y**2/4)/0.89, fill_value=0.0); cnt+=1
    return sig/cnt

def linreg_slope_sign(logp, win=252):
    """Sign of OLS slope of log price on time over a rolling window."""
    x=np.arange(win); xc=x-x.mean(); denom=(xc**2).sum()
    def f(a):
        yc=a-a.mean()
        return np.sign((xc*yc).sum()/denom)
    return logp.rolling(win, min_periods=win).apply(f, raw=True)

def build_signals(px):
    rets = px.pct_change()
    logp = np.log(px)
    sigma = ewstd(rets, 60)*np.sqrt(ANN)        # ex-ante annualised vol per asset
    sig={}
    sig["TSMOM"]  = np.sign(px.pct_change(252))                       # 12m momentum sign
    sig["MACD"]   = pd.DataFrame({t:macd_baz(px[t]) for t in px}).clip(-2,2)
    sig["LinReg"] = pd.DataFrame({t:linreg_slope_sign(logp[t]) for t in px})
    # COMBO: equal-weight ensemble of the three (all ~[-1,1]) — cheap stand-in
    # for the paper's learned signal combination.
    sig["COMBO"]  = (sig["TSMOM"].fillna(0)+sig["LinReg"].fillna(0)
                     +sig["MACD"].clip(-1,1).fillna(0))/3
    return rets, sigma, sig

def backtest(rets, sigma, signal, cost_bps):
    # position weight w_{t} sized to per-asset VOL_TGT; trade on signal known at t,
    # earn r_{t+1}. Only where vol & signal valid.
    w = signal * (VOL_TGT/sigma)
    w = w.replace([np.inf,-np.inf], np.nan)
    valid = w.notna() & rets.shift(-1).notna()
    w = w.where(valid)
    n_active = valid.sum(axis=1).replace(0,np.nan)
    # gross strategy return (equal capital across active assets), next-day
    gross = (w * rets.shift(-1)).sum(axis=1)/n_active
    # turnover cost: bps on |Δw| averaged across capital
    dw = w.fillna(0).diff().abs().sum(axis=1)/n_active
    cost = dw*(cost_bps/1e4)
    net = (gross - cost).dropna()
    return net, dw.reindex(net.index)

def vol_target(net, span=60, tgt=VOL_TGT):
    rv = net.ewm(span=span, min_periods=30).std()*np.sqrt(ANN)
    scale = (tgt/rv).clip(upper=3.0).shift(1)   # ex-ante, capped leverage 3x
    return (net*scale).dropna()

def stats(r):
    r=r.dropna()
    if len(r)<60: return {}
    ann_ret=r.mean()*ANN; ann_vol=r.std()*np.sqrt(ANN)
    sharpe=ann_ret/ann_vol if ann_vol>0 else 0
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()
    return dict(ann_ret=round(ann_ret,4), ann_vol=round(ann_vol,4),
                sharpe=round(sharpe,3), max_dd=round(float(dd),4), n=len(r))

def subperiods(r):
    cuts=[("2008-2014","2008-01-01","2014-12-31"),
          ("2015-2019","2015-01-01","2019-12-31"),
          ("2020-2026","2020-01-01","2026-12-31")]
    return {nm:stats(r.loc[a:b]) for nm,a,b in cuts}

def main():
    px=load_panel()
    # restrict to common deep history; require >=20 assets present
    start=px.dropna(thresh=20).index.min()
    px=px.loc[start:]
    print(f"Panel: {px.shape[1]} ETFs, {px.index.min().date()} .. {px.index.max().date()}, {len(px)} days")
    rets,sigma,sig=build_signals(px)
    spy=px["SPY"].pct_change()

    report={"universe":{k:v for k,v in UNIVERSE.items()},
            "panel":{"n_etfs":int(px.shape[1]),"start":str(px.index.min().date()),
                     "end":str(px.index.max().date()),"vol_target":VOL_TGT},
            "results":{}}
    print("\n=== VOL-TARGETED PORTFOLIO (15%), net of costs ===")
    print(f"{'signal':8s} {'cost':>5s} {'Sharpe':>7s} {'ret':>7s} {'vol':>6s} {'maxDD':>7s} {'08-14':>6s} {'15-19':>6s} {'20-26':>6s} {'corrSPY':>7s}")
    for name,S in sig.items():
        report["results"][name]={}
        for c in COST_BPS:
            net,turn=backtest(rets,sigma,S,c)
            vt=vol_target(net)
            st=stats(vt); sp=subperiods(vt)
            corr=round(float(vt.reindex(spy.index).corr(spy)),3)
            report["results"][name][f"{c}bps"]=dict(full=st, sub=sp,
                  avg_turnover=round(float(turn.mean()),3), corr_spy=corr)
            if c in (0.0,3.0):
                print(f"{name:8s} {c:4.0f}b {st['sharpe']:7.2f} {st['ann_ret']*100:6.1f}% "
                      f"{st['ann_vol']*100:5.1f}% {st['max_dd']*100:6.1f}% "
                      f"{sp['2008-2014'].get('sharpe',0):6.2f} {sp['2015-2019'].get('sharpe',0):6.2f} "
                      f"{sp['2020-2026'].get('sharpe',0):6.2f} {corr:7.2f}")
    # naive long-only equity benchmark (SPY) at 15% vol for reference
    spyvt=vol_target(spy.dropna()); report["results"]["LongOnly_SPY"]={"15%vol":stats(spyvt)}
    print(f"\nLong-only SPY @15% vol: Sharpe {stats(spyvt)['sharpe']}, "
          f"ret {stats(spyvt)['ann_ret']*100:.1f}%, maxDD {stats(spyvt)['max_dd']*100:.1f}%")
    json.dump(report, open(f"{OUT}/baselines_results.json","w"), indent=1)
    print(f"\nWrote {OUT}/baselines_results.json")

if __name__=="__main__":
    main()
