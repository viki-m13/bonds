#!/usr/bin/env python3
"""
V10: Cross-Asset Carry + Leveraged ETF Volatility Decay
========================================================

Adds two new high-Sharpe, low-correlation engines:

ENGINE A: VOLATILITY DECAY CAPTURE
  Short leveraged ETFs + delta-hedge with underlying.
  Captures the daily rebalancing drag (volatility tax).
  E.g., Short TQQQ + 3x Long QQQ = pure vol decay.

ENGINE B: BULL/BEAR PAIR ARBITRAGE
  Short BOTH the bull and bear leveraged ETFs simultaneously.
  Both decay due to daily rebalancing, capturing pure volatility.
  E.g., Short FAS + Short FAZ = pure financial vol capture.

These require borrowing to short, so higher transaction costs (15bps).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path("/home/user/bonds/data")
ETF_DIR = DATA_DIR / "etfs"
FRED_PATH = DATA_DIR / "fred" / "_combined_fred.csv"
TC_BPS = 5
TC_LEV_BPS = 15  # Higher costs for leveraged ETF shorting
TARGET_VOL = 0.10
EVAL_WINDOW = 252
REBAL_FREQ = 21


def load_all_data():
    prices = {}
    for f in sorted(ETF_DIR.glob("*.csv")):
        if f.name.startswith("_"): continue
        try:
            df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
            df = df[~df.index.duplicated(keep="first")].sort_index()
            if "Close" in df.columns: prices[f.stem] = df["Close"]
        except: continue
    prices = pd.DataFrame(prices).sort_index()
    fred = pd.read_csv(FRED_PATH, parse_dates=["Date"]).set_index("Date")
    fred = fred[~fred.index.duplicated(keep="first")].sort_index()
    for c in fred.columns: fred[c] = pd.to_numeric(fred[c], errors="coerce")
    fred = fred.ffill()
    return prices, fred


def hedged_pair(ret, l, h, lb=252):
    if l not in ret.columns or h not in ret.columns: return None
    cov = ret[l].rolling(lb,min_periods=126).cov(ret[h])
    var = ret[h].rolling(lb,min_periods=126).var()
    beta = (cov/var.clip(lower=1e-8)).clip(-3,3)
    return (ret[l]-beta.shift(1)*ret[h]).dropna()


def generate_all_streams(ret, fred):
    streams = {}
    MIN = 252

    # ===== V8 ENGINES (carry, dividend, etc.) =====
    # Bond carry
    for l,h in [("HYG","IEF"),("HYG","TLT"),("HYG","SHY"),("JNK","IEF"),("LQD","IEF"),
                 ("VCIT","IEI"),("VCSH","SHY"),("IGIB","IEI"),("EMB","IEF"),("EMB","TLT"),
                 ("MUB","SHY"),("MUB","IEI"),("MBB","IEF"),("TIP","IEF")]:
        r = hedged_pair(ret,l,h); 
        if r is not None and len(r)>=MIN: streams[f"bc_{l}_{h}"] = r

    # Dividend-equity carry
    for eq in ["SCHD","HDV","VIG","DVY"]:
        for bd in ["SHY","IEF","AGG"]:
            r = hedged_pair(ret,eq,bd)
            if r is not None and len(r)>=MIN: streams[f"div_{eq}_{bd}"] = r

    # Defensive equity
    for eq in ["XLP","XLU","XLV"]:
        for bd in ["SHY","IEF"]:
            r = hedged_pair(ret,eq,bd)
            if r is not None and len(r)>=MIN: streams[f"def_{eq}_{bd}"] = r

    # Commodity carry
    for cm in ["GLD","SLV","PDBC"]:
        for hd in ["SPY","IEF"]:
            r = hedged_pair(ret,cm,hd)
            if r is not None and len(r)>=MIN: streams[f"cm_{cm}_{hd}"] = r

    # Preferred/loan
    for l,h,n in [("PFF","IEF","pi"),("PFF","SHY","ps"),("BKLN","SHY","ls"),
                    ("SRLN","SHY","ss"),("CWB","SPY","cs")]:
        r = hedged_pair(ret,l,h)
        if r is not None and len(r)>=MIN: streams[f"pf_{n}"] = r

    # REIT, Intl bond, Currency, Sector, TSMOM (same as V8)
    for reit in ["VNQ","IYR","VNQI","REM"]:
        r = hedged_pair(ret,reit,"IEF")
        if r is not None and len(r)>=MIN: streams[f"rt_{reit}"] = r
    for l,h,n in [("BNDX","AGG","ba"),("EMLC","IEF","ei"),("PCY","IEF","pi")]:
        r = hedged_pair(ret,l,h)
        if r is not None and len(r)>=MIN: streams[f"ib_{n}"] = r
    for l,s,n in [("FXA","FXY","aj"),("FXB","FXY","bj"),("CEW","UUP","em")]:
        if l in ret.columns and s in ret.columns:
            r = (ret[l]-ret[s]).dropna()
            if len(r)>=MIN: streams[f"fx_{n}"] = r
    for s in ["XLK","XLP","XLU","XLV"]:
        r = hedged_pair(ret,s,"SPY")
        if r is not None and len(r)>=MIN: streams[f"sec_{s}"] = r
    for a in ["SPY","GLD","VNQ","EEM","EWJ"]:
        if a not in ret.columns: continue
        sigs = []
        for lb in [21,63,126,252]:
            pr = ret[a].rolling(lb,min_periods=int(lb*0.7)).mean()*np.sqrt(252)
            pv = ret[a].rolling(lb,min_periods=int(lb*0.7)).std()*np.sqrt(252)
            sigs.append(pr/pv.clip(lower=0.01))
        cs = pd.concat(sigs,axis=1).mean(axis=1)
        pos = cs.clip(-2,2)/2
        sr = pos.shift(1)*ret[a] - pos.diff().abs()*(TC_BPS/10000)
        if len(sr.dropna())>=MIN: streams[f"xm_{a}"] = sr.dropna()
    for cm in ["GLD","SLV","USO","DBC"]:
        if cm not in ret.columns: continue
        sigs = []
        for lb in [21,63,126,252]:
            pr = ret[cm].rolling(lb,min_periods=int(lb*0.7)).mean()*np.sqrt(252)
            pv = ret[cm].rolling(lb,min_periods=int(lb*0.7)).std()*np.sqrt(252)
            sigs.append(pr/pv.clip(lower=0.01))
        cs = pd.concat(sigs,axis=1).mean(axis=1)
        pos = cs.clip(-2,2)/2
        sr = pos.shift(1)*ret[cm] - pos.diff().abs()*(10/10000)
        if len(sr.dropna())>=MIN: streams[f"ctm_{cm}"] = sr.dropna()

    # ===== NEW ENGINE A: VOLATILITY DECAY CAPTURE =====
    # Short leveraged + delta-hedge with underlying
    decay_trades = [
        ("UPRO","SPY",3,"updecay"),   # Short UPRO + 3x long SPY
        ("TQQQ","QQQ",3,"tqdecay"),   # Short TQQQ + 3x long QQQ
        ("SSO","SPY",2,"ssdecay"),    # Short SSO + 2x long SPY
        ("QLD","QQQ",2,"qldecay"),    # Short QLD + 2x long QQQ
        ("TMF","TLT",3,"tmfdecay"),   # Short TMF + 3x long TLT
        ("UBT","TLT",2,"ubtdecay"),   # Short UBT + 2x long TLT
        ("UGL","GLD",2,"ugldecay"),   # Short UGL + 2x long GLD
    ]
    for lev, base, mult, name in decay_trades:
        if lev not in ret.columns or base not in ret.columns: continue
        decay = mult * ret[base] - ret[lev]
        tc = pd.Series(TC_LEV_BPS / 10000 / 252, index=decay.index)  # Daily borrow cost
        result = (decay - tc).dropna()
        if len(result) >= MIN: streams[f"vdecay_{name}"] = result

    # ===== NEW ENGINE B: BULL/BEAR PAIR ARBITRAGE =====
    # Short both bull and bear = capture pure volatility decay
    bb_pairs = [
        ("FAS","FAZ","fin3x"),
        ("TECL","TECS","tech3x"),
        ("SOXL","SOXS","semi3x"),
        ("TQQQ","SQQQ","qqq3x"),
        ("UPRO","SPXU","spy3x"),
        ("ERX","ERY","energy2x"),
        ("DRN","DRV","reit3x"),
        ("EDC","EDZ","em3x"),
        ("YINN","YANG","china3x"),
        ("TMF","TMV","treas3x"),
        ("NUGT","DUST","gold2x"),
    ]
    for bull, bear, name in bb_pairs:
        if bull not in ret.columns or bear not in ret.columns: continue
        # Short both equally
        short_both = -(ret[bull] + ret[bear])
        tc = pd.Series(2 * TC_LEV_BPS / 10000 / 252, index=short_both.index)
        result = (short_both - tc).dropna()
        if len(result) >= MIN: streams[f"bbpair_{name}"] = result

    return streams


def adaptive_portfolio(all_streams, fred, min_warmup=504):
    df = pd.DataFrame(all_streams).dropna(how="all").dropna(thresh=5).fillna(0)
    vol_t = pd.DataFrame(index=df.index)
    for col in df.columns:
        rv = df[col].rolling(63,min_periods=21).std()*np.sqrt(252)
        sc = (0.03/rv.clip(lower=0.003)).clip(0.1,8.0)
        vol_t[col] = df[col]*sc.shift(1)
    vol_t = vol_t.fillna(0)

    portfolio_ret = pd.Series(0.0, index=vol_t.index)
    start_idx = min_warmup
    if start_idx >= len(vol_t): return None, 0
    current_weights = pd.Series(0.0, index=vol_t.columns)
    active_counts = []

    for i in range(start_idx, len(vol_t)):
        if (i-start_idx) % REBAL_FREQ == 0:
            eval_data = vol_t.iloc[max(0,i-EVAL_WINDOW):i]
            trailing_sharpe = {}
            for col in vol_t.columns:
                s = eval_data[col]
                if s.std()>0 and s.count()>=63:
                    trailing_sharpe[col] = s.mean()/s.std()*np.sqrt(252)
            selected = {k:v for k,v in trailing_sharpe.items() if v>0}
            if selected:
                sv = pd.Series(selected)
                sv = sv.clip(upper=sv.quantile(0.9))
                weights = sv/sv.sum()
                current_weights = pd.Series(0.0, index=vol_t.columns)
                for k,w in weights.items(): current_weights[k] = w
            else:
                current_weights = pd.Series(0.0, index=vol_t.columns)
            active_counts.append(len(selected))
        portfolio_ret.iloc[i] = (current_weights*vol_t.iloc[i]).sum()

    portfolio_ret = portfolio_ret.iloc[start_idx:]
    vix = fred.get("VIXCLS")
    if vix is not None:
        vix_a = vix.reindex(portfolio_ret.index).ffill()
        vix_pctl = vix_a.rolling(252,min_periods=126).rank(pct=True)
        stress = (1.2-0.6*vix_pctl).clip(0.5,1.2)
        portfolio_ret = portfolio_ret*stress.shift(1)
    cum = (1+portfolio_ret).cumprod()
    dd = (cum-cum.cummax())/cum.cummax()
    dd_scale = np.exp(dd*5).clip(0.2,1.0)
    portfolio_ret = portfolio_ret*dd_scale.shift(1)
    pv = portfolio_ret.rolling(63,min_periods=21).std()*np.sqrt(252)
    ps = (TARGET_VOL/pv.clip(lower=0.005)).clip(0.2,5.0)
    portfolio_ret = portfolio_ret*ps.shift(1)
    return portfolio_ret.dropna(), np.mean(active_counts) if active_counts else 0


def metrics(r):
    r=r.dropna()
    if len(r)<60: return None
    ar=r.mean()*252; av=r.std()*np.sqrt(252); sr=ar/av if av>0 else 0
    cum=(1+r).cumprod(); mdd=((cum-cum.cummax())/cum.cummax()).min()
    cal=ar/abs(mdd) if mdd!=0 else 0; wr=(r>0).mean()
    ds=r[r<0].std()*np.sqrt(252) if (r<0).any() else av
    sortino=ar/ds if ds>0 else 0
    return {"sr":sr,"ret":ar,"vol":av,"mdd":mdd,"cal":cal,"wr":wr,"sortino":sortino,"n":len(r)}


def main():
    print("="*80)
    print("V10: CROSS-ASSET CARRY + LEVERAGED ETF VOL DECAY")
    print("="*80)
    prices, fred = load_all_data()
    ret = prices.pct_change()
    print(f"Universe: {prices.shape[1]} ETFs")

    streams = generate_all_streams(ret, fred)
    print(f"Total streams: {len(streams)}")

    # Count by engine
    eng = {}
    for n in streams:
        p = n.split("_")[0]
        eng[p] = eng.get(p,0)+1
    for e,c in sorted(eng.items()): print(f"  {e}: {c}")

    # Engine-level perf
    print(f"\n--- Engine Performance ---")
    engine_groups = {}
    for name, s in streams.items():
        prefix = name.split("_")[0]
        engine_groups.setdefault(prefix, {})[name] = s
    for eng_name, eng_dict in sorted(engine_groups.items()):
        edf = pd.DataFrame(eng_dict).dropna(how="all").fillna(0)
        er = edf.mean(axis=1)
        m = metrics(er)
        if m: print(f"  {eng_name:12s}: Sharpe={m['sr']:+.3f}  Ret={m['ret']*100:+.1f}%  Streams={len(eng_dict)}")

    # Top streams
    print(f"\n--- Top 15 Streams ---")
    sm = {n: metrics(s) for n,s in streams.items() if metrics(s)}
    for n,m in sorted(sm.items(), key=lambda x:-x[1]["sr"])[:15]:
        print(f"  {n:30s}: Sharpe={m['sr']:+.3f}  Ret={m['ret']*100:+.1f}%")

    # Portfolio
    print(f"\n{'='*80}")
    port, avg_act = adaptive_portfolio(streams, fred)
    if port is None: print("FAILED!"); return
    m = metrics(port)
    print(f"FULL SAMPLE ({avg_act:.0f} active):")
    print(f"  Sharpe:  {m['sr']:.3f}  |  Sortino:  {m['sortino']:.3f}")
    print(f"  Return:  {m['ret']*100:+.2f}%  |  Vol:  {m['vol']*100:.2f}%")
    print(f"  MaxDD:   {m['mdd']*100:.2f}%  |  Calmar:  {m['cal']:.3f}")
    print(f"  WinRate: {m['wr']*100:.1f}%  |  Days:  {m['n']}")

    sp = int(len(port)*0.6)
    for nm,r in [("TRAIN",port.iloc[:sp]),("TEST",port.iloc[sp:])]:
        m2=metrics(r)
        if m2: print(f"\n  {nm}: Sharpe={m2['sr']:.3f}  Ret={m2['ret']*100:+.2f}%  MaxDD={m2['mdd']*100:.2f}%  Sortino={m2['sortino']:.3f}")

    print(f"\n  {'Year':>6} {'Ret':>9} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8}")
    for yr,g in port.groupby(port.index.year):
        if len(g)<20: continue
        ar=g.mean()*252; av=g.std()*np.sqrt(252); sr=ar/av if av>0 else 0
        c=(1+g).cumprod(); mdd=((c-c.cummax())/c.cummax()).min()
        print(f"  {yr:>6} {ar*100:>+8.2f}% {av*100:>7.2f}% {sr:>+7.3f} {mdd*100:>+7.2f}%")

    # Walk-forward
    print(f"\n  WALK-FORWARD:")
    nt=len(port); fs=nt//6; wf=[]
    for fold in range(5):
        s=(fold+1)*fs; e=min(s+fs,nt); fr=port.iloc[s:e]; fm=metrics(fr)
        if fm:
            wf.append(fm['sr'])
            print(f"    Fold {fold+1} ({fr.index[0].date()} to {fr.index[-1].date()}): Sharpe={fm['sr']:.3f}")
    if wf: print(f"    Mean: {np.mean(wf):.3f}  Std: {np.std(wf):.3f}  Min: {np.min(wf):.3f}")

    print(f"\n  Autocorr(1): {port.autocorr(1):.4f}")
    n_trials = len(streams)*2
    dsr = np.sqrt(2*np.log(n_trials))/np.sqrt(m['n']/252)
    print(f"  Deflated Sharpe: {m['sr']-dsr:.3f} (raw {m['sr']:.3f} - {dsr:.3f})")

    # Diversification
    sdf = pd.DataFrame({k:v for k,v in streams.items() if len(v.dropna())>=504}).dropna(how="all").fillna(0)
    cr=sdf.corr(); up=cr.where(np.triu(np.ones(cr.shape),k=1).astype(bool))
    ac=up.stack().mean(); n=sdf.shape[1]
    dm=np.sqrt(n*(1-ac)/(1+(n-1)*ac)) if (1+(n-1)*ac)>0 else 1
    print(f"  Avg corr: {ac:.3f}  Streams: {n}  Div mult: {dm:.2f}x")

    # Save
    rd=DATA_DIR/"results"; rd.mkdir(exist_ok=True)
    port.to_csv(rd/"strategy_v10_returns.csv", header=["return"])
    (1+port).cumprod().to_csv(rd/"strategy_v10_cumulative.csv", header=["cumulative"])
    port.to_csv(rd/"dichs_returns.csv", header=["return"])
    (1+port).cumprod().to_csv(rd/"dichs_cumulative.csv", header=["cumulative"])
    print(f"\n  Saved to {rd}")

if __name__ == "__main__":
    main()
