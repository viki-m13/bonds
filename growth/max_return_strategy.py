#!/usr/bin/env python3
"""
GROWTH STRATEGY: Maximize Total Return
========================================
Separate from the main DICHS strategy (which maximizes Sharpe).

Key differences from DICHS:
- Optimize for RETURN, not risk-adjusted return
- Higher vol target (15-20% instead of 10%)
- No drawdown control (let it ride)
- Fewer hedges (hedges reduce return)
- Concentrate into highest-returning streams
- Include leveraged ETFs for amplification
- Weight by trailing RETURN, not Sharpe
- Accept higher drawdowns for higher compounding

STILL MAINTAINS:
- No look-ahead bias (shift(1) on all signals)
- Transaction costs modeled
- Walk-forward validation
- Monthly rebalancing
"""

import pandas as pd
import numpy as np
import sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

DATA_DIR = Path(__file__).parent.parent / "data"
ETF_DIR = DATA_DIR / "etfs"
FRED_PATH = DATA_DIR / "fred" / "_combined_fred.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TC_BPS = 5
REBAL_FREQ = 21
EVAL_WINDOW = 252


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


def hedged(ret, l, h, hw):
    if l not in ret.columns or h not in ret.columns: return None
    w1 = 1.0/(1.0+hw); w2 = hw/(1.0+hw)
    return (w1*ret[l]+w2*ret[h]).dropna()


def generate_growth_streams(ret, fred):
    """
    Generate streams optimized for RETURN, not Sharpe.
    Key differences:
    - Include unhedged positions (higher return, higher vol)
    - Include leveraged ETFs directly
    - Include momentum on high-beta assets
    - Fewer inverse ETF hedges
    """
    streams = {}
    MIN = 252

    # ===== UNHEDGED HIGH-RETURN ASSETS (no inverse ETF drag) =====
    for etf in ["QQQ","SPY","IWM","SMH","ARKK","TQQQ","UPRO","SSO","QLD",
                "VNQ","IYR","AMLP","XLK","XLE","XLF","XLI","XLY",
                "EEM","EFA","EWJ","EWZ","FXI","INDA","EWY",
                "GLD","SLV","DBC","URA","KWEB","TAN","LIT",
                "IBIT","GBTC","BITO"]:
        if etf not in ret.columns: continue
        # Pure long momentum
        sigs = []
        for lb in [21,63,126,252]:
            pr = ret[etf].rolling(lb,min_periods=int(lb*0.7)).mean()*np.sqrt(252)
            pv = ret[etf].rolling(lb,min_periods=int(lb*0.7)).std()*np.sqrt(252)
            sigs.append(pr/pv.clip(lower=0.01))
        cs = pd.concat(sigs,axis=1).mean(axis=1)
        pos = cs.clip(0,2)/2
        sr = pos.shift(1)*ret[etf] - pos.diff().abs()*(TC_BPS/10000)
        if len(sr.dropna())>=MIN: streams[f"mom_{etf}"] = sr.dropna()

    # ===== LEVERAGED MOMENTUM (3x for max growth) =====
    for etf in ["TQQQ","UPRO","SOXL","TECL","FAS","TMF","ERX",
                "LABU","EDC","DRN","YINN"]:
        if etf not in ret.columns: continue
        sigs = []
        for lb in [21,63,126]:
            pr = ret[etf].rolling(lb,min_periods=int(lb*0.7)).mean()*np.sqrt(252)
            pv = ret[etf].rolling(lb,min_periods=int(lb*0.7)).std()*np.sqrt(252)
            sigs.append(pr/pv.clip(lower=0.01))
        cs = pd.concat(sigs,axis=1).mean(axis=1)
        pos = cs.clip(0,2)/2
        sr = pos.shift(1)*ret[etf] - pos.diff().abs()*(10/10000)
        if len(sr.dropna())>=MIN: streams[f"lev_{etf}"] = sr.dropna()

    # ===== CRYPTO MOMENTUM (highest return potential) =====
    for etf in ["BTC_USD","ETH_USD","SOL_USD","ADA_USD","GBTC","IBIT","BITO","ETHE"]:
        if etf not in ret.columns: continue
        sigs = []
        for lb in [21,63,126,252]:
            pr = ret[etf].rolling(lb,min_periods=int(lb*0.7)).mean()*np.sqrt(252)
            pv = ret[etf].rolling(lb,min_periods=int(lb*0.7)).std()*np.sqrt(252)
            sigs.append(pr/pv.clip(lower=0.01))
        cs = pd.concat(sigs,axis=1).mean(axis=1)
        pos = cs.clip(0,2)/2
        sr = pos.shift(1)*ret[etf] - pos.diff().abs()*(15/10000)
        if len(sr.dropna())>=MIN: streams[f"crypto_{etf}"] = sr.dropna()

    # ===== HIGH-YIELD CARRY (unhedged for max return) =====
    for etf in ["HYG","JNK","BKLN","SRLN","EMB","EMLC","PFF","AMLP"]:
        if etf in ret.columns:
            r = ret[etf].dropna()
            if len(r)>=MIN: streams[f"carry_{etf}"] = r

    # ===== DIVIDEND EQUITY (unhedged for total return) =====
    for etf in ["SCHD","HDV","DVY","VIG","XLP","XLU"]:
        if etf in ret.columns:
            r = ret[etf].dropna()
            if len(r)>=MIN: streams[f"div_{etf}"] = r

    # ===== LIGHTLY HEDGED CARRY (keep some hedging for crisis protection) =====
    for l,h,hw,n in [
        ("HYG","TBF",0.2,"hyg_tbf"),("SCHD","SH",0.15,"schd_sh"),
        ("HDV","SH",0.15,"hdv_sh"),("AMLP","SH",0.15,"amlp_sh"),
        ("QQQ","SQQQ",0.05,"qqq_sqqq"),("SPY","SPXU",0.05,"spy_spxu"),
    ]:
        r = hedged(ret, l, h, hw)
        if r is not None and len(r)>=MIN: streams[f"lh_{n}"] = r

    # ===== CLO (high carry, low vol = good for compounding) =====
    for etf in ["JAAA","JBBB"]:
        if etf in ret.columns:
            r = ret[etf].dropna()
            if len(r)>=MIN: streams[f"clo_{etf}"] = r

    return streams


def run_growth_portfolio(streams, fred, top_n=5, target_vol=0.15,
                          weighting="return", min_warmup=504,
                          dd_control=False, dd_strength=3):
    """
    Portfolio construction optimized for growth:
    - Weight by trailing RETURN (not Sharpe)
    - Higher vol target
    - Optional drawdown control (off by default)
    """
    df = pd.DataFrame(streams).dropna(how="all").dropna(thresh=5).fillna(0)

    # Vol-target each stream to 5% (higher than DICHS's 3%)
    vol_t = pd.DataFrame(index=df.index)
    for col in df.columns:
        rv = df[col].rolling(63,min_periods=21).std()*np.sqrt(252)
        sc = (0.05/rv.clip(lower=0.003)).clip(0.1,10.0)
        vol_t[col] = df[col]*sc.shift(1)
    vol_t = vol_t.fillna(0)

    portfolio_ret = pd.Series(0.0, index=vol_t.index)
    start_idx = min_warmup
    if start_idx >= len(vol_t): return None
    cw = pd.Series(0.0, index=vol_t.columns)
    last_selected = {}

    for i in range(start_idx, len(vol_t)):
        if (i-start_idx) % REBAL_FREQ == 0:
            eval_data = vol_t.iloc[max(0,i-EVAL_WINDOW):i]

            if weighting == "return":
                # Weight by trailing annualized return (not Sharpe)
                trailing_ret = {}
                for col in vol_t.columns:
                    s = eval_data[col]
                    if s.count() >= 63:
                        tr = s.mean() * 252  # Annualized return
                        if tr > 0:
                            trailing_ret[col] = tr
                selected = trailing_ret
            elif weighting == "sharpe":
                trailing_sharpe = {}
                for col in vol_t.columns:
                    s = eval_data[col]
                    if s.std()>0 and s.count()>=63:
                        ts = s.mean()/s.std()*np.sqrt(252)
                        if ts > 0:
                            trailing_sharpe[col] = ts
                selected = trailing_sharpe
            elif weighting == "return_sq":
                trailing_ret = {}
                for col in vol_t.columns:
                    s = eval_data[col]
                    if s.count() >= 63:
                        tr = s.mean() * 252
                        if tr > 0:
                            trailing_ret[col] = tr
                selected = trailing_ret
            else:
                selected = {}

            if selected:
                sv = pd.Series(selected).nlargest(top_n)
                sv = sv.clip(upper=sv.quantile(0.9) if len(sv)>3 else 99)

                if weighting == "return_sq":
                    sq = sv**2
                    w = sq/sq.sum()
                else:
                    w = sv/sv.sum()

                cw = pd.Series(0.0, index=vol_t.columns)
                for k,v in w.items(): cw[k] = v
                last_selected = dict(zip(w.index, w.values))
            else:
                cw = pd.Series(0.0, index=vol_t.columns)

        portfolio_ret.iloc[i] = (cw*vol_t.iloc[i]).sum()

    p = portfolio_ret.iloc[start_idx:]

    # Optional VIX scaling (lighter than DICHS)
    vix = fred.get("VIXCLS")
    if vix is not None:
        va = vix.reindex(p.index).ffill()
        vp = va.rolling(252,min_periods=126).rank(pct=True)
        # Lighter scaling: only reduce at very high VIX
        stress = (1.1 - 0.3*vp).clip(0.7, 1.1)
        p = p * stress.shift(1)

    # Optional drawdown control
    if dd_control:
        cum = (1+p).cumprod()
        dd = (cum-cum.cummax())/cum.cummax()
        dd_scale = np.exp(dd*dd_strength).clip(0.3,1.0)
        p = p*dd_scale.shift(1)

    # Vol target (higher than DICHS)
    pv = p.rolling(63,min_periods=21).std()*np.sqrt(252)
    ps = (target_vol/pv.clip(lower=0.005)).clip(0.2,5.0)
    p = p*ps.shift(1)

    return p.dropna(), last_selected


def m(r, name=""):
    r=r.dropna()
    if len(r)<60: return None
    ar=r.mean()*252; av=r.std()*np.sqrt(252); sr=ar/av if av>0 else 0
    cum=(1+r).cumprod(); mdd=((cum-cum.cummax())/cum.cummax()).min()
    cal=ar/abs(mdd) if mdd!=0 else 0; wr=(r>0).mean()
    ds=r[r<0].std()*np.sqrt(252) if (r<0).any() else av
    sortino=ar/ds if ds>0 else 0
    total = cum.iloc[-1]-1
    sp = int(len(r)*0.6)
    test_sr = r.iloc[sp:].mean()/r.iloc[sp:].std()*np.sqrt(252) if r.iloc[sp:].std()>0 else 0
    test_ret = r.iloc[sp:].mean()*252
    nt=len(r); fs=nt//6; wf_sr=[]; wf_ret=[]
    for fold in range(5):
        s=(fold+1)*fs; e=min(s+fs,nt)
        fr=r.iloc[s:e]
        if len(fr)>60 and fr.std()>0:
            wf_sr.append(fr.mean()/fr.std()*np.sqrt(252))
            wf_ret.append(fr.mean()*252)
    return {"name":name,"sharpe":round(sr,3),"ann_ret":round(ar*100,2),"ann_vol":round(av*100,2),
            "max_dd":round(mdd*100,2),"calmar":round(cal,3),"win_rate":round(wr*100,1),
            "sortino":round(sortino,3),"total_ret":round(total*100,2),
            "test_sr":round(test_sr,3),"test_ret":round(test_ret*100,2),
            "wf_sr_mean":round(np.mean(wf_sr),3) if wf_sr else 0,
            "wf_ret_mean":round(np.mean(wf_ret)*100,2) if wf_ret else 0,
            "ac1":round(r.autocorr(1),4),"n":len(r),
            "final_nav":round(float(cum.iloc[-1]),2)}


def main():
    print("="*80)
    print("GROWTH STRATEGY: MAXIMIZE TOTAL RETURN")
    print("="*80)

    prices, fred = load_all_data()
    ret = prices.pct_change()
    print(f"Universe: {len(prices.columns)} ETFs")

    streams = generate_growth_streams(ret, fred)
    print(f"Growth streams: {len(streams)}")

    # Count by engine
    eng = {}
    for n in streams:
        p = n.split("_")[0]
        eng[p] = eng.get(p,0)+1
    for e,c in sorted(eng.items()): print(f"  {e}: {c}")

    # Top 20 streams by raw return
    print(f"\n--- Top 20 Streams by Annualized Return ---")
    stream_m = {}
    for name, s in streams.items():
        mx = m(s, name)
        if mx: stream_m[name] = mx
    for name, mx in sorted(stream_m.items(), key=lambda x: -x[1]["ann_ret"])[:20]:
        print(f"  {name:30s}: Ret={mx['ann_ret']:+.1f}%  SR={mx['sharpe']:.3f}  Vol={mx['ann_vol']:.1f}%  MDD={mx['max_dd']:.1f}%")

    # ================================================================
    # EXPERIMENT GRID
    # ================================================================
    results = []

    configs = [
        # (name, top_n, target_vol, weighting, dd_control, dd_strength)
        ("Growth_Top3_15vol", 3, 0.15, "return", False, 0),
        ("Growth_Top3_20vol", 3, 0.20, "return", False, 0),
        ("Growth_Top5_15vol", 5, 0.15, "return", False, 0),
        ("Growth_Top5_20vol", 5, 0.20, "return", False, 0),
        ("Growth_Top5_25vol", 5, 0.25, "return", False, 0),
        ("Growth_Top3_RetSq", 3, 0.20, "return_sq", False, 0),
        ("Growth_Top5_RetSq", 5, 0.20, "return_sq", False, 0),
        ("Growth_Top3_DD", 3, 0.20, "return", True, 3),
        ("Growth_Top5_DD", 5, 0.20, "return", True, 3),
        ("Growth_Top3_Sharpe", 3, 0.20, "sharpe", False, 0),
        ("Growth_Top5_Sharpe", 5, 0.20, "sharpe", False, 0),
        # Ultra aggressive
        ("Ultra_Top3_30vol", 3, 0.30, "return_sq", False, 0),
        ("Ultra_Top3_30vol_DD", 3, 0.30, "return_sq", True, 3),
    ]

    for name, tn, tv, wt, dd, dds in configs:
        p, lw = run_growth_portfolio(streams, fred, top_n=tn, target_vol=tv,
                                      weighting=wt, dd_control=dd, dd_strength=dds)
        if p is not None:
            mx = m(p, name)
            results.append(mx)
            print(f"\n  {name:25s}: Ret={mx['ann_ret']:+.1f}%  SR={mx['sharpe']:.3f}  "
                  f"Vol={mx['ann_vol']:.1f}%  MDD={mx['max_dd']:.1f}%  NAV={mx['final_nav']}x  "
                  f"TestRet={mx['test_ret']:+.1f}%  WF_Ret={mx['wf_ret_mean']:+.1f}%")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*80}")
    print("SUMMARY: RANKED BY ANNUALIZED RETURN")
    print(f"{'='*80}")
    print(f"{'Name':25s} {'Ret':>8} {'Vol':>8} {'SR':>7} {'MDD':>8} {'NAV':>8} {'TestRet':>9} {'WF Ret':>9}")
    print("-"*80)
    for r in sorted(results, key=lambda x: -x["ann_ret"]):
        print(f"  {r['name']:23s} {r['ann_ret']:>+7.1f}% {r['ann_vol']:>7.1f}% {r['sharpe']:>6.3f} "
              f"{r['max_dd']:>+7.1f}% {r['final_nav']:>7.1f}x {r['test_ret']:>+8.1f}% {r['wf_ret_mean']:>+8.1f}%")

    # Best by return
    best = max(results, key=lambda x: x["ann_ret"])
    print(f"\n  BEST BY RETURN: {best['name']}")
    print(f"    Ann Return: {best['ann_ret']:+.2f}%")
    print(f"    Sharpe:     {best['sharpe']:.3f}")
    print(f"    Max DD:     {best['max_dd']:.2f}%")
    print(f"    Final NAV:  {best['final_nav']}x")

    # Compare to DICHS
    print(f"\n{'='*80}")
    print("COMPARISON TO DICHS (SHARPE-OPTIMIZED)")
    print(f"{'='*80}")
    dichs = pd.read_csv(DATA_DIR/"results"/"dichs_returns.csv", parse_dates=[0])
    dichs.columns=["Date","return"]; dichs=dichs.set_index("Date")["return"]
    dm = m(dichs, "DICHS")
    if dm:
        print(f"  {'':25s} {'DICHS':>12} {'Best Growth':>12}")
        print(f"  {'Ann Return':25s} {dm['ann_ret']:>+11.1f}% {best['ann_ret']:>+11.1f}%")
        print(f"  {'Sharpe':25s} {dm['sharpe']:>12.3f} {best['sharpe']:>12.3f}")
        print(f"  {'Max Drawdown':25s} {dm['max_dd']:>11.1f}% {best['max_dd']:>11.1f}%")
        print(f"  {'Final NAV':25s} {dm['final_nav']:>11.1f}x {best['final_nav']:>11.1f}x")
        print(f"  {'Vol':25s} {dm['ann_vol']:>11.1f}% {best['ann_vol']:>11.1f}%")
        print(f"  {'Sortino':25s} {dm['sortino']:>12.3f} {best['sortino']:>12.3f}")

    # Save results
    with open(RESULTS_DIR/"growth_experiments.json", "w") as f:
        json.dump({"experiments": results, "dichs": dm, "best": best}, f, indent=2)

    # Save best strategy returns
    p_best, _ = run_growth_portfolio(
        streams, fred,
        top_n=int(best["name"].split("_")[1].replace("Top","")) if "Top" in best["name"] else 5,
        target_vol=float(best["name"].split("vol")[0].split("_")[-1].replace("v",""))/100 if "vol" in best["name"] else 0.20,
        weighting="return_sq" if "RetSq" in best["name"] else ("sharpe" if "Sharpe" in best["name"] else "return"),
        dd_control="DD" in best["name"],
        dd_strength=3 if "DD" in best["name"] else 0,
    )
    if p_best is not None:
        p_best.to_csv(RESULTS_DIR/"growth_best_returns.csv", header=["return"])
        (1+p_best).cumprod().to_csv(RESULTS_DIR/"growth_best_cumulative.csv", header=["cumulative"])

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
