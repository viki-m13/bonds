#!/usr/bin/env python3
"""
GROWTH V2: Push for maximum possible returns.

Ideas to test:
1. More leveraged ETFs (3x only, drop 1x/2x)
2. Concentration: Top 1 or Top 2 instead of Top 3
3. Higher vol target (40-50%)
4. No VIX scaling at all (let it ride)
5. Use RETURN momentum (not risk-adjusted) for signal
6. Shorter lookback for faster trend-following
7. Include leveraged crypto (BITX = 2x Bitcoin)
8. Combo: leveraged ETF + crypto + no hedging
9. Dollar-cost-averaging into winners (scale up, not rebalance flat)
10. Momentum crash protection: only exit, never reduce to partial
"""
import pandas as pd, numpy as np, sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

DATA_DIR = Path(__file__).parent.parent / "data"
ETF_DIR = DATA_DIR / "etfs"
FRED_PATH = DATA_DIR / "fred" / "_combined_fred.csv"
RESULTS_DIR = Path(__file__).parent / "results"

def load():
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

def gen_streams(ret):
    streams = {}
    MIN = 252

    # === PURE 3x LEVERAGED MOMENTUM ===
    for etf in ["TQQQ","UPRO","SOXL","TECL","FAS","ERX","LABU","EDC","DRN","YINN","TMF"]:
        if etf not in ret.columns: continue
        for lookbacks, name_suffix in [
            ([10,21,42], "fast"),
            ([21,63,126], "med"),
            ([63,126,252], "slow"),
        ]:
            sigs = []
            for lb in lookbacks:
                pr = ret[etf].rolling(lb,min_periods=max(5,int(lb*0.6))).mean()*np.sqrt(252)
                pv = ret[etf].rolling(lb,min_periods=max(5,int(lb*0.6))).std()*np.sqrt(252)
                sigs.append(pr/pv.clip(lower=0.01))
            cs = pd.concat(sigs,axis=1).mean(axis=1)
            pos = cs.clip(0,2)/2
            sr = pos.shift(1)*ret[etf] - pos.diff().abs()*(10/10000)
            if len(sr.dropna())>=MIN: streams[f"lev3x_{etf}_{name_suffix}"] = sr.dropna()

    # === 2x LEVERAGED MOMENTUM ===
    for etf in ["SSO","QLD","UBT","UGL","UCO"]:
        if etf not in ret.columns: continue
        sigs = []
        for lb in [21,63,126]:
            pr = ret[etf].rolling(lb,min_periods=int(lb*0.7)).mean()*np.sqrt(252)
            pv = ret[etf].rolling(lb,min_periods=int(lb*0.7)).std()*np.sqrt(252)
            sigs.append(pr/pv.clip(lower=0.01))
        cs = pd.concat(sigs,axis=1).mean(axis=1)
        pos = cs.clip(0,2)/2
        sr = pos.shift(1)*ret[etf] - pos.diff().abs()*(10/10000)
        if len(sr.dropna())>=MIN: streams[f"lev2x_{etf}"] = sr.dropna()

    # === 1x MOMENTUM (high-beta assets) ===
    for etf in ["QQQ","SPY","IWM","SMH","ARKK","VNQ","EEM","EWZ","FXI",
                "KWEB","TAN","LIT","URA","XLE","XLK","INDA","EWY"]:
        if etf not in ret.columns: continue
        sigs = []
        for lb in [21,63,126,252]:
            pr = ret[etf].rolling(lb,min_periods=int(lb*0.7)).mean()*np.sqrt(252)
            pv = ret[etf].rolling(lb,min_periods=int(lb*0.7)).std()*np.sqrt(252)
            sigs.append(pr/pv.clip(lower=0.01))
        cs = pd.concat(sigs,axis=1).mean(axis=1)
        pos = cs.clip(0,2)/2
        sr = pos.shift(1)*ret[etf] - pos.diff().abs()*(5/10000)
        if len(sr.dropna())>=MIN: streams[f"mom_{etf}"] = sr.dropna()

    # === CRYPTO MOMENTUM ===
    for etf in ["BTC_USD","ETH_USD","SOL_USD","ADA_USD","GBTC","IBIT","BITO","ETHE","BITX"]:
        if etf not in ret.columns: continue
        for lookbacks, name_suffix in [
            ([10,21,42], "fast"),
            ([21,63,126,252], "std"),
        ]:
            sigs = []
            for lb in lookbacks:
                pr = ret[etf].rolling(lb,min_periods=max(5,int(lb*0.6))).mean()*np.sqrt(252)
                pv = ret[etf].rolling(lb,min_periods=max(5,int(lb*0.6))).std()*np.sqrt(252)
                sigs.append(pr/pv.clip(lower=0.01))
            cs = pd.concat(sigs,axis=1).mean(axis=1)
            pos = cs.clip(0,2)/2
            sr = pos.shift(1)*ret[etf] - pos.diff().abs()*(15/10000)
            if len(sr.dropna())>=MIN: streams[f"crypto_{etf}_{name_suffix}"] = sr.dropna()

    # === RAW RETURN MOMENTUM (not risk-adjusted) ===
    for etf in ["TQQQ","UPRO","SOXL","BTC_USD","ETH_USD","SOL_USD","QQQ","SMH"]:
        if etf not in ret.columns: continue
        # Signal: raw trailing return > 0 → go long
        for lb in [21,63,126]:
            raw_ret = ret[etf].rolling(lb,min_periods=int(lb*0.6)).sum()
            pos = (raw_ret > 0).astype(float)
            sr = pos.shift(1)*ret[etf] - pos.diff().abs()*(10/10000)
            if len(sr.dropna())>=MIN: streams[f"rawmom_{etf}_{lb}d"] = sr.dropna()

    # === PURE BUY-AND-HOLD (for comparison) ===
    for etf in ["TQQQ","UPRO","SOXL","QQQ","SPY","BTC_USD","IBIT","SCHD","JAAA"]:
        if etf in ret.columns:
            r = ret[etf].dropna()
            if len(r)>=MIN: streams[f"bah_{etf}"] = r

    # === UNHEDGED CARRY (high yield) ===
    for etf in ["HYG","JNK","BKLN","SRLN","AMLP","PFF","EMLC","EMB","JAAA"]:
        if etf in ret.columns:
            r = ret[etf].dropna()
            if len(r)>=MIN: streams[f"carry_{etf}"] = r

    return streams


def run_portfolio(streams, fred, top_n=3, target_vol=0.30, 
                   weighting="return_sq", eval_window=252,
                   rebal_freq=21, stream_vol=0.05,
                   vix_scale=False, dd_control=False, dd_strength=3,
                   min_warmup=504, min_trailing=0.0):
    df = pd.DataFrame(streams).dropna(how="all").dropna(thresh=5).fillna(0)
    vol_t = pd.DataFrame(index=df.index)
    for col in df.columns:
        rv = df[col].rolling(63,min_periods=21).std()*np.sqrt(252)
        sc = (stream_vol/rv.clip(lower=0.003)).clip(0.1,10.0)
        vol_t[col] = df[col]*sc.shift(1)
    vol_t = vol_t.fillna(0)

    p = pd.Series(0.0, index=vol_t.index)
    start = min_warmup
    if start >= len(vol_t): return None, {}
    cw = pd.Series(0.0, index=vol_t.columns)
    last = {}

    for i in range(start, len(vol_t)):
        if (i-start) % rebal_freq == 0:
            ev = vol_t.iloc[max(0,i-eval_window):i]
            scores = {}
            for col in vol_t.columns:
                s = ev[col]
                if s.count() < 63: continue
                if weighting in ["return","return_sq"]:
                    tr = s.mean() * 252
                    if tr > min_trailing: scores[col] = tr
                elif weighting == "sharpe":
                    if s.std() > 0:
                        sr = s.mean()/s.std()*np.sqrt(252)
                        if sr > min_trailing: scores[col] = sr
                else:
                    tr = s.mean()*252
                    if tr > 0: scores[col] = tr

            if scores:
                sv = pd.Series(scores).nlargest(top_n)
                sv = sv.clip(upper=sv.quantile(0.9) if len(sv)>3 else 99)
                if weighting == "return_sq":
                    sq = sv**2; w = sq/sq.sum()
                else:
                    w = sv/sv.sum()
                cw = pd.Series(0.0, index=vol_t.columns)
                for k,v in w.items(): cw[k] = v
                last = dict(zip(w.index, w.values))
            else:
                cw = pd.Series(0.0, index=vol_t.columns)
        p.iloc[i] = (cw*vol_t.iloc[i]).sum()

    p = p.iloc[start:]

    if vix_scale:
        vix = fred.get("VIXCLS")
        if vix is not None:
            va = vix.reindex(p.index).ffill()
            vp = va.rolling(252,min_periods=126).rank(pct=True)
            p = p*(1.1-0.3*vp).clip(0.7,1.1).shift(1)

    if dd_control:
        cum = (1+p).cumprod(); dd = (cum-cum.cummax())/cum.cummax()
        p = p*np.exp(dd*dd_strength).clip(0.3,1.0).shift(1)

    pv = p.rolling(63,min_periods=21).std()*np.sqrt(252)
    ps = (target_vol/pv.clip(lower=0.005)).clip(0.2,8.0)
    p = p*ps.shift(1)

    return p.dropna(), last


def m(r, name=""):
    r=r.dropna()
    if len(r)<60: return None
    ar=r.mean()*252; av=r.std()*np.sqrt(252); sr=ar/av if av>0 else 0
    cum=(1+r).cumprod(); mdd=((cum-cum.cummax())/cum.cummax()).min()
    wr=(r>0).mean(); total=cum.iloc[-1]-1
    sp=int(len(r)*0.6)
    test_ret = r.iloc[sp:].mean()*252
    nt=len(r); fs=nt//6; wf=[]
    for fold in range(5):
        s=(fold+1)*fs; e=min(s+fs,nt); fr=r.iloc[s:e]
        if len(fr)>60: wf.append(fr.mean()*252)
    return {"name":name,"sharpe":round(sr,3),"ann_ret":round(ar*100,2),
            "ann_vol":round(av*100,2),"max_dd":round(mdd*100,2),
            "win_rate":round(wr*100,1),"total_ret":round(total*100,1),
            "test_ret":round(test_ret*100,2),
            "wf_ret_mean":round(np.mean(wf)*100,2) if wf else 0,
            "final_nav":round(float(cum.iloc[-1]),1),
            "ac1":round(r.autocorr(1),4)}


print("="*80)
print("GROWTH V2: PUSHING FOR MAXIMUM RETURNS")
print("="*80)

prices, fred = load()
ret = prices.pct_change()
streams = gen_streams(ret)
print(f"Streams: {len(streams)}")
eng = {}
for n in streams:
    p = n.split("_")[0]
    eng[p] = eng.get(p,0)+1
for e,c in sorted(eng.items()): print(f"  {e}: {c}")

# Top 20 by raw return
print(f"\n--- Top 20 Streams by Return ---")
sm = {}
for name, s in streams.items():
    mx = m(s, name)
    if mx: sm[name] = mx
for name, mx in sorted(sm.items(), key=lambda x:-x[1]["ann_ret"])[:20]:
    print(f"  {name:30s}: Ret={mx['ann_ret']:>+7.1f}%  SR={mx['sharpe']:.3f}  Vol={mx['ann_vol']:>6.1f}%  MDD={mx['max_dd']:>7.1f}%")

# ================================================================
# MASSIVE EXPERIMENT GRID
# ================================================================
results = []

configs = [
    # (name, top_n, target_vol, weighting, eval_win, rebal_freq, stream_vol, vix, dd, dd_str, min_trail)
    # Concentration experiments
    ("Top1_30v", 1, 0.30, "return_sq", 252, 21, 0.05, False, False, 0, 0),
    ("Top1_40v", 1, 0.40, "return_sq", 252, 21, 0.05, False, False, 0, 0),
    ("Top1_50v", 1, 0.50, "return_sq", 252, 21, 0.05, False, False, 0, 0),
    ("Top2_30v", 2, 0.30, "return_sq", 252, 21, 0.05, False, False, 0, 0),
    ("Top2_40v", 2, 0.40, "return_sq", 252, 21, 0.05, False, False, 0, 0),
    ("Top2_50v", 2, 0.50, "return_sq", 252, 21, 0.05, False, False, 0, 0),
    ("Top3_30v", 3, 0.30, "return_sq", 252, 21, 0.05, False, False, 0, 0),
    ("Top3_40v", 3, 0.40, "return_sq", 252, 21, 0.05, False, False, 0, 0),
    ("Top3_50v", 3, 0.50, "return_sq", 252, 21, 0.05, False, False, 0, 0),
    # Higher stream vol (let individual streams be more aggressive)
    ("Top3_30v_sv8", 3, 0.30, "return_sq", 252, 21, 0.08, False, False, 0, 0),
    ("Top3_40v_sv8", 3, 0.40, "return_sq", 252, 21, 0.08, False, False, 0, 0),
    ("Top2_40v_sv8", 2, 0.40, "return_sq", 252, 21, 0.08, False, False, 0, 0),
    ("Top1_50v_sv10", 1, 0.50, "return_sq", 252, 21, 0.10, False, False, 0, 0),
    # Faster lookback
    ("Top3_30v_fast", 3, 0.30, "return_sq", 126, 21, 0.05, False, False, 0, 0),
    ("Top3_30v_vfast", 3, 0.30, "return_sq", 63, 21, 0.05, False, False, 0, 0),
    # Faster rebalance
    ("Top3_30v_10d", 3, 0.30, "return_sq", 252, 10, 0.05, False, False, 0, 0),
    ("Top3_30v_5d", 3, 0.30, "return_sq", 252, 5, 0.05, False, False, 0, 0),
    # With light DD control
    ("Top3_40v_DD", 3, 0.40, "return_sq", 252, 21, 0.05, False, True, 3, 0),
    ("Top2_50v_DD", 2, 0.50, "return_sq", 252, 21, 0.05, False, True, 3, 0),
    # Linear return weighting
    ("Top3_30v_lin", 3, 0.30, "return", 252, 21, 0.05, False, False, 0, 0),
    # Sharpe weighting (for comparison)
    ("Top3_30v_sr", 3, 0.30, "sharpe", 252, 21, 0.05, False, False, 0, 0),
    # Min trailing return threshold
    ("Top3_40v_min5", 3, 0.40, "return_sq", 252, 21, 0.05, False, False, 0, 0.05),
    ("Top3_40v_min10", 3, 0.40, "return_sq", 252, 21, 0.05, False, False, 0, 0.10),
    # Ultra aggressive
    ("Top1_60v_sv10", 1, 0.60, "return_sq", 252, 21, 0.10, False, False, 0, 0),
    ("Top1_60v_DD", 1, 0.60, "return_sq", 252, 21, 0.10, False, True, 3, 0),
    ("Top2_60v", 2, 0.60, "return_sq", 252, 21, 0.08, False, False, 0, 0),
]

for name, tn, tv, wt, ew, rf, sv, vix, dd, dds, mt in configs:
    p, lw = run_portfolio(streams, fred, top_n=tn, target_vol=tv, weighting=wt,
                          eval_window=ew, rebal_freq=rf, stream_vol=sv,
                          vix_scale=vix, dd_control=dd, dd_strength=dds, min_trailing=mt)
    if p is not None:
        mx = m(p, name)
        results.append(mx)

# Summary
print(f"\n{'='*80}")
print("RANKED BY ANNUALIZED RETURN")
print(f"{'='*80}")
print(f"{'Name':22s} {'Ret':>8} {'Vol':>7} {'SR':>6} {'MDD':>8} {'NAV':>8} {'TestRet':>9} {'WF Ret':>8} {'AC1':>6}")
print("-"*85)
for r in sorted(results, key=lambda x:-x["ann_ret"])[:30]:
    print(f"  {r['name']:20s} {r['ann_ret']:>+7.1f}% {r['ann_vol']:>6.1f}% {r['sharpe']:>5.3f} "
          f"{r['max_dd']:>+7.1f}% {r['final_nav']:>7.1f}x {r['test_ret']:>+8.1f}% {r['wf_ret_mean']:>+7.1f}% {r['ac1']:>5.3f}")

best = max(results, key=lambda x: x["ann_ret"])
print(f"\nBEST: {best['name']} → {best['ann_ret']:+.1f}% return, {best['final_nav']}x NAV, {best['max_dd']}% MDD")

# Current positions for best
for name, tn, tv, wt, ew, rf, sv, vix, dd, dds, mt in configs:
    if name == best["name"]:
        p, lw = run_portfolio(streams, fred, top_n=tn, target_vol=tv, weighting=wt,
                              eval_window=ew, rebal_freq=rf, stream_vol=sv,
                              vix_scale=vix, dd_control=dd, dd_strength=dds, min_trailing=mt)
        print(f"\nCurrent positions ({best['name']}):")
        for sn, wt in sorted(lw.items(), key=lambda x:-x[1]):
            print(f"  {sn:35s}: {wt*100:.1f}%")
        if p is not None:
            p.to_csv(RESULTS_DIR/"growth_v2_best_returns.csv", header=["return"])
        break

# Compare to V1
v1 = pd.read_csv(RESULTS_DIR/"growth_best_returns.csv", parse_dates=[0])
v1.columns=["Date","return"]; v1=v1.set_index("Date")["return"]
v1m = m(v1, "Growth V1")
print(f"\n{'='*80}")
print(f"V1 vs V2:")
print(f"  V1: Ret={v1m['ann_ret']:+.1f}%  NAV={v1m['final_nav']}x  MDD={v1m['max_dd']}%  SR={v1m['sharpe']}")
print(f"  V2: Ret={best['ann_ret']:+.1f}%  NAV={best['final_nav']}x  MDD={best['max_dd']}%  SR={best['sharpe']}")

with open(RESULTS_DIR/"growth_v2_experiments.json","w") as f:
    json.dump({"experiments":results,"best":best,"v1":v1m},f,indent=2)

print(f"\nSaved {len(results)} experiments")
