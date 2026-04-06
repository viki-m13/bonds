#!/usr/bin/env python3
"""
SORTINO STRATEGY: Maximize Sortino Ratio
==========================================
Separate research — does NOT touch Sharpe or Growth pages.

SORTINO = Return / Downside Deviation

Key insight: Sortino penalizes ONLY downside vol, not upside.
So the optimal strategy should:
- Maximize upside variance (let winners run)
- Minimize downside variance (cut losers fast)
- Prefer positively-skewed return distributions
- Use asymmetric risk management (tight stops, loose targets)

DIFFERENCES FROM SHARPE & GROWTH:
- Select streams by trailing SORTINO, not Sharpe or Return
- Weight by Sortino-squared (reward low downside risk)
- Asymmetric drawdown control (aggressive pullback on losses,
  no cap on gains)
- Prefer streams with positive skewness
- VIX scaling more aggressive (downside protection focus)
- Moderate vol target (15%) — between Sharpe (10%) and Growth (30%)
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


def generate_sortino_streams(ret, fred):
    """
    Generate streams from ALL engines (carry, momentum, leveraged, crypto, etc.)
    Same universe as Growth+Sharpe combined.
    """
    streams = {}
    MIN = 252

    # === CARRY (hedged with inverse — good downside protection) ===
    for l,h,hw,n in [
        ("HYG","TBF",0.5,"hyg_tbf"),("JNK","TBF",0.5,"jnk_tbf"),
        ("LQD","TBF",0.6,"lqd_tbf"),("VCIT","TYO",0.3,"vcit_tyo"),
        ("IGIB","TYO",0.3,"igib_tyo"),("EMB","TBF",0.4,"emb_tbf"),
        ("MUB","TBF",0.3,"mub_tbf"),("MBB","TBF",0.6,"mbb_tbf"),
        ("TIP","TBF",0.5,"tip_tbf"),("PFF","TBF",0.4,"pff_tbf"),
        ("SCHD","SH",0.3,"schd_sh"),("HDV","SH",0.3,"hdv_sh"),
        ("VIG","SH",0.3,"vig_sh"),("DVY","SH",0.3,"dvy_sh"),
        ("XLP","SH",0.4,"xlp_sh"),("XLU","SH",0.3,"xlu_sh"),
        ("XLV","SH",0.3,"xlv_sh"),
        ("SCHD","TBF",0.2,"schd_tbf"),("HDV","TBF",0.2,"hdv_tbf"),
        ("DVY","TBF",0.2,"dvy_tbf"),
        ("ANGL","SH",0.3,"angl_sh"),("ANGL","TBF",0.4,"angl_tbf"),
        ("SHYG","SH",0.3,"shyg_sh"),("SHYG","TBF",0.3,"shyg_tbf"),
        ("BOND","TBF",0.3,"bond_tbf"),("TOTL","TBF",0.3,"totl_tbf"),
        ("CWB","SH",0.3,"cwb_sh"),
        ("AMLP","SH",0.3,"amlp_sh"),("IGF","SH",0.3,"igf_sh"),
        ("QQQ","SQQQ",0.15,"qqq_sqqq"),("SPY","SPXU",0.15,"spy_spxu"),
        ("TLT","TMV",0.15,"tlt_tmv"),("GLD","GLL",0.2,"gld_gll"),
        ("VNQ","TBF",0.4,"vnq_tbf"),("IYR","TBF",0.4,"iyr_tbf"),
        ("JPST","TBF",0.2,"jpst_tbf"),("SGOV","TBF",0.3,"sgov_tbf"),
        ("XLK","SH",0.3,"xlk_sh"),("SMH","SH",0.3,"smh_sh"),
        ("EFA","SH",0.3,"efa_sh"),("EEM","SH",0.3,"eem_sh"),
    ]:
        r = hedged(ret, l, h, hw)
        if r is not None and len(r)>=MIN: streams[f"carry_{n}"] = r

    # === UNHEDGED CARRY (higher return, less downside protection) ===
    for etf in ["BKLN","SRLN","JAAA","JBBB","BNDX","PCY","EMLC"]:
        if etf in ret.columns:
            r = ret[etf].dropna()
            if len(r)>=MIN: streams[f"solo_{etf}"] = r

    # === MANAGED FUTURES (positive skew, crisis alpha) ===
    for mf in ["DBMF","CTA","KMLM"]:
        if mf in ret.columns:
            r = ret[mf].dropna()
            if len(r)>=MIN: streams[f"mfut_{mf}"] = r

    # === MOMENTUM (long-only, all assets) ===
    for etf in ["SPY","QQQ","IWM","EFA","EEM","VNQ","GLD","SLV","TLT",
                "HYG","EWJ","SMH","XBI","AMLP","KWEB","TAN","URA",
                "IBIT","GBTC","BITO","BTC_USD","ETH_USD","SOL_USD","ADA_USD",
                "TQQQ","UPRO","SOXL","TECL"]:
        if etf not in ret.columns: continue
        sigs = []
        for lb in [21,63,126,252]:
            pr = ret[etf].rolling(lb,min_periods=int(lb*0.7)).mean()*np.sqrt(252)
            pv = ret[etf].rolling(lb,min_periods=int(lb*0.7)).std()*np.sqrt(252)
            sigs.append(pr/pv.clip(lower=0.01))
        cs = pd.concat(sigs,axis=1).mean(axis=1)
        pos = cs.clip(0,2)/2
        tc = 15 if etf in ["BTC_USD","ETH_USD","SOL_USD","ADA_USD"] else (10 if etf in ["TQQQ","UPRO","SOXL","TECL"] else 5)
        sr = pos.shift(1)*ret[etf] - pos.diff().abs()*(tc/10000)
        if len(sr.dropna())>=MIN: streams[f"mom_{etf}"] = sr.dropna()

    # === UNHEDGED DIVIDEND / EQUITY (for growth component) ===
    for etf in ["SCHD","HDV","DVY","VIG","XLP","XLU"]:
        if etf in ret.columns:
            r = ret[etf].dropna()
            if len(r)>=MIN: streams[f"div_{etf}"] = r

    return streams


def compute_sortino(r, min_periods=63):
    """Compute annualized Sortino ratio."""
    if len(r) < min_periods or r.std() == 0: return 0
    ar = r.mean() * 252
    downside = r[r < 0]
    if len(downside) < 10: return r.mean()/r.std()*np.sqrt(252)  # fallback to Sharpe
    dd = downside.std() * np.sqrt(252)
    return ar / dd if dd > 0 else 0


def compute_skewness(r, min_periods=63):
    """Compute skewness of returns."""
    if len(r) < min_periods: return 0
    return r.skew()


def run_sortino_portfolio(streams, fred, top_n=5, target_vol=0.15,
                           weighting="sortino_sq", min_warmup=504,
                           dd_strength=5, skew_bonus=True,
                           min_sortino=0.0, vix_aggression=0.8):
    """
    Portfolio construction optimized for Sortino:
    - Select by trailing Sortino (not Sharpe or Return)
    - Weight by Sortino-squared
    - Optional skewness bonus (prefer positively-skewed streams)
    - Asymmetric VIX scaling (more aggressive protection)
    """
    df = pd.DataFrame(streams).dropna(how="all").dropna(thresh=5).fillna(0)
    vol_t = pd.DataFrame(index=df.index)
    for col in df.columns:
        rv = df[col].rolling(63,min_periods=21).std()*np.sqrt(252)
        sc = (0.04/rv.clip(lower=0.003)).clip(0.1,8.0)
        vol_t[col] = df[col]*sc.shift(1)
    vol_t = vol_t.fillna(0)

    portfolio_ret = pd.Series(0.0, index=vol_t.index)
    start_idx = min_warmup
    if start_idx >= len(vol_t): return None, {}
    cw = pd.Series(0.0, index=vol_t.columns)
    last_selected = {}

    for i in range(start_idx, len(vol_t)):
        if (i-start_idx) % REBAL_FREQ == 0:
            eval_data = vol_t.iloc[max(0,i-EVAL_WINDOW):i]

            scores = {}
            for col in vol_t.columns:
                s = eval_data[col]
                if s.count() < 63: continue

                sortino = compute_sortino(s)
                if sortino <= min_sortino: continue

                score = sortino
                if skew_bonus:
                    skew = compute_skewness(s)
                    # Bonus for positive skew (up to 30% boost)
                    skew_mult = 1.0 + min(0.3, max(-0.15, skew * 0.1))
                    score *= skew_mult

                scores[col] = score

            if scores:
                sv = pd.Series(scores).nlargest(top_n)
                sv = sv.clip(upper=sv.quantile(0.9) if len(sv)>3 else 99)

                if weighting == "sortino_sq":
                    sq = sv**2
                    w = sq/sq.sum()
                elif weighting == "sortino":
                    w = sv/sv.sum()
                elif weighting == "equal":
                    w = pd.Series(1.0/len(sv), index=sv.index)
                else:
                    w = sv/sv.sum()

                cw = pd.Series(0.0, index=vol_t.columns)
                for k,v in w.items(): cw[k] = v
                last_selected = dict(zip(w.index, w.values))
            else:
                cw = pd.Series(0.0, index=vol_t.columns)

        portfolio_ret.iloc[i] = (cw*vol_t.iloc[i]).sum()

    p = portfolio_ret.iloc[start_idx:]

    # Asymmetric VIX scaling (more aggressive than Sharpe, same as Growth)
    vix = fred.get("VIXCLS")
    if vix is not None:
        va = vix.reindex(p.index).ffill()
        vp = va.rolling(252,min_periods=126).rank(pct=True)
        stress = (1.2 - vix_aggression*vp).clip(0.4, 1.2)
        p = p * stress.shift(1)

    # Asymmetric drawdown control — more aggressive than Sharpe
    cum = (1+p).cumprod()
    dd = (cum-cum.cummax())/cum.cummax()
    dd_scale = np.exp(dd*dd_strength).clip(0.2,1.0)
    p = p*dd_scale.shift(1)

    # Vol target
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
    skew = r.skew()
    sp = int(len(r)*0.6)
    test_sr = r.iloc[sp:].mean()/r.iloc[sp:].std()*np.sqrt(252) if r.iloc[sp:].std()>0 else 0
    test_sortino = 0
    test_r = r.iloc[sp:]
    test_ds = test_r[test_r<0].std()*np.sqrt(252) if (test_r<0).any() else test_r.std()*np.sqrt(252)
    if test_ds > 0: test_sortino = test_r.mean()*252/test_ds
    nt=len(r); fs=nt//6; wf_sortino=[]
    for fold in range(5):
        s=(fold+1)*fs; e=min(s+fs,nt); fr=r.iloc[s:e]
        if len(fr)>60:
            fds = fr[fr<0].std()*np.sqrt(252) if (fr<0).any() else fr.std()*np.sqrt(252)
            if fds > 0: wf_sortino.append(fr.mean()*252/fds)
    return {"name":name,"sharpe":round(sr,3),"sortino":round(sortino,3),
            "ann_ret":round(ar*100,2),"ann_vol":round(av*100,2),
            "max_dd":round(mdd*100,2),"calmar":round(cal,3),"win_rate":round(wr*100,1),
            "skew":round(skew,3),"total_ret":round(total*100,2),
            "test_sharpe":round(test_sr,3),"test_sortino":round(test_sortino,3),
            "wf_sortino_mean":round(np.mean(wf_sortino),3) if wf_sortino else 0,
            "ac1":round(r.autocorr(1),4),"n":len(r),
            "final_nav":round(float(cum.iloc[-1]),2)}


def main():
    print("="*80)
    print("SORTINO STRATEGY: MAXIMIZE SORTINO RATIO")
    print("="*80)

    prices, fred = load_all_data()
    ret = prices.pct_change()
    print(f"Universe: {len(prices.columns)} ETFs")

    streams = generate_sortino_streams(ret, fred)
    print(f"Sortino streams: {len(streams)}")

    eng = {}
    for n in streams:
        p = n.split("_")[0]
        eng[p] = eng.get(p,0)+1
    for e,c in sorted(eng.items()): print(f"  {e}: {c}")

    # Top 20 by Sortino
    print(f"\n--- Top 20 Streams by Sortino ---")
    sm = {}
    for name, s in streams.items():
        mx = m(s, name)
        if mx: sm[name] = mx
    for name, mx in sorted(sm.items(), key=lambda x: -x[1]["sortino"])[:20]:
        print(f"  {name:30s}: Sortino={mx['sortino']:.3f}  SR={mx['sharpe']:.3f}  "
              f"Ret={mx['ann_ret']:+.1f}%  Skew={mx['skew']:+.3f}  MDD={mx['max_dd']:.1f}%")

    # ================================================================
    # EXPERIMENT GRID
    # ================================================================
    results = []

    configs = [
        # (name, top_n, target_vol, weighting, dd_strength, skew_bonus, min_sortino, vix_agg)
        ("Sort_Top3_15v", 3, 0.15, "sortino_sq", 5, True, 0.0, 0.8),
        ("Sort_Top5_15v", 5, 0.15, "sortino_sq", 5, True, 0.0, 0.8),
        ("Sort_Top5_10v", 5, 0.10, "sortino_sq", 5, True, 0.0, 0.8),
        ("Sort_Top5_20v", 5, 0.20, "sortino_sq", 5, True, 0.0, 0.8),
        ("Sort_Top3_20v", 3, 0.20, "sortino_sq", 5, True, 0.0, 0.8),
        # Without skew bonus
        ("Sort_Top5_NoSkew", 5, 0.15, "sortino_sq", 5, False, 0.0, 0.8),
        # Linear sortino weighting
        ("Sort_Top5_Lin", 5, 0.15, "sortino", 5, True, 0.0, 0.8),
        # Equal weight
        ("Sort_Top5_Eq", 5, 0.15, "equal", 5, True, 0.0, 0.8),
        # Min sortino thresholds
        ("Sort_Top5_Min0.5", 5, 0.15, "sortino_sq", 5, True, 0.5, 0.8),
        ("Sort_Top5_Min1.0", 5, 0.15, "sortino_sq", 5, True, 1.0, 0.8),
        ("Sort_Top5_Min1.5", 5, 0.15, "sortino_sq", 5, True, 1.5, 0.8),
        # Stronger drawdown control
        ("Sort_Top5_DD8", 5, 0.15, "sortino_sq", 8, True, 0.0, 0.8),
        ("Sort_Top5_DD10", 5, 0.15, "sortino_sq", 10, True, 0.0, 0.8),
        # More aggressive VIX
        ("Sort_Top5_VIX1.0", 5, 0.15, "sortino_sq", 5, True, 0.0, 1.0),
        ("Sort_Top5_VIX0.6", 5, 0.15, "sortino_sq", 5, True, 0.0, 0.6),
        # Combined best
        ("Sort_Best_v1", 5, 0.15, "sortino_sq", 5, True, 0.75, 0.8),
        ("Sort_Best_v2", 4, 0.15, "sortino_sq", 7, True, 0.75, 0.8),
        ("Sort_Best_v3", 3, 0.15, "sortino_sq", 7, True, 1.0, 0.8),
        ("Sort_Best_v4", 5, 0.12, "sortino_sq", 8, True, 0.75, 1.0),
        ("Sort_Aggressive", 3, 0.20, "sortino_sq", 3, True, 0.5, 0.6),
    ]

    for name, tn, tv, wt, dd, sb, ms, va in configs:
        p, lw = run_sortino_portfolio(streams, fred, top_n=tn, target_vol=tv,
                                       weighting=wt, dd_strength=dd,
                                       skew_bonus=sb, min_sortino=ms,
                                       vix_aggression=va)
        if p is not None:
            mx = m(p, name)
            results.append(mx)

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS RANKED BY SORTINO RATIO")
    print(f"{'='*80}")
    print(f"{'Name':22s} {'Sortino':>8} {'Sharpe':>7} {'Ret':>8} {'Vol':>7} {'MDD':>7} {'Skew':>6} {'TestSort':>9} {'WF Sort':>8} {'NAV':>7}")
    print("-"*95)
    for r in sorted(results, key=lambda x: -x["sortino"]):
        print(f"  {r['name']:20s} {r['sortino']:>7.3f} {r['sharpe']:>6.3f} {r['ann_ret']:>+7.1f}% "
              f"{r['ann_vol']:>6.1f}% {r['max_dd']:>+6.1f}% {r['skew']:>+5.2f} "
              f"{r['test_sortino']:>8.3f} {r['wf_sortino_mean']:>7.3f} {r['final_nav']:>6.1f}x")

    # Best by Sortino
    best = max(results, key=lambda x: x["sortino"])

    # ================================================================
    # COMPARE ALL THREE STRATEGIES
    # ================================================================
    print(f"\n{'='*80}")
    print("COMPARISON: SHARPE vs GROWTH vs SORTINO")
    print(f"{'='*80}")

    # Load Sharpe and Growth
    dichs = pd.read_csv(DATA_DIR/"results"/"dichs_returns.csv", parse_dates=[0])
    dichs.columns=["Date","return"]; dichs=dichs.set_index("Date")["return"]
    dm = m(dichs, "Sharpe (DICHS)")

    growth = pd.read_csv(Path(__file__).parent.parent/"growth"/"results"/"growth_best_returns.csv", parse_dates=[0])
    growth.columns=["Date","return"]; growth=growth.set_index("Date")["return"]
    gm = m(growth, "Growth")

    print(f"\n  {'Metric':25s} {'Sharpe Strat':>14} {'Growth Strat':>14} {'Sortino Strat':>14}")
    print(f"  {'-'*70}")
    for label, dk, gk, sk in [
        ("Sharpe Ratio", dm['sharpe'], gm['sharpe'], best['sharpe']),
        ("Sortino Ratio", dm['sortino'], gm['sortino'], best['sortino']),
        ("Ann. Return", f"{dm['ann_ret']:+.1f}%", f"{gm['ann_ret']:+.1f}%", f"{best['ann_ret']:+.1f}%"),
        ("Ann. Volatility", f"{dm['ann_vol']:.1f}%", f"{gm['ann_vol']:.1f}%", f"{best['ann_vol']:.1f}%"),
        ("Max Drawdown", f"{dm['max_dd']:.1f}%", f"{gm['max_dd']:.1f}%", f"{best['max_dd']:.1f}%"),
        ("Calmar Ratio", dm['calmar'], gm['calmar'], best['calmar']),
        ("Skewness", dm['skew'], gm['skew'], best['skew']),
        ("Win Rate", f"{dm['win_rate']:.1f}%", f"{gm['win_rate']:.1f}%", f"{best['win_rate']:.1f}%"),
        ("Final NAV", f"{dm['final_nav']:.1f}x", f"{gm['final_nav']:.1f}x", f"{best['final_nav']:.1f}x"),
        ("Test Sortino", dm['test_sortino'], gm['test_sortino'], best['test_sortino']),
        ("WF Sortino Mean", dm['wf_sortino_mean'], gm['wf_sortino_mean'], best['wf_sortino_mean']),
    ]:
        print(f"  {label:25s} {str(dk):>14} {str(gk):>14} {str(sk):>14}")

    # Best config details
    print(f"\n  Best Sortino config: {best['name']}")
    print(f"  Autocorrelation: {best['ac1']}")

    # Run the best config to get current positions
    best_name = best["name"]
    # Parse config from name
    for name, tn, tv, wt, dd, sb, ms, va in configs:
        if name == best_name:
            p_best, lw_best = run_sortino_portfolio(
                streams, fred, top_n=tn, target_vol=tv, weighting=wt,
                dd_strength=dd, skew_bonus=sb, min_sortino=ms, vix_aggression=va)
            if p_best is not None:
                print(f"\n  Current Top {tn} Positions:")
                for sname, wt in sorted(lw_best.items(), key=lambda x:-x[1]):
                    print(f"    {sname:35s}: {wt*100:.1f}%")
                p_best.to_csv(RESULTS_DIR/"sortino_best_returns.csv", header=["return"])
                (1+p_best).cumprod().to_csv(RESULTS_DIR/"sortino_best_cumulative.csv", header=["cumulative"])
            break

    # Yearly breakdown
    if p_best is not None:
        print(f"\n  Yearly:")
        print(f"  {'Year':>6} {'Sortino Strat':>14} {'Sharpe Strat':>14} {'SPY':>10}")
        spy = pd.read_csv(DATA_DIR/"etfs"/"SPY.csv", parse_dates=["Date"]).set_index("Date")["Close"].pct_change()
        common = p_best.index.intersection(dichs.index).intersection(spy.index)
        for yr in sorted(set(common.year)):
            mask = common.year == yr
            if mask.sum() < 20: continue
            sort_r = ((1+p_best.loc[common[mask]]).prod()-1)*100
            dichs_r = ((1+dichs.loc[common[mask]]).prod()-1)*100
            spy_r = ((1+spy.loc[common[mask]]).prod()-1)*100
            print(f"  {yr:>6} {sort_r:>+13.1f}% {dichs_r:>+13.1f}% {spy_r:>+9.1f}%")

    # Save
    with open(RESULTS_DIR/"sortino_experiments.json", "w") as f:
        json.dump({"experiments": results, "best": best,
                   "sharpe_comparison": dm, "growth_comparison": gm}, f, indent=2)

    print(f"\nSaved {len(results)} experiments to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
