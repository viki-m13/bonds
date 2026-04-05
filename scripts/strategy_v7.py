#!/usr/bin/env python3
"""
Adaptive Filtered Cross-Asset Carry Strategy V7
=================================================

Key improvement: ADAPTIVE STREAM SELECTION.
Instead of equal-weighting all streams (V6), use a rolling walk-forward
filter: only include streams that had positive Sharpe in the PRIOR
evaluation window. This prevents negative-alpha streams from diluting
the portfolio while avoiding lookahead bias.

Uses all 9 engines from V6 but with:
1. Rolling 252-day evaluation window (expanding after 504 days)
2. Only include stream if prior-window Sharpe > 0 
3. Weight by prior-window Sharpe (Sharpe-weighted, not equal)
4. Re-evaluate monthly (21 days)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("/home/user/bonds/data")
ETF_DIR = DATA_DIR / "etfs"
FRED_PATH = DATA_DIR / "fred" / "_combined_fred.csv"
TC_BPS = 5
TARGET_VOL = 0.10
EVAL_WINDOW = 252  # 1 year lookback for stream evaluation
REBAL_FREQ = 21    # Monthly rebalance


def load_all_data():
    prices = {}
    for f in sorted(ETF_DIR.glob("*.csv")):
        if f.name.startswith("_"):
            continue
        try:
            df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
            df = df[~df.index.duplicated(keep="first")].sort_index()
            if "Close" in df.columns:
                prices[f.stem] = df["Close"]
        except Exception:
            continue
    prices = pd.DataFrame(prices).sort_index()
    fred = pd.read_csv(FRED_PATH, parse_dates=["Date"]).set_index("Date")
    fred = fred[~fred.index.duplicated(keep="first")].sort_index()
    for c in fred.columns:
        fred[c] = pd.to_numeric(fred[c], errors="coerce")
    fred = fred.ffill()
    return prices, fred


def hedged_pair(ret, long_e, hedge_e, lb=252):
    if long_e not in ret.columns or hedge_e not in ret.columns:
        return None
    cov = ret[long_e].rolling(lb, min_periods=126).cov(ret[hedge_e])
    var = ret[hedge_e].rolling(lb, min_periods=126).var()
    beta = (cov / var.clip(lower=1e-8)).clip(-3, 3)
    return (ret[long_e] - beta.shift(1) * ret[hedge_e]).dropna()


def generate_all_streams(ret, fred):
    """Generate all candidate streams from all engines."""
    streams = {}

    # ENGINE 1: Bond carry
    for l, h in [("HYG","IEF"),("HYG","TLT"),("HYG","SHY"),("JNK","IEF"),
                  ("LQD","IEF"),("VCIT","IEI"),("VCSH","SHY"),("IGIB","IEI"),
                  ("EMB","IEF"),("EMB","TLT"),("MUB","SHY"),("MUB","IEI"),
                  ("MBB","IEF"),("TIP","IEF")]:
        r = hedged_pair(ret, l, h)
        if r is not None and len(r)>=252: streams[f"bc_{l}_{h}"] = r

    # ENGINE 2: Equity sector carry
    for s in ["XLF","XLE","XLU","XLP","XLV","XLI","XLY","XLK","XLB","XLRE","XLC"]:
        r = hedged_pair(ret, s, "SPY")
        if r is not None and len(r)>=252: streams[f"sec_{s}"] = r

    # ENGINE 3: REIT carry
    for reit in ["VNQ","IYR","VNQI","REM"]:
        for h in ["SPY","IEF"]:
            r = hedged_pair(ret, reit, h)
            if r is not None and len(r)>=252: streams[f"reit_{reit}_{h}"] = r

    # ENGINE 4: Commodity momentum
    for comm in ["GLD","SLV","USO","UNG","DBA","DBC","PDBC","CPER"]:
        if comm not in ret.columns: continue
        sigs = []
        for lb in [21,63,126,252]:
            pr = ret[comm].rolling(lb, min_periods=int(lb*0.7)).mean()*np.sqrt(252)
            pv = ret[comm].rolling(lb, min_periods=int(lb*0.7)).std()*np.sqrt(252)
            sigs.append(pr / pv.clip(lower=0.01))
        cs = pd.concat(sigs, axis=1).mean(axis=1)
        pos = cs.clip(-2,2)/2
        sr = pos.shift(1)*ret[comm] - pos.diff().abs()*(10/10000)
        sr = sr.dropna()
        if len(sr)>=252: streams[f"cm_{comm}"] = sr

    # ENGINE 5: Currency carry
    for l,s,n in [("FXA","FXY","aj"),("FXB","FXY","bj"),("CEW","UUP","em"),("FXA","UUP","au")]:
        if l in ret.columns and s in ret.columns:
            r = (ret[l]-ret[s]).dropna()
            if len(r)>=252: streams[f"fx_{n}"] = r

    # ENGINE 6: Cross-asset TSMOM
    for a in ["SPY","QQQ","IWM","EFA","EEM","VNQ","GLD","TLT","LQD","HYG","DBC","UUP","EWJ","EWZ"]:
        if a not in ret.columns: continue
        r12 = ret[a].rolling(252,min_periods=200).mean()*252
        r1 = ret[a].rolling(21,min_periods=15).mean()*252
        sig = r12 - r1
        vol = ret[a].rolling(63,min_periods=21).std()*np.sqrt(252)
        pos = (sig/vol.clip(lower=0.01)).clip(-2,2)/2
        sr = pos.shift(1)*ret[a] - pos.diff().abs()*(TC_BPS/10000)
        sr = sr.dropna()
        if len(sr)>=252: streams[f"xm_{a}"] = sr

    # ENGINE 7: Preferred/loan carry
    for l,h,n in [("PFF","IEF","pi"),("PFF","SHY","ps"),("PGX","IEF","gi"),
                    ("CWB","SPY","cs"),("BKLN","SHY","ls"),("SRLN","SHY","ss")]:
        r = hedged_pair(ret, l, h)
        if r is not None and len(r)>=252: streams[f"pf_{n}"] = r

    # ENGINE 8: Intl bond carry
    for l,h,n in [("BNDX","AGG","ba"),("IGOV","IEF","gi"),("EMLC","IEF","ei"),("PCY","IEF","pi")]:
        r = hedged_pair(ret, l, h)
        if r is not None and len(r)>=252: streams[f"ib_{n}"] = r

    # ENGINE 9: Dividend carry
    for l,h,n in [("DVY","SPY","ds"),("SCHD","SPY","ss"),("HDV","SPY","hs"),("VIG","SPY","vs")]:
        r = hedged_pair(ret, l, h)
        if r is not None and len(r)>=252: streams[f"div_{n}"] = r

    return streams


def adaptive_portfolio(all_streams, fred, min_warmup=504):
    """
    Build portfolio with adaptive stream selection.
    At each rebalance date, evaluate each stream's trailing Sharpe.
    Only include streams with positive trailing Sharpe.
    Weight by trailing Sharpe.
    """
    # Align all streams
    df = pd.DataFrame(all_streams).dropna(how="all")
    df = df.dropna(thresh=5).fillna(0)

    # Vol-target each stream to 3% ann
    vol_t = pd.DataFrame(index=df.index)
    for col in df.columns:
        rv = df[col].rolling(63, min_periods=21).std() * np.sqrt(252)
        sc = (0.03 / rv.clip(lower=0.003)).clip(0.1, 8.0)
        vol_t[col] = df[col] * sc.shift(1)
    vol_t = vol_t.fillna(0)

    # Adaptive selection: compute trailing Sharpe at each rebalance
    portfolio_ret = pd.Series(0.0, index=vol_t.index)
    active_counts = []

    # Start after warmup
    start_idx = min_warmup
    if start_idx >= len(vol_t):
        return None, None

    current_weights = pd.Series(0.0, index=vol_t.columns)

    for i in range(start_idx, len(vol_t)):
        dt = vol_t.index[i]

        # Rebalance every REBAL_FREQ days
        if (i - start_idx) % REBAL_FREQ == 0:
            # Evaluate each stream's trailing performance
            eval_start = max(0, i - EVAL_WINDOW)
            eval_data = vol_t.iloc[eval_start:i]

            trailing_sharpe = {}
            for col in vol_t.columns:
                s = eval_data[col]
                if s.std() > 0 and s.count() >= 63:
                    ts = s.mean() / s.std() * np.sqrt(252)
                    trailing_sharpe[col] = ts

            # Select streams with positive trailing Sharpe
            selected = {k: v for k, v in trailing_sharpe.items() if v > 0}

            if selected:
                # Weight by Sharpe (capped)
                sharpe_vals = pd.Series(selected)
                sharpe_vals = sharpe_vals.clip(upper=sharpe_vals.quantile(0.9))  # Cap outliers
                weights = sharpe_vals / sharpe_vals.sum()
                current_weights = pd.Series(0.0, index=vol_t.columns)
                for k, w in weights.items():
                    current_weights[k] = w
            else:
                current_weights = pd.Series(0.0, index=vol_t.columns)

            active_counts.append(len(selected))

        # Daily return
        portfolio_ret.iloc[i] = (current_weights * vol_t.iloc[i]).sum()

    portfolio_ret = portfolio_ret.iloc[start_idx:]

    # VIX stress scaling
    vix = fred.get("VIXCLS")
    if vix is not None:
        vix_a = vix.reindex(portfolio_ret.index).ffill()
        vix_pctl = vix_a.rolling(252, min_periods=126).rank(pct=True)
        stress = (1.2 - 0.6 * vix_pctl).clip(0.5, 1.2)
        portfolio_ret = portfolio_ret * stress.shift(1)

    # Drawdown control
    cum = (1 + portfolio_ret).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    dd_scale = np.exp(dd * 5).clip(0.2, 1.0)
    portfolio_ret = portfolio_ret * dd_scale.shift(1)

    # Portfolio vol target
    pv = portfolio_ret.rolling(63, min_periods=21).std() * np.sqrt(252)
    ps = (TARGET_VOL / pv.clip(lower=0.005)).clip(0.2, 5.0)
    portfolio_ret = portfolio_ret * ps.shift(1)

    avg_active = np.mean(active_counts) if active_counts else 0
    return portfolio_ret.dropna(), avg_active


def compute_metrics(r):
    r = r.dropna()
    if len(r) < 60: return None
    ar=r.mean()*252; av=r.std()*np.sqrt(252)
    sr=ar/av if av>0 else 0; cum=(1+r).cumprod()
    mdd=((cum-cum.cummax())/cum.cummax()).min()
    cal=ar/abs(mdd) if mdd!=0 else 0; wr=(r>0).mean()
    ds=r[r<0].std()*np.sqrt(252) if (r<0).any() else av
    sortino=ar/ds if ds>0 else 0
    return {"ann_ret":ar,"ann_vol":av,"sharpe":sr,"sortino":sortino,
            "max_dd":mdd,"calmar":cal,"win_rate":wr,
            "skew":r.skew(),"kurt":r.kurtosis(),"n_days":len(r)}


def main():
    print("="*80)
    print("ADAPTIVE FILTERED CROSS-ASSET CARRY V7")
    print("="*80)

    prices, fred = load_all_data()
    ret = prices.pct_change()
    print(f"Universe: {prices.shape[1]} ETFs")

    # Generate all candidate streams
    print("\nGenerating candidate streams...")
    all_streams = generate_all_streams(ret, fred)
    print(f"Total candidates: {len(all_streams)}")

    # Show stream counts by engine
    engine_counts = {}
    for name in all_streams:
        prefix = name.split("_")[0]
        engine_counts[prefix] = engine_counts.get(prefix, 0) + 1
    for eng, cnt in sorted(engine_counts.items()):
        print(f"  {eng}: {cnt} streams")

    # Build adaptive portfolio
    print("\nBuilding adaptive portfolio...")
    portfolio, avg_active = adaptive_portfolio(all_streams, fred)
    if portfolio is None:
        print("FAILED!"); return

    m = compute_metrics(portfolio)
    print(f"\n{'='*80}")
    print(f"RESULTS (avg {avg_active:.0f} active streams)")
    print(f"{'='*80}")
    print(f"  Sharpe:     {m['sharpe']:.3f}")
    print(f"  Ann Return: {m['ann_ret']*100:+.2f}%")
    print(f"  Ann Vol:    {m['ann_vol']*100:.2f}%")
    print(f"  Sortino:    {m['sortino']:.3f}")
    print(f"  Max DD:     {m['max_dd']*100:.2f}%")
    print(f"  Calmar:     {m['calmar']:.3f}")
    print(f"  Win Rate:   {m['win_rate']*100:.1f}%")
    print(f"  Skew:       {m['skew']:.3f}")
    print(f"  Kurt:       {m['kurt']:.3f}")

    # Train/Test
    sp = int(len(portfolio)*0.6)
    for nm, r in [("TRAIN 60%",portfolio.iloc[:sp]),("TEST 40%",portfolio.iloc[sp:])]:
        m2 = compute_metrics(r)
        if m2:
            print(f"\n  {nm}: Sharpe={m2['sharpe']:.3f}  AnnRet={m2['ann_ret']*100:+.2f}%  "
                  f"MaxDD={m2['max_dd']*100:.2f}%  Sortino={m2['sortino']:.3f}  WinRate={m2['win_rate']*100:.1f}%")

    # Yearly
    print(f"\n  {'Year':>6} {'Ret':>9} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8}")
    for yr, g in portfolio.groupby(portfolio.index.year):
        if len(g)<20: continue
        ar=g.mean()*252; av=g.std()*np.sqrt(252); sr=ar/av if av>0 else 0
        c=(1+g).cumprod(); mdd=((c-c.cummax())/c.cummax()).min()
        print(f"  {yr:>6} {ar*100:>+8.2f}% {av*100:>7.2f}% {sr:>+7.3f} {mdd*100:>+7.2f}%")

    # Walk-forward
    print(f"\n  WALK-FORWARD (5 folds):")
    nt=len(portfolio); fs=nt//6; wf=[]
    for fold in range(5):
        s=(fold+1)*fs; e=min(s+fs,nt); fr=portfolio.iloc[s:e]
        fm=compute_metrics(fr)
        if fm:
            wf.append(fm['sharpe'])
            print(f"    Fold {fold+1} ({fr.index[0].date()} to {fr.index[-1].date()}): Sharpe={fm['sharpe']:.3f}")
    if wf: print(f"    Mean: {np.mean(wf):.3f}  Std: {np.std(wf):.3f}")

    # Diagnostics
    print(f"\n  Autocorr(1): {portfolio.autocorr(1):.4f}")
    n_trials = len(all_streams)*2
    dsr = np.sqrt(2*np.log(n_trials))/np.sqrt(m['n_days']/252)
    print(f"  Deflated Sharpe: {m['sharpe']-dsr:.3f} (raw {m['sharpe']:.3f} - {dsr:.3f})")

    # Save
    rd = DATA_DIR/"results"; rd.mkdir(exist_ok=True)
    portfolio.to_csv(rd/"strategy_v7_returns.csv", header=["return"])
    (1+portfolio).cumprod().to_csv(rd/"strategy_v7_cumulative.csv", header=["cumulative"])
    print(f"\n  Saved to {rd}")


if __name__ == "__main__":
    main()
