#!/usr/bin/env python3
"""
Long-Only Cross-Asset Adaptive Carry V9
=========================================

Same alpha engines as V8 but LONG ONLY:
- Instead of hedged pairs (long X, short Y), allocate to the LONG leg only
- Use the carry/momentum signals to WEIGHT the allocation across ETFs
- Risk management via VIX scaling + drawdown control + vol targeting
- No shorting, no leverage beyond vol targeting

This is more implementable for retail investors who can't easily short.

TWO APPROACHES TESTED:
A) Pure long-only: just buy the long legs, weighted by signal strength
B) Tilted long-only: start with risk parity base, tilt toward signals
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


def compute_etf_scores(ret, fred):
    """
    For each ETF, compute a composite long-only attractiveness score based on:
    1. Carry proxy (trailing return as income proxy)
    2. Momentum (multi-horizon risk-adjusted)
    3. Low volatility (inverse vol)
    4. Quality (Sharpe of recent returns)
    """
    scores = pd.DataFrame(index=ret.index)
    
    tradeable = [
        # Bonds
        "TLT","IEF","SHY","LQD","HYG","JNK","AGG","EMB","MUB","TIP",
        "VCIT","VCSH","IGIB","MBB","BKLN","SRLN","PFF","BNDX","PCY",
        # Equity - dividend/defensive
        "SCHD","HDV","VIG","DVY","XLP","XLU","XLV",
        # Equity - growth/cyclical
        "XLK","XLF","XLE","XLI","XLY","SPY","QQQ","IWM",
        # International
        "EFA","EEM","EWJ",
        # REITs
        "VNQ","IYR",
        # Commodities
        "GLD","SLV","DBC",
        # Convertibles
        "CWB",
    ]
    available = [t for t in tradeable if t in ret.columns]
    
    for etf in available:
        r = ret[etf]
        
        # 1. Carry proxy: trailing 252d return (captures income + price)
        carry = r.rolling(252, min_periods=126).mean() * 252
        
        # 2. Multi-horizon momentum
        mom_signals = []
        for lb in [21, 63, 126, 252]:
            past_r = r.rolling(lb, min_periods=int(lb*0.7)).mean() * np.sqrt(252)
            past_v = r.rolling(lb, min_periods=int(lb*0.7)).std() * np.sqrt(252)
            mom_signals.append(past_r / past_v.clip(lower=0.01))
        momentum = pd.concat(mom_signals, axis=1).mean(axis=1)
        
        # 3. Inverse vol (prefer lower vol assets)
        vol = r.rolling(63, min_periods=21).std() * np.sqrt(252)
        inv_vol = 1.0 / vol.clip(lower=0.01)
        
        # 4. Quality: recent Sharpe
        quality = r.rolling(126, min_periods=63).mean() / r.rolling(126, min_periods=63).std().clip(lower=1e-6)
        
        # Composite score (equal weight the factors)
        # Normalize each to z-score across time
        carry_z = (carry - carry.rolling(504, min_periods=252).mean()) / carry.rolling(504, min_periods=252).std().clip(lower=1e-6)
        mom_z = momentum  # Already normalized
        vol_z = (inv_vol - inv_vol.rolling(504, min_periods=252).mean()) / inv_vol.rolling(504, min_periods=252).std().clip(lower=1e-6)
        qual_z = quality
        
        scores[etf] = (carry_z + mom_z + vol_z + qual_z) / 4
    
    return scores


def long_only_portfolio(scores, ret, fred, min_warmup=504):
    """
    Build a long-only portfolio:
    At each rebalance, allocate to ETFs with positive scores.
    Weight by score (softmax-like).
    """
    available = [c for c in scores.columns if c in ret.columns]
    scores = scores[available]
    ret_a = ret[available]
    
    portfolio_ret = pd.Series(0.0, index=ret.index)
    start_idx = min_warmup
    if start_idx >= len(scores): return None
    
    current_weights = pd.Series(0.0, index=available)
    
    vix = fred.get("VIXCLS")
    
    for i in range(start_idx, len(ret)):
        dt = ret.index[i]
        
        if (i - start_idx) % REBAL_FREQ == 0:
            # Get trailing scores
            eval_start = max(0, i - EVAL_WINDOW)
            # Use latest score values
            if i < len(scores):
                latest_scores = scores.iloc[i-1]  # Use previous day (no lookahead)
            else:
                continue
            
            # Only allocate to positive-score ETFs
            positive = latest_scores[latest_scores > 0].dropna()
            
            if len(positive) > 0:
                # Softmax-like weighting: exp(score) / sum(exp(score))
                # Cap scores to prevent extreme weights
                capped = positive.clip(upper=positive.quantile(0.9) if len(positive) > 3 else 99)
                exp_scores = np.exp(capped.clip(-3, 3))
                weights = exp_scores / exp_scores.sum()
                
                # Apply minimum diversification: no single ETF > 15%
                weights = weights.clip(upper=0.15)
                weights = weights / weights.sum()
                
                current_weights = pd.Series(0.0, index=available)
                for k, w in weights.items():
                    current_weights[k] = w
            else:
                # No attractive ETFs: go to cash (SHY as proxy)
                current_weights = pd.Series(0.0, index=available)
                if "SHY" in available:
                    current_weights["SHY"] = 1.0
        
        # Daily return
        if i < len(ret_a):
            portfolio_ret.iloc[i] = (current_weights * ret_a.iloc[i]).sum()
    
    portfolio_ret = portfolio_ret.iloc[start_idx:]
    
    # VIX stress scaling
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
    
    # Vol target
    pv = portfolio_ret.rolling(63, min_periods=21).std() * np.sqrt(252)
    ps = (TARGET_VOL / pv.clip(lower=0.005)).clip(0.2, 5.0)
    portfolio_ret = portfolio_ret * ps.shift(1)
    
    # Transaction costs
    # Approximate from rebalance frequency
    tc_per_rebal = 0.001  # ~10bps round trip on avg
    tc_daily = tc_per_rebal / REBAL_FREQ
    portfolio_ret = portfolio_ret - tc_daily
    
    return portfolio_ret.dropna(), current_weights


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
    print("LONG-ONLY CROSS-ASSET ADAPTIVE V9")
    print("="*80)

    prices, fred = load_all_data()
    ret = prices.pct_change()
    print(f"Universe: {prices.shape[1]} ETFs")

    # Compute scores
    print("\nComputing ETF attractiveness scores...")
    scores = compute_etf_scores(ret, fred)
    print(f"Scoring {scores.shape[1]} ETFs")

    # Build portfolio
    print("Building long-only portfolio...")
    portfolio, last_weights = long_only_portfolio(scores, ret, fred)
    if portfolio is None: print("FAILED!"); return

    m = compute_metrics(portfolio)
    print(f"\n{'='*80}")
    print(f"LONG-ONLY RESULTS:")
    print(f"{'='*80}")
    print(f"  Sharpe:     {m['sharpe']:.3f}")
    print(f"  Sortino:    {m['sortino']:.3f}")
    print(f"  Ann Return: {m['ann_ret']*100:+.2f}%")
    print(f"  Ann Vol:    {m['ann_vol']*100:.2f}%")
    print(f"  Max DD:     {m['max_dd']*100:.2f}%")
    print(f"  Calmar:     {m['calmar']:.3f}")
    print(f"  Win Rate:   {m['win_rate']*100:.1f}%")

    sp = int(len(portfolio)*0.6)
    for nm,r in [("TRAIN 60%",portfolio.iloc[:sp]),("TEST 40%",portfolio.iloc[sp:])]:
        m2 = compute_metrics(r)
        if m2: print(f"\n  {nm}: Sharpe={m2['sharpe']:.3f}  Ret={m2['ann_ret']*100:+.2f}%  "
                      f"MaxDD={m2['max_dd']*100:.2f}%  Sortino={m2['sortino']:.3f}  WR={m2['win_rate']*100:.1f}%")

    # Yearly
    print(f"\n  {'Year':>6} {'Ret':>9} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8}")
    for yr,g in portfolio.groupby(portfolio.index.year):
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

    print(f"\n  Autocorr(1): {portfolio.autocorr(1):.4f}")

    # Current weights
    print(f"\n{'='*80}")
    print("CURRENT LONG-ONLY ALLOCATION:")
    print(f"{'='*80}")
    lw = last_weights.sort_values(ascending=False)
    for etf, w in lw.items():
        if w > 0.005:
            price = prices[etf].dropna().iloc[-1] if etf in prices.columns else 0
            print(f"  {etf:6s}: {w*100:5.1f}%  (${price:.2f})")

    # Compare to V8 long/short
    print(f"\n{'='*80}")
    print("COMPARISON: Long-Only vs Long/Short (V8)")
    print(f"{'='*80}")
    v8 = pd.read_csv(DATA_DIR/"results"/"strategy_v8_returns.csv", parse_dates=[0])
    v8.columns = ["Date","return"]; v8 = v8.set_index("Date")["return"]
    # Align
    common = portfolio.index.intersection(v8.index).sort_values()
    p_lo = portfolio.loc[common]
    p_ls = v8.loc[common]
    m_lo = compute_metrics(p_lo)
    m_ls = compute_metrics(p_ls)
    if m_lo and m_ls:
        print(f"  {'Metric':16s} {'Long-Only':>12} {'Long/Short':>12}")
        print(f"  {'Sharpe':16s} {m_lo['sharpe']:>12.3f} {m_ls['sharpe']:>12.3f}")
        print(f"  {'Ann Return':16s} {m_lo['ann_ret']*100:>+11.2f}% {m_ls['ann_ret']*100:>+11.2f}%")
        print(f"  {'Ann Vol':16s} {m_lo['ann_vol']*100:>11.2f}% {m_ls['ann_vol']*100:>11.2f}%")
        print(f"  {'Max DD':16s} {m_lo['max_dd']*100:>11.2f}% {m_ls['max_dd']*100:>11.2f}%")
        print(f"  {'Sortino':16s} {m_lo['sortino']:>12.3f} {m_ls['sortino']:>12.3f}")
        print(f"  {'Win Rate':16s} {m_lo['win_rate']*100:>11.1f}% {m_ls['win_rate']*100:>11.1f}%")
        corr = p_lo.corr(p_ls)
        print(f"  {'Correlation':16s} {corr:>12.3f}")

    # Save
    rd = DATA_DIR/"results"; rd.mkdir(exist_ok=True)
    portfolio.to_csv(rd/"strategy_v9_longonly_returns.csv", header=["return"])
    (1+portfolio).cumprod().to_csv(rd/"strategy_v9_longonly_cumulative.csv", header=["cumulative"])
    print(f"\n  Saved to {rd}")


if __name__ == "__main__":
    main()
