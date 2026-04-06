#!/usr/bin/env python3
"""
V11: Long-Only Multi-Asset Strategy Using Leveraged & Inverse ETFs
====================================================================

KEY INSIGHT: Inverse ETFs let you go "short" while only buying.
  - Want short S&P? Buy SH (1x inverse) or SDS (2x inverse)
  - Want short bonds? Buy TBF (1x inv) or TBT (2x inv)
  - Want vol decay? Buy the INVERSE of the leveraged ETF

ENGINES (ALL LONG-ONLY):

1. CARRY: Long high-yield ETFs (HYG, SCHD, DVY, PFF, BKLN, etc.)
2. DURATION HEDGE: Long inverse treasury ETFs (TBF, TBT, TYO, TMV) 
   instead of shorting TLT/IEF
3. VOL DECAY HARVEST: Long inverse leveraged ETFs to capture decay
   - Long SQQQ benefits from TQQQ vol decay (when combined with QQQ)
   - Long TMV benefits from TMF vol decay
4. EQUITY HEDGE: Long inverse equity ETFs (SH, PSQ, SDS) for hedging
5. DEFENSIVE ROTATION: Long defensive ETFs (XLP, XLU, GLD, SHY)
   when VIX is elevated
6. MOMENTUM: Long trending assets across all classes

PORTFOLIO: Adaptive Sharpe-weighted selection, vol-targeted, 
drawdown-controlled. 100% long-only, no shorting required.
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
TOP_N = 5
MIN_TRAILING_SHARPE = 0.75  # Only include streams with strong trailing performance
WEIGHTING = "sharpe_sq"     # Sharpe-squared weighting for higher conviction


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


def hedged_long_only(ret, long_etf, inv_hedge_etf, hedge_weight=1.0, lb=252):
    """
    Long-only hedged pair: Buy both the carry asset AND an inverse ETF.
    E.g., Buy HYG + Buy TBF = rate-hedged high yield carry.
    hedge_weight is the fraction allocated to the hedge leg.
    """
    if long_etf not in ret.columns or inv_hedge_etf not in ret.columns:
        return None
    # Dynamic hedge ratio: how much inverse ETF to buy
    # Use rolling beta of long asset to the UNDERLYING of the inverse
    # For simplicity, use fixed weight and let adaptive filter handle sizing
    carry_ret = ret[long_etf]
    hedge_ret = ret[inv_hedge_etf]
    # Combined return: w1 * carry + w2 * inverse_hedge
    w1 = 1.0 / (1.0 + hedge_weight)
    w2 = hedge_weight / (1.0 + hedge_weight)
    combined = w1 * carry_ret + w2 * hedge_ret
    return combined.dropna()


def generate_all_streams(ret, fred):
    streams = {}
    MIN = 252

    # ===== ENGINE 1: CARRY + INVERSE HEDGE (replaces short hedging) =====
    # Long credit + Long inverse treasury = rate-hedged carry
    for carry, inv_hedge, hw, name in [
        # High yield + inverse treasury (replaces: long HYG, short IEF)
        ("HYG", "TBF", 0.5, "hyg_tbf"),
        ("HYG", "TBT", 0.25, "hyg_tbt"),  # TBT is 2x so use half weight
        ("JNK", "TBF", 0.5, "jnk_tbf"),
        ("JNK", "TBT", 0.25, "jnk_tbt"),
        # IG credit + inverse treasury
        ("LQD", "TBF", 0.6, "lqd_tbf"),
        ("LQD", "TBT", 0.3, "lqd_tbt"),
        ("VCIT", "TYO", 0.3, "vcit_tyo"),  # TYO is 3x inv 7-10Y
        ("IGIB", "TYO", 0.3, "igib_tyo"),
        # EM bonds + inverse treasury
        ("EMB", "TBF", 0.4, "emb_tbf"),
        ("EMB", "TBT", 0.2, "emb_tbt"),
        # Muni + inverse short treasury
        ("MUB", "TBF", 0.3, "mub_tbf"),
        # MBS + inverse treasury
        ("MBB", "TBF", 0.6, "mbb_tbf"),
        # TIPS + inverse nominal treasury (real rate play)
        ("TIP", "TBF", 0.5, "tip_tbf"),
        # Preferred + inverse treasury
        ("PFF", "TBF", 0.4, "pff_tbf"),
        # Bank loans (floating rate, natural rate hedge)
        ("BKLN", "SHY", 0.0, "bkln_solo"),  # BKLN barely needs hedging
        ("SRLN", "SHY", 0.0, "srln_solo"),
    ]:
        if hw == 0.0:
            # Solo long
            if carry in ret.columns:
                r = ret[carry].dropna()
                if len(r) >= MIN: streams[f"carry_{name}"] = r
        else:
            r = hedged_long_only(ret, carry, inv_hedge, hw)
            if r is not None and len(r) >= MIN: streams[f"carry_{name}"] = r

    # ===== ENGINE 2: DIVIDEND EQUITY + INVERSE EQUITY HEDGE =====
    # Long dividend ETFs + Long inverse S&P (replaces: long SCHD, short SPY)
    for div_etf, inv_eq, hw, name in [
        ("SCHD", "SH", 0.3, "schd_sh"),    # SH = 1x inverse SPY
        ("HDV", "SH", 0.3, "hdv_sh"),
        ("VIG", "SH", 0.3, "vig_sh"),
        ("DVY", "SH", 0.3, "dvy_sh"),
        ("XLP", "SH", 0.4, "xlp_sh"),
        ("XLU", "SH", 0.3, "xlu_sh"),
        ("XLV", "SH", 0.3, "xlv_sh"),
        # Also try with inverse bonds for rate hedging
        ("SCHD", "TBF", 0.2, "schd_tbf"),
        ("HDV", "TBF", 0.2, "hdv_tbf"),
        ("DVY", "TBF", 0.2, "dvy_tbf"),
    ]:
        r = hedged_long_only(ret, div_etf, inv_eq, hw)
        if r is not None and len(r) >= MIN: streams[f"diveq_{name}"] = r

    # ===== ENGINE 3: VOL DECAY VIA INVERSE LEVERAGED (long-only!) =====
    # Buy the inverse of a leveraged ETF to capture its decay
    # E.g., buy SQQQ to profit as TQQQ decays (but SQQQ also decays...)
    # Better approach: buy underlying + buy inverse leveraged
    # QQQ + SQQQ: when QQQ flat, SQQQ decays. When QQQ up, QQQ profits > SQQQ loss.
    for underlying, inv_lev, ratio, name in [
        ("QQQ", "SQQQ", 0.15, "qqq_sqqq"),   # Small SQQQ position as vol hedge
        ("SPY", "SPXU", 0.15, "spy_spxu"),
        ("TLT", "TMV", 0.15, "tlt_tmv"),
        ("GLD", "GLL", 0.2, "gld_gll"),
    ]:
        r = hedged_long_only(ret, underlying, inv_lev, ratio)
        if r is not None and len(r) >= MIN: streams[f"vdlo_{name}"] = r

    # ===== ENGINE 3b: MLP / INFRASTRUCTURE CARRY =====
    for mlp, inv, hw, name in [
        ("AMLP", "TBF", 0.3, "amlp_tbf"),
        ("AMLP", "SH", 0.3, "amlp_sh"),
        ("IGF", "SH", 0.3, "igf_sh"),
    ]:
        r = hedged_long_only(ret, mlp, inv, hw)
        if r is not None and len(r) >= MIN: streams[f"mlp_{name}"] = r

    # ===== ENGINE 3c: CLO / STRUCTURED CREDIT CARRY =====
    for clo, inv, hw, name in [
        ("JAAA", "SHY", 0.0, "jaaa_solo"),
        ("JBBB", "TBF", 0.3, "jbbb_tbf"),
    ]:
        if hw == 0.0:
            if clo in ret.columns:
                r = ret[clo].dropna()
                if len(r) >= MIN: streams[f"clo_{name}"] = r
        else:
            r = hedged_long_only(ret, clo, inv, hw)
            if r is not None and len(r) >= MIN: streams[f"clo_{name}"] = r

    # ===== ENGINE 3d: FALLEN ANGEL / SHORT HY CARRY =====
    for fa, inv, hw, name in [
        ("ANGL", "TBF", 0.4, "angl_tbf"),
        ("SHYG", "TBF", 0.3, "shyg_tbf"),
    ]:
        r = hedged_long_only(ret, fa, inv, hw)
        if r is not None and len(r) >= MIN: streams[f"fa_{name}"] = r

    # ===== ENGINE 3e: MANAGED FUTURES (trend-following, uncorrelated) =====
    for mf in ["DBMF", "CTA", "KMLM"]:
        if mf in ret.columns:
            r = ret[mf].dropna()
            if len(r) >= MIN: streams[f"mfut_{mf}"] = r

    # ===== ENGINE 3f: NEW CARRY STREAMS (from experiments) =====
    # Fallen angel + short HY carry (hedged with inverse)
    for l,h,hw,n in [
        ("ANGL","SH",0.3,"angl_sh"),("ANGL","TBF",0.4,"angl_tbf"),
        ("SHYG","SH",0.3,"shyg_sh"),("SHYG","TBF",0.3,"shyg_tbf"),
        # Active bond funds hedged
        ("BOND","TBF",0.3,"bond_tbf"),("TOTL","TBF",0.3,"totl_tbf"),
        # Ultra-short carry
        ("JPST","TBF",0.2,"jpst_tbf"),("SGOV","TBF",0.3,"sgov_tbf"),
        # Convertible hedged with equity inverse
        ("CWB","SH",0.3,"cwb_sh"),
        # More sector equity carry
        ("XLK","SH",0.3,"xlk_sh"),("XLI","SH",0.3,"xli_sh"),
        ("XLE","SH",0.3,"xle_sh"),("XLY","SH",0.3,"xly_sh"),
        # International hedged
        ("EFA","SH",0.3,"efa_sh"),("EEM","SH",0.3,"eem_sh"),
        # Homebuilder carry
        ("SMH","SH",0.3,"smh_sh"),("ITB","SH",0.3,"itb_sh"),
    ]:
        r = hedged_long_only(ret, l, h, hw)
        if r is not None and len(r) >= MIN: streams[f"newcarry_{n}"] = r

    # ===== ENGINE 4: PURE INVERSE ETF MOMENTUM =====
    # When markets trend down, inverse ETFs trend up
    # Use momentum signals to time when to hold inverse ETFs
    for inv_etf, name in [("SH","sh"),("PSQ","psq"),("TBF","tbf"),("TBT","tbt"),
                           ("SDS","sds"),("SQQQ","sqqq"),("GLL","gll")]:
        if inv_etf not in ret.columns: continue
        sigs = []
        for lb in [21,63,126]:
            pr = ret[inv_etf].rolling(lb,min_periods=int(lb*0.7)).mean()*np.sqrt(252)
            pv = ret[inv_etf].rolling(lb,min_periods=int(lb*0.7)).std()*np.sqrt(252)
            sigs.append(pr/pv.clip(lower=0.01))
        cs = pd.concat(sigs,axis=1).mean(axis=1)
        # Only go long when signal positive (never short the inverse)
        pos = cs.clip(0, 2) / 2
        sr = pos.shift(1)*ret[inv_etf] - pos.diff().abs()*(TC_BPS/10000)
        if len(sr.dropna())>=MIN: streams[f"invmom_{name}"] = sr.dropna()

    # ===== ENGINE 5: COMMODITY CARRY & MOMENTUM (long-only) =====
    for cm in ["GLD","SLV","DBC","PDBC","URA","PPLT","GSG","JO","SOYB","GLTR"]:
        if cm not in ret.columns: continue
        # Momentum
        sigs = []
        for lb in [21,63,126,252]:
            pr = ret[cm].rolling(lb,min_periods=int(lb*0.7)).mean()*np.sqrt(252)
            pv = ret[cm].rolling(lb,min_periods=int(lb*0.7)).std()*np.sqrt(252)
            sigs.append(pr/pv.clip(lower=0.01))
        cs = pd.concat(sigs,axis=1).mean(axis=1)
        pos = cs.clip(0,2)/2  # Long-only: no shorting
        sr = pos.shift(1)*ret[cm] - pos.diff().abs()*(TC_BPS/10000)
        if len(sr.dropna())>=MIN: streams[f"commom_{cm}"] = sr.dropna()

    # ===== ENGINE 6: CROSS-ASSET LONG MOMENTUM =====
    for a in ["SPY","QQQ","IWM","EFA","EEM","VNQ","GLD","TLT","HYG","EWJ",
              "GBTC","BTC_USD","ETH_USD","ETHE","SOL_USD","ADA_USD",
              "SMH","XBI","XOP","KRE","URA","KWEB","AMLP","DBMF","ARKK"]:
        if a not in ret.columns: continue
        sigs = []
        for lb in [21,63,126,252]:
            pr = ret[a].rolling(lb,min_periods=int(lb*0.7)).mean()*np.sqrt(252)
            pv = ret[a].rolling(lb,min_periods=int(lb*0.7)).std()*np.sqrt(252)
            sigs.append(pr/pv.clip(lower=0.01))
        cs = pd.concat(sigs,axis=1).mean(axis=1)
        pos = cs.clip(0,2)/2  # LONG ONLY
        sr = pos.shift(1)*ret[a] - pos.diff().abs()*(TC_BPS/10000)
        if len(sr.dropna())>=MIN: streams[f"longmom_{a}"] = sr.dropna()

    # ===== ENGINE 7: REIT + INVERSE RATE HEDGE =====
    for reit in ["VNQ","IYR"]:
        for inv, hw, n in [("TBF",0.4,"tbf"),("TBT",0.2,"tbt")]:
            r = hedged_long_only(ret, reit, inv, hw)
            if r is not None and len(r)>=MIN: streams[f"reit_{reit}_{n}"] = r

    # ===== ENGINE 8: INTL BOND CARRY =====
    for intl in ["BNDX","PCY","EMLC"]:
        if intl in ret.columns:
            r = ret[intl].dropna()
            if len(r)>=MIN: streams[f"intlbd_{intl}"] = r

    # ===== ENGINE 9: CURRENCY (long-only FX ETFs) =====
    for fx in ["FXA","FXB","CEW","FXE","UUP","DBV","UDN","FXF","FXS","FXC"]:
        if fx not in ret.columns: continue
        sigs = []
        for lb in [21,63,126]:
            pr = ret[fx].rolling(lb,min_periods=int(lb*0.7)).mean()*np.sqrt(252)
            pv = ret[fx].rolling(lb,min_periods=int(lb*0.7)).std()*np.sqrt(252)
            sigs.append(pr/pv.clip(lower=0.01))
        cs = pd.concat(sigs,axis=1).mean(axis=1)
        pos = cs.clip(0,2)/2
        sr = pos.shift(1)*ret[fx] - pos.diff().abs()*(TC_BPS/10000)
        if len(sr.dropna())>=MIN: streams[f"fxmom_{fx}"] = sr.dropna()

    # ===== ENGINE 10: BITCOIN (long-only momentum + carry) =====
    # Use BTC_USD for longest history, GBTC/IBIT/BITO as tradeable proxies
    for btc in ["BTC_USD","GBTC","IBIT","BITO","ETH_USD","ETHE","ETHA"]:
        if btc not in ret.columns: continue
        # Momentum (multi-horizon, long-only)
        sigs = []
        for lb in [21,63,126,252]:
            pr = ret[btc].rolling(lb,min_periods=int(lb*0.7)).mean()*np.sqrt(252)
            pv = ret[btc].rolling(lb,min_periods=int(lb*0.7)).std()*np.sqrt(252)
            sigs.append(pr/pv.clip(lower=0.01))
        cs = pd.concat(sigs,axis=1).mean(axis=1)
        pos = cs.clip(0,2)/2  # Long-only
        sr = pos.shift(1)*ret[btc] - pos.diff().abs()*(10/10000)  # 10bps crypto costs
        if len(sr.dropna())>=MIN: streams[f"btcmom_{btc}"] = sr.dropna()

    return streams


def adaptive_portfolio(all_streams, fred, min_warmup=504):
    df = pd.DataFrame(all_streams).dropna(how="all").dropna(thresh=5).fillna(0)
    # Vol-target each to 3%
    vol_t = pd.DataFrame(index=df.index)
    for col in df.columns:
        rv = df[col].rolling(63,min_periods=21).std()*np.sqrt(252)
        sc = (0.03/rv.clip(lower=0.003)).clip(0.1,8.0)
        vol_t[col] = df[col]*sc.shift(1)
    vol_t = vol_t.fillna(0)

    portfolio_ret = pd.Series(0.0, index=vol_t.index)
    start_idx = min_warmup
    if start_idx >= len(vol_t): return None, 0, None
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
            selected = {k:v for k,v in trailing_sharpe.items() if v>MIN_TRAILING_SHARPE}
            if selected:
                sv = pd.Series(selected).nlargest(TOP_N)
                sv = sv.clip(upper=sv.quantile(0.9) if len(sv)>3 else 99)
                # Sharpe-squared weighting for higher conviction
                if WEIGHTING == "sharpe_sq":
                    sq = sv**2
                    weights = sq/sq.sum()
                else:
                    weights = sv/sv.sum()
                current_weights = pd.Series(0.0, index=vol_t.columns)
                for k,w in weights.items(): current_weights[k] = w
            else:
                current_weights = pd.Series(0.0, index=vol_t.columns)
            active_counts.append(len(selected))
        portfolio_ret.iloc[i] = (current_weights*vol_t.iloc[i]).sum()

    portfolio_ret = portfolio_ret.iloc[start_idx:]

    # VIX scaling
    vix = fred.get("VIXCLS")
    if vix is not None:
        vix_a = vix.reindex(portfolio_ret.index).ffill()
        vix_pctl = vix_a.rolling(252,min_periods=126).rank(pct=True)
        stress = (1.2-0.6*vix_pctl).clip(0.5,1.2)
        portfolio_ret = portfolio_ret*stress.shift(1)

    # Drawdown control
    cum = (1+portfolio_ret).cumprod()
    dd = (cum-cum.cummax())/cum.cummax()
    dd_scale = np.exp(dd*5).clip(0.2,1.0)
    portfolio_ret = portfolio_ret*dd_scale.shift(1)

    # Vol target
    pv = portfolio_ret.rolling(63,min_periods=21).std()*np.sqrt(252)
    ps = (TARGET_VOL/pv.clip(lower=0.005)).clip(0.2,5.0)
    portfolio_ret = portfolio_ret*ps.shift(1)

    avg_active = np.mean(active_counts) if active_counts else 0
    return portfolio_ret.dropna(), avg_active, current_weights


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
    print("V11: LONG-ONLY WITH LEVERAGED & INVERSE ETFs")
    print("="*80)
    prices, fred = load_all_data()
    ret = prices.pct_change()
    print(f"Universe: {prices.shape[1]} ETFs")

    streams = generate_all_streams(ret, fred)
    print(f"Total streams: {len(streams)}")

    eng = {}
    for n in streams:
        p = n.split("_")[0]
        eng[p] = eng.get(p,0)+1
    for e,c in sorted(eng.items()): print(f"  {e}: {c}")

    # Engine-level
    print(f"\n--- Engine Performance ---")
    engine_groups = {}
    for name, s in streams.items():
        prefix = name.split("_")[0]
        engine_groups.setdefault(prefix, {})[name] = s
    for en, ed in sorted(engine_groups.items()):
        edf = pd.DataFrame(ed).dropna(how="all").fillna(0)
        er = edf.mean(axis=1)
        m = metrics(er)
        if m: print(f"  {en:12s}: Sharpe={m['sr']:+.3f}  Ret={m['ret']*100:+.1f}%  Streams={len(ed)}")

    # Top streams
    print(f"\n--- Top 20 Streams ---")
    sm = {n:metrics(s) for n,s in streams.items() if metrics(s)}
    for n,m in sorted(sm.items(), key=lambda x:-x[1]["sr"])[:20]:
        print(f"  {n:30s}: Sharpe={m['sr']:+.3f}  Ret={m['ret']*100:+.1f}%  Vol={m['vol']*100:.1f}%")

    # Portfolio
    print(f"\n{'='*80}")
    port, avg_act, last_w = adaptive_portfolio(streams, fred)
    if port is None: print("FAILED!"); return
    m = metrics(port)
    print(f"FULL SAMPLE ({avg_act:.0f} active, ALL LONG-ONLY):")
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
    dsr=np.sqrt(2*np.log(n_trials))/np.sqrt(m['n']/252)
    print(f"  Deflated Sharpe: {m['sr']-dsr:.3f} (raw {m['sr']:.3f} - {dsr:.3f})")

    # Diversification
    sdf = pd.DataFrame({k:v for k,v in streams.items() if len(v.dropna())>=504}).dropna(how="all").fillna(0)
    cr=sdf.corr(); up=cr.where(np.triu(np.ones(cr.shape),k=1).astype(bool))
    ac=up.stack().mean(); n=sdf.shape[1]
    dm=np.sqrt(n*(1-ac)/(1+(n-1)*ac)) if (1+(n-1)*ac)>0 else 1
    print(f"  Avg corr: {ac:.3f}  Streams: {n}  Div mult: {dm:.2f}x")

    # Current allocation
    print(f"\n{'='*80}")
    print("CURRENT ALLOCATION (long-only, per $100K):")
    print(f"{'='*80}")
    if last_w is not None:
        lw = last_w.sort_values(ascending=False)
        for etf, w in lw.items():
            if w > 0.005:
                # Parse stream name to show what ETFs to buy
                print(f"  {etf:30s}: {w*100:5.1f}%  (${100000*w:>8,.0f})")

    # Save
    rd = DATA_DIR/"results"; rd.mkdir(exist_ok=True)
    port.to_csv(rd/"strategy_v11_returns.csv", header=["return"])
    (1+port).cumprod().to_csv(rd/"strategy_v11_cumulative.csv", header=["cumulative"])
    port.to_csv(rd/"dichs_returns.csv", header=["return"])
    (1+port).cumprod().to_csv(rd/"dichs_cumulative.csv", header=["cumulative"])
    print(f"\n  Saved to {rd}")


if __name__ == "__main__":
    main()
