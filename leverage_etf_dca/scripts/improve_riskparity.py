"""LEVERAGED RISK-PARITY improvement to VOLT.

IDEA: base VOLT vol-targets only TQQQ (3x NASDAQ). Diversify the leverage across
UNCORRELATED sleeves — TQQQ (3x NASDAQ), TMF (3x 20y Treasury), UGL (2x gold),
optional UPRO (3x S&P) / SOXL (3x semis) — each sized by inverse-vol (risk parity)
or equal-risk-budget, with an overall portfolio-vol target; rest in CASH.
Monthly rebalance, 10bps/side, all weights read at PRIOR month-end (no look-ahead).

Leveraged sleeves reconstructed from real underlyings (validated 0.999 vs real TQQQ).
Underlyings TLT/GLD start 2005 -> full book runs 2005/2006-2026.

Run:  python3 improve_riskparity.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import strategy as S   # base VOLT harness: close, mgrid, mret, tqqq_weight, strat_ret, dca, lump_stats

close, mgrid, mret = S.close, S.mgrid, S.mret
retd = close.pct_change()

# ---- cash monthly return: BIL, fallback SHY, fallback 0 (honest, low, no bond tailwind) ----
def _mret_of(tk):
    return close[tk].reindex(mgrid).pct_change()
cash_m = _mret_of("BIL")
cash_m = cash_m.fillna(_mret_of("SHY")).fillna(0.0)

# ---- per-sleeve trailing annualized vol at each month-end (NOT yet shifted) ----
VOLWIN = 63
def sleeve_vol(tk, win=VOLWIN):
    return (retd[tk].rolling(win, min_periods=40).std() * np.sqrt(252)).reindex(mgrid, method="ffill")

# ---- ex-ante portfolio vol of a weight vector using trailing daily covariance ----
def _exante_vol(sleeves, nvec, dt, win=VOLWIN):
    # nvec: dict sleeve->weight, at month-end dt; trailing daily cov ending dt
    win_ret = retd.loc[:dt, sleeves].iloc[-win:]
    if len(win_ret) < 40:
        return np.nan
    cov = win_ret.cov().values * 252.0
    n = np.array([nvec[s] for s in sleeves])
    var = float(n @ cov @ n)
    return np.sqrt(var) if var > 0 else np.nan

# ---- build monthly weight matrix for a construction ----
# mode: 'rp'  = inverse-vol normalized (sum=1) then scaled to target using ex-ante portfolio vol
#       'erb' = equal-risk-budget: each sleeve vol-targeted to target/sqrt(N) (assumes independence), then capped
def build_weights(sleeves, target=0.15, kmax=1.0, mode="rp", win=VOLWIN):
    vols = {s: sleeve_vol(s, win) for s in sleeves}
    W = pd.DataFrame(0.0, index=mgrid, columns=sleeves)
    N = len(sleeves)
    for dt in mgrid:
        vv = {s: vols[s].get(dt, np.nan) for s in sleeves}
        if any(not np.isfinite(vv[s]) or vv[s] <= 0 for s in sleeves):
            continue
        inv = {s: 1.0 / vv[s] for s in sleeves}
        if mode == "rp":
            tot = sum(inv.values())
            n = {s: inv[s] / tot for s in sleeves}          # risk-parity proportions, sum=1
            sp = _exante_vol(sleeves, n, dt, win)           # ex-ante portfolio vol of that book
            if not np.isfinite(sp) or sp <= 0:
                continue
            k = min(target / sp, kmax)                       # scale whole book to target vol
            for s in sleeves:
                W.loc[dt, s] = k * n[s]
        elif mode == "erb":
            per = target / np.sqrt(N)                        # per-sleeve vol target (independence assumption)
            raw = {s: per / vv[s] for s in sleeves}
            gross = sum(raw.values())
            scale = min(1.0, kmax / gross) if gross > 0 else 0.0
            for s in sleeves:
                W.loc[dt, s] = raw[s] * scale
    return W.shift(1)   # decided at prior month-end

# ---- book return: sum_i w_i*ret_i + cash*cash_ret - turnover cost (10bps/side per sleeve leg) ----
def book_ret(Wsh, start, end, cost=0.001):
    g = mgrid[(mgrid >= start) & (mgrid <= end)]
    sleeves = list(Wsh.columns)
    prev = {s: 0.0 for s in sleeves}
    rr = []
    for dt in g:
        w = {}
        for s in sleeves:
            x = Wsh.loc[dt, s]
            w[s] = float(x) if np.isfinite(x) else 0.0
        wsum = sum(w.values())
        cashw = max(0.0, 1.0 - wsum)
        port = cashw * (cash_m.get(dt, 0.0) if np.isfinite(cash_m.get(dt, 0.0)) else 0.0)
        turn = 0.0
        for s in sleeves:
            rs = mret.loc[dt, s]; rs = rs if np.isfinite(rs) else 0.0
            port += w[s] * rs
            turn += abs(w[s] - prev[s])
            prev[s] = w[s]
        rr.append((dt, port - turn * cost))
    return pd.Series(dict(rr))

# ---- metrics ----
def dca_final(r, contrib=1000.0):
    V = 0.0
    for x in r.dropna(): V = V * (1 + x) + contrib
    return V
def qqq_dca_final(g):
    V = 0.0
    for x in mret["QQQ"].reindex(g).fillna(0): V = V * (1 + x) + 1000.0
    return V
def stats(r):
    return S.lump_stats(r)

# ---- eras: match base VOLT (bond/gold ETFs need 2005+, so start 2006) ----
ERAS = [("2006-01","2009-12","06-09"),("2010-01","2014-12","10-14"),
        ("2015-01","2019-12","15-19"),("2020-01","2026-06","20-26"),
        ("2010-01","2026-06","10-26"),("2006-01","2026-06","full06-26")]
Sd, Ed = pd.Timestamp("2006-01-01"), pd.Timestamp("2026-07-01")

# base VOLT reference series (vt30 TQQQ | GLD-TLT)
base_r = S.strat_ret(Sd, Ed, target=0.30, defense=("GLD","TLT"))

def era_ratios_vs_qqq(rfun):
    out = []
    for a,b,_ in ERAS:
        s,e = pd.Timestamp(a+"-01"), pd.Timestamp(b+"-01")
        g = mgrid[(mgrid>=s)&(mgrid<=e)]
        out.append(rfun(s,e) / qqq_dca_final(g))
    return out

def era_ratios_vs_base(rfun):
    out = []
    for a,b,_ in ERAS:
        s,e = pd.Timestamp(a+"-01"), pd.Timestamp(b+"-01")
        out.append(rfun(s,e) / dca_final(S.strat_ret(s,e,target=0.30,defense=("GLD","TLT"))))
    return out

CONSTR = {
    "RP3 TQQQ+TMF+UGL 15%":   dict(sleeves=["TQQQ","TMF","UGL"], target=0.15, mode="rp"),
    "RP3 TQQQ+TMF+UGL 20%":   dict(sleeves=["TQQQ","TMF","UGL"], target=0.20, mode="rp"),
    "RP3 TQQQ+TMF+UGL 12%":   dict(sleeves=["TQQQ","TMF","UGL"], target=0.12, mode="rp"),
    "ERB3 TQQQ+TMF+UGL 15%":  dict(sleeves=["TQQQ","TMF","UGL"], target=0.15, mode="erb"),
    "RP2 TQQQ+TMF 15%":       dict(sleeves=["TQQQ","TMF"],       target=0.15, mode="rp"),
    "RP2 TQQQ+TMF 20%":       dict(sleeves=["TQQQ","TMF"],       target=0.20, mode="rp"),
    "RP3 TQQQ+TMF+UGL 25%":   dict(sleeves=["TQQQ","TMF","UGL"], target=0.25, mode="rp"),
    "RP3 TQQQ+TMF+UGL 30%":   dict(sleeves=["TQQQ","TMF","UGL"], target=0.30, mode="rp", kmax=1.5),
    "RP4 +UPRO 15%":          dict(sleeves=["TQQQ","TMF","UGL","UPRO"], target=0.15, mode="rp"),
    "RP5 +UPRO+SOXL 15%":     dict(sleeves=["TQQQ","TMF","UGL","UPRO","SOXL"], target=0.15, mode="rp"),
}

def realized_vol(r):
    r = r.dropna(); return r.std()*np.sqrt(12)

if __name__ == "__main__":
    print("="*118)
    print("LEVERAGED RISK PARITY vs base VOLT vs QQQ-DCA")
    print("="*118)

    # precompute weight matrices
    WMATS = {nm: build_weights(**kw) for nm,kw in CONSTR.items()}
    RFUN = {nm: (lambda s,e,W=W: dca_final(book_ret(W,s,e))) for nm,W in WMATS.items()}

    hdr = f"{'construction':26}" + " ".join(f"{lab:>9}" for *_,lab in ERAS)
    print("\n--- DCA final-wealth ratio vs QQQ-DCA ($1000/mo) ---")
    print(hdr)
    # base VOLT reference row
    b = era_ratios_vs_qqq(lambda s,e: dca_final(S.strat_ret(s,e,target=0.30,defense=("GLD","TLT"))))
    print(f"{'BASE VOLT vt30 TQQQ|GT':26}" + " ".join(f"{v:>9.2f}" for v in b))
    for nm in CONSTR:
        r = era_ratios_vs_qqq(RFUN[nm])
        print(f"{nm:26}" + " ".join(f"{v:>9.2f}" for v in r))

    print("\n--- DCA final-wealth ratio vs BASE VOLT (>1 = beats base) ---")
    print(hdr)
    for nm in CONSTR:
        r = era_ratios_vs_base(RFUN[nm])
        print(f"{nm:26}" + " ".join(f"{v:>9.2f}" for v in r))

    print("\n--- Honest lump-sum $1 risk, full 2006-2026 (RealVol = realized annualized vol) ---")
    print(f"{'strategy':26} {'CAGR':>6} {'RealVol':>7} {'Shrp':>5} {'maxDD':>7} {'wrst12':>7} {'UW':>3} {'mult':>7}")
    bs = stats(base_r)
    print(f"{'BASE VOLT vt30 TQQQ|GT':26} {bs['cagr']:>6.1%} {realized_vol(base_r):>7.0%} {bs['sharpe']:>5.2f} {bs['maxdd']:>7.0%} {bs['worst12']:>7.0%} {bs['uw']:>3d} {bs['mult']:>6.0f}x")
    for nm in CONSTR:
        r = book_ret(WMATS[nm], Sd, Ed); st = stats(r)
        print(f"{nm:26} {st['cagr']:>6.1%} {realized_vol(r):>7.0%} {st['sharpe']:>5.2f} {st['maxdd']:>7.0%} {st['worst12']:>7.0%} {st['uw']:>3d} {st['mult']:>6.0f}x")
    for nm,tk in [("QQQ buy&hold","QQQ"),("TQQQ buy&hold","TQQQ")]:
        r = mret[tk].reindex(mgrid[(mgrid>=Sd)&(mgrid<=Ed)]); st = stats(r)
        print(f"{nm:26} {st['cagr']:>6.1%} {realized_vol(r):>7.0%} {st['sharpe']:>5.2f} {st['maxdd']:>7.0%} {st['worst12']:>7.0%} {st['uw']:>3d} {st['mult']:>6.0f}x")

    # ---- 2022 stress: stocks AND bonds fell together. does gold save it? ----
    print("\n--- 2022 STRESS (stocks+bonds down together; did gold help?) ---")
    s22,e22 = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")
    g22 = mgrid[(mgrid>=s22)&(mgrid<=e22)]
    def cum(r): return (1+r.dropna()).prod()-1
    print(f"{'series':26} {'2022 ret':>9} {'2022 maxDD':>11}")
    def ddof(r):
        c=(1+r.dropna()).cumprod(); return (c/c.cummax()-1).min()
    print(f"{'BASE VOLT':26} {cum(S.strat_ret(s22,e22,target=0.30,defense=('GLD','TLT'))):>9.1%} {ddof(S.strat_ret(s22,e22,target=0.30,defense=('GLD','TLT'))):>11.1%}")
    for nm in ["RP3 TQQQ+TMF+UGL 15%","RP2 TQQQ+TMF 15%"]:
        r22 = book_ret(WMATS[nm], s22, e22)
        print(f"{nm:26} {cum(r22):>9.1%} {ddof(r22):>11.1%}")
    for tk in ["TQQQ","TMF","UGL","QQQ"]:
        r=mret[tk].reindex(g22)
        print(f"{tk+' (underlying sleeve)':26} {cum(r):>9.1%} {ddof(r):>11.1%}")

    # ---- phase robustness: shift rebalance day within the month ----
    print("\n--- PHASE ROBUSTNESS (rebalance-day sensitivity, RP3 15%, full 06-26 Sharpe/CAGR/maxDD) ---")
    def phase_grid(sleeves, target, mode, offsets=(-8,-4,0,4,8)):
        res=[]
        for off in offsets:
            # shift the monthly grid by `off` trading days
            alld = close.index
            newg = []
            for dt in mgrid:
                pos = alld.get_indexer([dt])[0] + off
                pos = min(max(pos,0), len(alld)-1)
                newg.append(alld[pos])
            newg = pd.DatetimeIndex(sorted(set(newg)))
            res.append((off, newg))
        return res
    # simple robustness: rebuild weights on shifted grids
    def build_weights_grid(sleeves, target, mode, grid, win=VOLWIN):
        vols = {s: (retd[s].rolling(win,min_periods=40).std()*np.sqrt(252)).reindex(grid,method="ffill") for s in sleeves}
        W = pd.DataFrame(0.0, index=grid, columns=sleeves); N=len(sleeves)
        for dt in grid:
            vv={s:vols[s].get(dt,np.nan) for s in sleeves}
            if any(not np.isfinite(vv[s]) or vv[s]<=0 for s in sleeves): continue
            inv={s:1.0/vv[s] for s in sleeves}
            if mode=="rp":
                tot=sum(inv.values()); n={s:inv[s]/tot for s in sleeves}
                sp=_exante_vol(sleeves,n,dt,win)
                if not np.isfinite(sp) or sp<=0: continue
                k=min(target/sp,1.0)
                for s in sleeves: W.loc[dt,s]=k*n[s]
        return W.shift(1)
    def book_ret_grid(Wsh, grid, start, end, cost=0.001):
        gclose = close.reindex(grid)
        gret = gclose.pct_change()
        gcash = cash_m.reindex(grid, method="nearest").fillna(0.0)
        g=grid[(grid>=start)&(grid<=end)]; sleeves=list(Wsh.columns); prev={s:0.0 for s in sleeves}; rr=[]
        for dt in g:
            w={s:(float(Wsh.loc[dt,s]) if np.isfinite(Wsh.loc[dt,s]) else 0.0) for s in sleeves}
            wsum=sum(w.values()); cashw=max(0.0,1-wsum)
            port=cashw*(gcash.get(dt,0.0) if np.isfinite(gcash.get(dt,0.0)) else 0.0); turn=0.0
            for s in sleeves:
                rs=gret.loc[dt,s]; rs=rs if np.isfinite(rs) else 0.0
                port+=w[s]*rs; turn+=abs(w[s]-prev[s]); prev[s]=w[s]
            rr.append((dt,port-turn*cost))
        return pd.Series(dict(rr))
    print(f"{'offset(days)':14} {'CAGR':>6} {'Shrp':>5} {'maxDD':>7} {'mult':>7}")
    for off,newg in phase_grid(["TQQQ","TMF","UGL"],0.15,"rp"):
        Wg=build_weights_grid(["TQQQ","TMF","UGL"],0.15,"rp",newg)
        st=stats(book_ret_grid(Wg,newg,Sd,Ed))
        print(f"{off:>+14d} {st['cagr']:>6.1%} {st['sharpe']:>5.2f} {st['maxdd']:>7.0%} {st['mult']:>6.0f}x")

    print("\ndone.")
