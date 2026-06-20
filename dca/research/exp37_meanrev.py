"""Exp 37 — reproduce Quantitativo IBS/lower-band MEAN-REVERSION (claimed Sharpe
~2.1) EXACTLY, on QQQ & SPY. Rules: HLmean=rolling(25) mean of (High-Low);
IBS=(Close-Low)/(High-Low); lower_band=rolling(10) max(High) - 2.5*HLmean;
ENTER long when Close<lower_band AND IBS<0.3; EXIT when Close>prior-day High.
Honest gauntlet: net of cost, exposure-aware Sharpe, OOS train/test, drawdown,
#trades, vs buy&hold. Signal lagged 1 day (enter next close)."""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
t0 = time.time()
raw = yf.download(["QQQ", "SPY"], start="1999-03-01", auto_adjust=True, progress=False)
C, H, L = raw["Close"], raw["High"], raw["Low"]
COST = 0.0005   # 5 bps per round-trip side (liquid ETF, MOC)


def run(tk):
    c, h, l = C[tk].dropna(), H[tk], L[tk]
    h, l = h.reindex(c.index), l.reindex(c.index)
    hl = (h - l)
    hlmean = hl.rolling(25).mean()
    ibs = (c - l) / hl.replace(0, np.nan)
    lower = h.rolling(10).max() - 2.5 * hlmean
    entry = (c < lower) & (ibs < 0.3)
    exit_ = c > h.shift(1)
    pos = np.zeros(len(c)); inpos = False
    e = entry.values; x = exit_.values
    for i in range(len(c)):
        if not inpos and e[i]:
            inpos = True
        elif inpos and x[i]:
            inpos = False
        pos[i] = 1.0 if inpos else 0.0
    pos = pd.Series(pos, index=c.index)
    ret = c.pct_change()
    posl = pos.shift(1).fillna(0)                      # act next day (no lookahead)
    trades = posl.diff().abs().fillna(0)
    sret = posl * ret - COST * trades
    return sret, posl, pos


def stats(sret, posl, lo=None, hi=None):
    if lo:
        m = (sret.index >= lo) & (sret.index < hi); sret = sret[m]; posl = posl[m]
    sret = sret.dropna()
    eq = (1 + sret).cumprod(); yrs = len(sret) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sh = sret.mean() / (sret.std() + 1e-12) * np.sqrt(252)
    mdd = float((eq / eq.cummax() - 1).min())
    expo = (posl > 0).mean()
    ntr = int((posl.diff().abs() > 0).sum() / 2)
    # Sharpe on days-in-market only (the deployed-capital Sharpe)
    inm = sret[posl > 0]
    sh_in = inm.mean() / (inm.std() + 1e-12) * np.sqrt(252) if len(inm) > 30 else np.nan
    return cagr, sh, sh_in, mdd, expo, ntr


for tk in ("QQQ", "SPY"):
    sret, posl, pos = run(tk)
    print(f"\n=== {tk} IBS mean-reversion (net {COST*1e4:.0f}bps/side) ===", flush=True)
    for tag, lo, hi in (("FULL 1999+", "1999-01-01", "2026-12-31"),
                        ("TRAIN <2013", "1999-01-01", "2013-01-01"),
                        ("TEST 2013+", "2013-01-01", "2026-12-31")):
        c_, sh, shin, dd, ex, nt = stats(sret, posl, lo, hi)
        print(f"  {tag:12s} Sharpe {sh:.2f} (in-mkt {shin:.2f})  CAGR {c_*100:4.1f}%  "
              f"maxDD {dd*100:4.0f}%  exposure {ex*100:3.0f}%  trades {nt}", flush=True)
    # buy&hold reference
    bh = C[tk].pct_change()
    eqb = (1 + bh.dropna()).cumprod()
    print(f"  buy&hold     Sharpe {bh.mean()/bh.std()*np.sqrt(252):.2f}  "
          f"CAGR {eqb.iloc[-1]**(252/len(bh.dropna()))-1:.1%}", flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
