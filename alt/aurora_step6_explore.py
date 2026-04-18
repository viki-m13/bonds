"""Step 6: Explore untested high-Sharpe or diversifying ETFs.
Focus on merger arb (MNA), hedge-fund replication (QAI), preferreds (PFF/PFFD),
active bonds (BOND/FBND), convertibles (CWB), buy-write ladders (QYLD/XYLD/RYLD).
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("/home/user/bonds/data")
ETF = DATA / "etfs"


def load(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()


def metrics(r):
    r = r.loc[r.ne(0).idxmax():] if (r != 0).any() else r
    if r.std() == 0 or len(r) == 0: return None
    ar, av = r.mean() * 252, r.std() * np.sqrt(252)
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return {"sharpe": ar/av if av > 0 else 0, "ret": ar, "vol": av, "mdd": dd,
            "n": len(r)/252, "start": str(r.index[0].date())}


candidates = ["MNA", "QAI", "BTAL", "TAIL", "SWAN", "IVOL", "BOND", "FBND",
              "PFF", "PFFD", "CWB", "QYLD", "XYLD", "RYLD", "KNG",
              "DBV", "CEW", "CYB", "SCHD", "DIVO", "JEPI", "JEPQ", "SPYI",
              "JBBB", "CLOI", "JAAA", "JPST", "MINT", "FPE", "HDV", "VIG",
              "AGG", "BND", "LQD", "HYG", "JNK", "USHY", "ANGL", "MBB",
              "IBDQ", "IBDR", "IBDS", "IBDT", "USFR", "FLOT", "SHYG"]

print(f"{'Ticker':<8}{'SR':>7}{'Ret':>8}{'Vol':>7}{'MDD':>8}{'Yrs':>6}  Since")
for t in candidates:
    p = load(t)
    if p is None: continue
    r = p.loc["2014-01-01":].pct_change().dropna()
    m = metrics(r)
    if m is None: continue
    print(f"{t:<8}{m['sharpe']:7.2f}{m['ret']*100:7.1f}%{m['vol']*100:6.1f}%"
          f"{m['mdd']*100:7.1f}%{m['n']:6.1f}  {m['start']}")
