"""Step 4: check covered-call / income ETFs and managed futures as
potential high-Sharpe low-vol sleeves."""
from pathlib import Path
import numpy as np
import pandas as pd

ETF = Path("/home/user/bonds/data/etfs")


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
    return ar/av if av > 0 else 0, ar, av, dd, len(r)/252


tickers = ["JEPI", "JEPQ", "QYLD", "XYLD", "SPYI", "RYLD", "DIVO", "KNG",
           "SCHD", "DVY", "HDV", "VIG", "NUSI",
           "DBMF", "KMLM", "CTA", "QAI", "BTAL", "MNA", "IVOL",
           "SWAN", "PFF", "FPE", "CWB", "PGX",
           "JAAA", "CLOI", "JBBB", "SGOV", "USFR"]
print(f"{'Tkr':<6}{'SR':>6}{'Ret':>8}{'Vol':>7}{'MDD':>8}{'Yrs':>6}")
for t in tickers:
    p = load(t)
    if p is None: continue
    r = p.pct_change().fillna(0)
    m = metrics(r)
    if m is None: continue
    print(f"{t:<6}{m[0]:6.2f}{m[1]*100:7.1f}%{m[2]*100:6.1f}%{m[3]*100:7.1f}%{m[4]:6.1f}")
