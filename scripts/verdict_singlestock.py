"""Provenance for verdict.html §4.4: the case against single stocks that has
nothing to do with volatility — computed for readers who say "I can hold
through anything, why not just buy Apple?"

Computes:
  1. Apple's own record: total multiple, near-death, longest underwater stretch,
     longest stretch LOSING TO QQQ (relative underwater), variance drag.
  2. The "it'll always go up" giants of March 2000: total return from the top,
     2000->today, vs QQQ — how many of the era's Apples beat the index.
  3. Terminal-loss scan over the full delisting-inclusive panel (~24k tickers):
     share of stocks that ended (or stand today) >=70% below their peak —
     the loss that holding power cannot fix.

Inputs (all committed): data/stocks/AAPL.csv (Tiingo daily, 1981-),
data/etfs/QQQ.csv, data/etfs/SPY.csv, dca/research/data/tiingo/prices/ac_*.parquet.

Run:  python3 scripts/verdict_singlestock.py
"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_etf(t):
    d = pd.read_csv(f"{ROOT}/data/etfs/{t}.csv", parse_dates=["Date"]).set_index("Date")
    s = d["Adj Close"] if ("Adj Close" in d.columns and d["Adj Close"].notna().sum() > 100) else d["Close"]
    return s.dropna()

aapl = pd.read_csv(f"{ROOT}/data/stocks/AAPL.csv", parse_dates=["date"]).set_index("date")["adjClose"].dropna()
qqq = load_etf("QQQ"); spy = load_etf("SPY")

# ---------- 1) Apple itself ----------
print("=== APPLE: the best-case stock in history, examined ===")
print(f"total multiple {aapl.index[0].date()} -> {aapl.index[-1].date()}: {aapl.iloc[-1]/aapl.iloc[0]:,.0f}x")

# longest absolute underwater stretch (peak -> recovery)
run = aapl / aapl.cummax()
under = run < 1.0
best = (None, None, 0.0, 0)  # start, end, depth, days
start = None; depth = 0.0
for i in range(len(aapl)):
    if under.iloc[i]:
        if start is None:
            start = aapl.index[i]; depth = run.iloc[i]
        depth = min(depth, run.iloc[i])
    else:
        if start is not None:
            days = (aapl.index[i] - start).days
            if days > best[3]:
                best = (start, aapl.index[i], depth, days)
            start = None
if start is not None:
    days = (aapl.index[-1] - start).days
    if days > best[3]:
        best = (start, None, depth, days)
print(f"longest underwater: {best[0].date()} -> {best[1].date() if best[1] else 'today'}"
      f"  ({best[3]/365.25:.1f} years, trough {best[2]-1:+.0%})")

# longest stretch losing to QQQ (relative wealth underwater)
common = aapl.index.intersection(qqq.index)
rel = (aapl[common] / qqq[common])
relrun = rel / rel.cummax()
start = None; depth = 1.0; bestr = (None, None, 1.0, 0)
for i in range(len(rel)):
    if relrun.iloc[i] < 1.0:
        if start is None:
            start = rel.index[i]; depth = relrun.iloc[i]
        depth = min(depth, relrun.iloc[i])
    else:
        if start is not None:
            days = (rel.index[i] - start).days
            if days > bestr[3]:
                bestr = (start, rel.index[i], depth, days)
            start = None
if start is not None and (rel.index[-1] - start).days > bestr[3]:
    bestr = (start, None, depth, (rel.index[-1] - start).days)
print(f"longest stretch LOSING to QQQ: {bestr[0].date()} -> {bestr[1].date() if bestr[1] else 'today'}"
      f"  ({bestr[3]/365.25:.1f} years, fell to {bestr[2]-1:+.0%} vs QQQ)")

# the 2012-2019 stretch specifically (biggest-company era)
w = rel[(rel.index >= "2012-09-01")]
w_trough = w / w.iloc[0]
print(f"from Sep 2012 (world's biggest company): AAPL/QQQ ratio trough {float(w_trough.min())-1:+.0%}"
      f" on {w_trough.idxmin().date()}, back to even {w_trough[w_trough>=1].index[0].date() if (w_trough>=1).any() else 'never'}")

# variance drag, last 10 years (monthly)
am = aapl.resample("ME").last().pct_change().dropna()
qm = qqq.resample("ME").last().pct_change().dropna()
sm = spy.resample("ME").last().pct_change().dropna()
for nm, s in [("AAPL", am), ("QQQ", qm), ("SPY", sm)]:
    s10 = s[s.index >= s.index[-1] - pd.DateOffset(years=10)]
    sig = float(s10.std()) * np.sqrt(12)
    print(f"sigma {nm} (10y monthly, ann.): {sig:.1%}  -> variance drag {sig**2/2*100:.1f} pp/yr")
sig_full = float(am.std()) * np.sqrt(12)
print(f"sigma AAPL full history: {sig_full:.1%} -> drag {sig_full**2/2*100:.1f} pp/yr")

# ---------- 2) the giants of March 2000 ----------
print("\n=== 'it will always go up', March 2000 edition: the 10 biggest US-traded stocks then, held to today ===")
GIANTS = ["MSFT", "GE", "CSCO", "WMT", "XOM", "INTC", "NOK", "PFE", "C", "IBM"]
peak = pd.Timestamp("2000-03-10")
files = {}
import pyarrow.parquet as pq
for f in sorted(glob.glob(f"{ROOT}/dca/research/data/tiingo/prices/ac_*.parquet")):
    cols = set(pq.ParquetFile(f).schema.names)
    for t in GIANTS:
        if t in cols:
            files.setdefault(t, f)
qqq_mult = float(qqq.iloc[-1] / qqq[qqq.index >= peak].iloc[0])
rows = []
for t in GIANTS:
    s = pd.read_parquet(files[t], columns=[t])[t].dropna()
    if len(s) == 0 or s.index[0] > peak:
        print(f"  {t}: insufficient history"); continue
    p0 = s[s.index >= peak].iloc[0]; p1 = s.iloc[-1]
    mult = float(p1 / p0)
    rows.append((t, mult))
    print(f"  {t:5} {mult:6.2f}x  ({mult-1:+8.0%})   beat QQQ ({qqq_mult:.2f}x): {'YES' if mult > qqq_mult else 'no'}")
print(f"  QQQ from same date: {qqq_mult:.2f}x  ({qqq_mult-1:+.0%})")
print(f"  giants beating QQQ over the 26 years: {sum(1 for _, m in rows if m > qqq_mult)}/{len(rows)}")

# ---------- 3) terminal loss scan ----------
print("\n=== terminal losses across the full delisting-inclusive panel ===")
tot = dead_gone = alive_deep = dead_deep = 0
last_date = None
for f in sorted(glob.glob(f"{ROOT}/dca/research/data/tiingo/prices/ac_*.parquet")):
    df = pd.read_parquet(f)
    if last_date is None:
        last_date = df.index[-1]
    for c in df.columns:
        s = df[c].dropna()
        if len(s) < 252:
            continue
        peakv = float(s.max())
        if peakv < 5.0:      # ignore perpetual penny stocks
            continue
        tot += 1
        endv = float(s.iloc[-1])
        alive = (df.index[-1] - s.index[-1]).days < 30
        deep = endv / peakv <= 0.30
        if deep and alive:
            alive_deep += 1
        elif deep and not alive:
            dead_deep += 1
        if not alive:
            dead_gone += 1
print(f"stocks with >=1yr history and a peak >= $5: {tot}")
print(f"no longer trading: {dead_gone} ({dead_gone/tot:.0%})")
print(f"ended >=70% below their peak, DEAD (permanent): {dead_deep} ({dead_deep/tot:.0%})")
print(f"stand >=70% below their peak TODAY (still waiting): {alive_deep} ({alive_deep/tot:.0%})")
print(f"total at/ending >=70% below peak: {(dead_deep+alive_deep)/tot:.0%}")
