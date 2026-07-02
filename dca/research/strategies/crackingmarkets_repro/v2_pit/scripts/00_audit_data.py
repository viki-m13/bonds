"""Audit the committed data assets before any backtesting.

Checks:
1. n100 PIT panels — adjustment sanity (AAPL/NVDA/TSLA splits must NOT appear
   as price jumps), OHLC consistency, membership coverage per year.
2. Tiingo survivorship-clean panel — load all ac_/vol_ chunks, report shape,
   delisted coverage, spot-check a few total-return series vs known values.
3. ETF OHLC (QQQ, TLT, IEF, GLD, SPY) — opens present, adjusted.
"""
import os, json
import numpy as np
import pandas as pd

ROOT = "/home/user/bonds"
PIT = os.path.join(ROOT, "data", "pit")
TII = os.path.join(ROOT, "dca", "research", "data", "tiingo", "prices")
ETF = os.path.join(ROOT, "data", "etfs")
ETFX = os.path.join(ROOT, "data", "etfs_extended")

print("=" * 70)
print("1. NDX-100 PIT panels")
print("=" * 70)
panels = {f: pd.read_parquet(os.path.join(PIT, f"n100_panel_{f}.parquet"))
          for f in ["open", "high", "low", "close", "volume", "member"]}
c, o, h, l, mem = (panels[k] for k in ["close", "open", "high", "low", "member"])
print(f"shape {c.shape}, {c.index[0].date()} -> {c.index[-1].date()}")

# split-adjustment check: big known splits should not show as ~1/N jumps
splits = {"AAPL": "2020-08-31", "TSLA": "2022-08-25", "NVDA": "2024-06-10",
          "AMZN": "2022-06-06", "GOOGL": "2022-07-18"}
for t, d in splits.items():
    if t in c.columns:
        d = pd.Timestamp(d)
        seg = c[t].loc[:d].dropna()
        if len(seg) > 1:
            r = seg.pct_change().iloc[-1]
            print(f"  {t} return on split date {d.date()}: {r:+.3%}  "
                  f"{'OK (adjusted)' if abs(r) < 0.2 else 'SPLIT NOT ADJUSTED!'}")

# OHLC consistency
viol = ((h < l) | (c > h * 1.001) | (c < l * 0.999) | (o > h * 1.001) | (o < l * 0.999))
print(f"  OHLC violations: {viol.sum().sum()} cells "
      f"({viol.sum().sum() / c.notna().sum().sum():.4%} of non-nan)")

# membership coverage: members with price data per year (2015+)
memyr = {}
for y in range(2015, 2027):
    days = mem.loc[str(y)]
    if not len(days):
        continue
    mid = days.index[len(days) // 2]
    members = mem.loc[mid]
    members = members[members].index
    have = c.loc[mid, members].notna().sum()
    memyr[y] = (len(members), int(have))
print("  year: members_in_mask, with_price")
for y, (n, hv) in memyr.items():
    print(f"  {y}: {n:4d} {hv:4d}  ({hv/n:.0%})")

# how many names ever member vs columns
print(f"  panel columns: {len(c.columns)}")

print()
print("=" * 70)
print("2. Tiingo survivorship-clean panel")
print("=" * 70)
acs = sorted(f for f in os.listdir(TII) if f.startswith("ac_"))
ac = pd.concat([pd.read_parquet(os.path.join(TII, f)) for f in acs], axis=1)
ac = ac.loc[:, ~ac.columns.duplicated()]
print(f"adjClose panel: {ac.shape}, {ac.index[0].date()} -> {ac.index[-1].date()}")
uni = pd.read_parquet(os.path.join(ROOT, "dca", "research", "data", "tiingo",
                                   "tiingo_universe_pit.parquet"))
print(f"universe rows: {len(uni)}, cols: {uni.columns.tolist()}")
delisted = uni[pd.to_datetime(uni["endDate"]) < "2026-06-01"]
print(f"delisted in universe: {len(delisted)}; with price data: "
      f"{sum(t in ac.columns for t in delisted['ticker'].head(2000))}/2000 sampled")

vols = sorted(f for f in os.listdir(TII) if f.startswith("vol_"))
vv = pd.concat([pd.read_parquet(os.path.join(TII, f)) for f in vols], axis=1)
vv = vv.loc[:, ~vv.columns.duplicated()]
print(f"volume panel: {vv.shape}")

# spot check: AAPL total return 2010->2020 should be ~10x+
for t in ["AAPL", "MSFT", "SPY", "QQQ", "TLT", "GLD", "IEF"]:
    if t in ac.columns:
        s = ac[t].dropna()
        print(f"  {t}: {s.index[0].date()} -> {s.index[-1].date()}, "
              f"last={s.iloc[-1]:.2f}")

print()
print("=" * 70)
print("3. ETF OHLC files")
print("=" * 70)
for d in [ETF, ETFX]:
    for t in ["QQQ", "SPY", "TLT", "IEF", "GLD"]:
        p = os.path.join(d, f"{t}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            print(f"  {os.path.basename(d)}/{t}: {df.shape} "
                  f"{df.index[0].date()}->{df.index[-1].date()} cols={list(df.columns)[:6]}")
