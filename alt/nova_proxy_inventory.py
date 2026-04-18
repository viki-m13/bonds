"""Check data availability for NOVA proxy extension back to ~2004.
For each leveraged ETF in the production universe, identify:
  - Real-ETF inception (when the actual ETF started)
  - Best available underlier for synthetic replication pre-inception
  - Underlier inception
"""
from pathlib import Path
import pandas as pd

ETF = Path("/home/user/bonds/data/etfs")

# Map each leveraged ETF to its natural underlier and leverage.
# For the treasury and gold names, use their cash-equivalent index ETF.
MAPPING = {
    "TQQQ": ("QQQ", 3),        # 3x Nasdaq-100
    "UPRO": ("SPY", 3),        # 3x S&P 500
    "QLD":  ("QQQ", 2),
    "SSO":  ("SPY", 2),
    "SOXL": ("SMH", 3),        # 3x semis (SMH is iShares semi ETF)
    "TECL": ("XLK", 3),        # 3x tech
    "FAS":  ("XLF", 3),        # 3x financials
    "LABU": ("XBI", 3),        # 3x biotech
    "ERX":  ("XLE", 2),        # 2x energy
    "NUGT": ("GLD", 2),        # 2x gold miners; no GDX earlier, use GLD as proxy
    "DRN":  ("IYR", 3),        # 3x REITs
    "EDC":  ("EEM", 3),        # 3x emerging markets
    "YINN": ("FXI", 3),        # 3x China
    "UGL":  ("GLD", 2),        # 2x gold
    "UCO":  ("USO", 2),        # 2x crude; USO starts 2006 — fall back to DBC
    "TMF":  ("TLT", 3),        # 3x long-bond
    "TYD":  ("IEF", 3),        # 3x mid-bond
    "UBT":  ("TLT", 2),        # 2x long-bond
}


def load(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()


print(f"{'Leveraged':<10s} {'Real':<12s} | {'Underlier':<10s} {'Underlier start':<16s} {'Lev':>4s}")
print("-" * 70)
for lev, (und, fac) in MAPPING.items():
    r = load(lev)
    u = load(und)
    real_start = r.index.min().date() if r is not None else "missing"
    und_start  = u.index.min().date() if u is not None else "missing"
    print(f"{lev:<10s} {str(real_start):<12s} | {und:<10s} {str(und_start):<16s} {fac}x")

print("\nAlso check SPY, BTC, ETH, BIL, VIXCLS for regime gates:")
for t in ["SPY", "BTC_USD", "ETH_USD", "BIL", "SHY"]:
    r = load(t)
    if r is not None:
        print(f"  {t:10s} {r.index.min().date()} ({len(r)} rows)")

# Test oldest feasible common start
ALL_REQUIRED = list(set(MAPPING.keys()) | set(v[0] for v in MAPPING.values()) | {"SPY"})
latest = None
for t in ALL_REQUIRED:
    r = load(t)
    if r is not None:
        s = r.index.min()
        if latest is None or s > latest:
            latest = s
            limiting = t
print(f"\nLatest-starting required series: {limiting} @ {latest.date()}")
print(f"  (This is the earliest we could run with ALL instruments live.)")

# More realistic: what if some aren't live yet, they just don't contribute?
# The strategy can still run as long as SPY is live (for the regime gate)
# and at least *some* instruments are ready.
print(f"\nSPY earliest: {load('SPY').index.min().date()} — we could start there.")
