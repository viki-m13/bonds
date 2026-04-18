"""Build nova_factsheet_data.json from nova_returns.csv + nova_rebalances.csv.

Schema matches docs/aurora.html (single-sleeve variant). NOVA is a unified
cross-sectional momentum strategy (7 x 3x ETFs + BTC + ETH) with per-name
33% cap. One sleeve, not three.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
RESULTS = ROOT / "data/results"

UNIVERSE = [
    "TQQQ", "UPRO", "SOXL", "TECL", "FAS", "TMF", "UGL",
    "LABU", "EDC", "YINN", "ERX", "NUGT", "DRN", "UCO", "TYD",
    "QLD", "SSO", "UBT",
    "BTC_USD", "ETH_USD",
]
ETF_META = {
    "TQQQ":    ("ProShares UltraPro QQQ", "3x Nasdaq-100", "Leveraged Equity"),
    "UPRO":    ("ProShares UltraPro S&P 500", "3x S&P 500", "Leveraged Equity"),
    "QLD":     ("ProShares Ultra QQQ", "2x Nasdaq-100", "Leveraged Equity"),
    "SSO":     ("ProShares Ultra S&P 500", "2x S&P 500", "Leveraged Equity"),
    "SOXL":    ("Direxion Daily Semiconductor Bull 3X", "3x semiconductors", "Leveraged Sector"),
    "TECL":    ("Direxion Daily Technology Bull 3X", "3x tech sector", "Leveraged Sector"),
    "FAS":     ("Direxion Daily Financial Bull 3X", "3x financials", "Leveraged Sector"),
    "LABU":    ("Direxion Daily S&P Biotech Bull 3X", "3x biotech", "Leveraged Sector"),
    "ERX":     ("Direxion Daily Energy Bull 2X", "2x energy sector", "Leveraged Sector"),
    "NUGT":    ("Direxion Daily Gold Miners Bull 2X", "2x gold miners", "Leveraged Sector"),
    "DRN":     ("Direxion Daily Real Estate Bull 3X", "3x REITs", "Leveraged Sector"),
    "EDC":     ("Direxion Daily MSCI Emerging Mkts Bull 3X", "3x emerging markets", "Leveraged Country"),
    "YINN":    ("Direxion Daily FTSE China Bull 3X", "3x China", "Leveraged Country"),
    "UGL":     ("ProShares Ultra Gold", "2x gold bullion", "Leveraged Commodity"),
    "UCO":     ("ProShares Ultra Bloomberg Crude Oil", "2x WTI crude", "Leveraged Commodity"),
    "TMF":     ("Direxion Daily 20+Y Treasury Bull 3X", "3x long-duration treasuries", "Leveraged Rates"),
    "TYD":     ("Direxion Daily 7-10Y Treasury Bull 3X", "3x mid-duration treasuries", "Leveraged Rates"),
    "UBT":     ("ProShares Ultra 20+Y Treasury", "2x long-duration treasuries", "Leveraged Rates"),
    "BTC_USD": ("Bitcoin", "Spot BTC (coinbase)", "Crypto"),
    "ETH_USD": ("Ethereum", "Spot ETH (coinbase)", "Crypto"),
}

CAP = 0.33
FEE_ANNUAL = 0.01


def load_etf(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()


def metrics(r):
    if len(r) == 0 or r.std() == 0:
        return {"sharpe": 0, "ann_return": 0, "ann_vol": 0, "max_dd": 0,
                "sortino": 0, "calmar": 0, "total_return": 0, "n_years": 0,
                "win_rate_daily": 0, "skew": 0, "kurt": 0, "best_month": 0,
                "worst_month": 0, "pct_pos_months": 0, "avg_dd_days": 0,
                "max_dd_days": 0, "inception": ""}
    ar = r.mean() * 252
    av = r.std() * np.sqrt(252)
    sr = ar / av if av > 0 else 0
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    mdd = dd.min()
    neg = r[r < 0]
    sor = ar / (neg.std() * np.sqrt(252)) if len(neg) and neg.std() > 0 else 999
    is_dd = dd < 0
    runs = []; cur = 0
    for v in is_dd:
        if v: cur += 1
        else:
            if cur > 0: runs.append(cur); cur = 0
    if cur > 0: runs.append(cur)
    avg_dd_days = int(np.mean(runs)) if runs else 0
    max_dd_days = int(max(runs)) if runs else 0
    monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    pct_pos = float((monthly > 0).sum() / len(monthly) * 100) if len(monthly) else 0
    return {
        "name": "",
        "total_return": round(float((cum.iloc[-1] - 1) * 100), 2),
        "ann_return": round(float(ar * 100), 2),
        "ann_vol": round(float(av * 100), 2),
        "sharpe": round(float(sr), 3),
        "sortino": round(float(sor), 3),
        "max_dd": round(float(mdd * 100), 2),
        "calmar": round(float(ar / abs(mdd)) if mdd < 0 else 0, 2),
        "win_rate_daily": round(float((r > 0).sum() / len(r) * 100), 1),
        "skew": round(float(r.skew()), 2),
        "kurt": round(float(r.kurtosis()), 2),
        "best_month": round(float(monthly.max() * 100) if len(monthly) else 0, 2),
        "worst_month": round(float(monthly.min() * 100) if len(monthly) else 0, 2),
        "pct_pos_months": round(float(pct_pos), 1),
        "avg_dd_days": avg_dd_days,
        "max_dd_days": max_dd_days,
        "n_years": round(float(len(r) / 252), 1),
        "inception": str(r.index[0].date()),
    }


def trailing(r):
    c = (1 + r).cumprod()
    today = c.iloc[-1]
    out = {}
    for name, days in [("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252),
                       ("3Y_ann", 252*3), ("5Y_ann", 252*5), ("10Y_ann", 252*10)]:
        if len(c) > days:
            if name.endswith("_ann"):
                yrs = days / 252
                out[name] = round(float(((today / c.iloc[-1 - days]) ** (1 / yrs) - 1) * 100), 2)
            else:
                out[name] = round(float((today / c.iloc[-1 - days] - 1) * 100), 2)
        else:
            out[name] = None
    year_start = pd.Timestamp(f"{c.index[-1].year}-01-01")
    ys = c.loc[c.index >= year_start]
    out["YTD"] = round(float((ys.iloc[-1] / ys.iloc[0] - 1) * 100), 2) if len(ys) > 1 else 0
    out["SI_ann"] = round(float((today ** (252 / len(c)) - 1) * 100), 2) if len(c) > 0 else 0
    return out


if __name__ == "__main__":
    print("NOVA factsheet builder ready (run nova_factsheet_run.py)")
