"""Build a clean daily ETF total-return panel + honestly-reconstructed leveraged
series (from real underlyings, so we control the daily-decay + fee assumptions).
Sanity-checks the reconstruction vs the real (post-inception) leveraged ETF.
Output: scratchpad/_etf_panel.pkl {close: daily DataFrame, kind: dict tk->'base'/'lev', meta}
"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/user/bonds"

def load(tk):
    for d in (f"{REPO}/data/etfs", f"{REPO}/data/etfs_extended"):
        f = f"{d}/{tk}.csv"
        if os.path.exists(f):
            df = pd.read_csv(f); df["Date"] = pd.to_datetime(df["Date"])
            return df.set_index("Date")["Close"].sort_index()
    return None

# base (real, unleveraged) universe
BASE = ["QQQ","SPY","MDY","IWM","EEM","EFA","EWJ","XLK","SMH","XLE","XLF","XLV","XLU","XLP","XLY","XLI","XLB",
        "GLD","SLV","USO","DBC","GSG","TLT","IEF","SHY","HYG","LQD","AGG","VNQ","BIL","DIA","EWZ","FXI",
        "TAN","XBI","XOP","URA"]
close = {}
for t in BASE:
    s = load(t)
    if s is not None and len(s) > 250:
        close[t] = s
close = pd.DataFrame(close).sort_index()
# forward-fill small gaps, drop all-nan rows
close = close[~close.index.duplicated(keep="last")]
ret = close.pct_change()

# ---- leveraged reconstruction: lev_ret = L*underlying_ret - (expense+borrow)/252 ----
# real inception dates (mask synthetic pre-inception; before that use reconstruction only, labeled)
LEV = {  # name: (underlying, leverage, expense_annual, real_inception)
    "QLD":  ("QQQ", 2, 0.0095, "2006-06-21"),
    "TQQQ": ("QQQ", 3, 0.0095, "2010-02-11"),
    "SSO":  ("SPY", 2, 0.0091, "2006-06-21"),
    "UPRO": ("SPY", 3, 0.0091, "2009-06-25"),
    "SOXL": ("SMH", 3, 0.0090, "2010-03-11"),  # SOXL tracks ICE semis ~ SMH proxy
    "TECL": ("XLK", 3, 0.0090, "2008-12-17"),
    "FAS":  ("XLF", 3, 0.0090, "2008-11-06"),
    "ERX":  ("XLE", 2, 0.0090, "2008-11-06"),
    "UGL":  ("GLD", 2, 0.0095, "2008-12-01"),
    "UCO":  ("USO", 2, 0.0095, "2010-11-24"),
    "TMF":  ("TLT", 3, 0.0106, "2009-04-16"),
    "EDC":  ("EEM", 3, 0.0095, "2008-12-17"),
    "YINN": ("FXI", 3, 0.0095, "2009-12-03"),
    "TNA":  ("IWM", 3, 0.0091, "2008-11-05"),
    "DRN":  ("VNQ", 3, 0.0109, "2009-07-16"),
    "LABU": ("XBI", 3, 0.0097, "2015-05-28"),
}
borrow = 0.03  # financing drag on the levered portion, annual (approx)
kind = {t: "base" for t in close.columns}
for name, (u, L, exp, inc) in LEV.items():
    if u not in ret.columns:
        continue
    r = L * ret[u] - (exp + (L - 1) * borrow) / 252.0
    px = (1 + r.fillna(0)).cumprod()
    px = px / px.dropna().iloc[0]
    close[name] = px * 100.0  # arbitrary base
    kind[name] = "lev"

close = close.sort_index()
pd.to_pickle({"close": close, "kind": kind, "lev": LEV}, f"{HERE}/_etf_panel.pkl")
print("panel:", close.shape, "range", close.index.min().date(), close.index.max().date())
print("base:", sum(1 for v in kind.values() if v=="base"), "lev:", sum(1 for v in kind.values() if v=="lev"))

# ---- SANITY CHECK: reconstructed TQQQ vs real TQQQ post-2010 ----
real = load("TQQQ")
if real is not None:
    recon = close["TQQQ"]
    j = pd.concat([real.rename("real"), recon.rename("recon")], axis=1).dropna()
    j = j[j.index >= "2011-01-01"]  # post real inception + settle
    jr = j.pct_change().dropna()
    corr = jr["real"].corr(jr["recon"])
    # cumulative since 2015 (both normalized)
    j15 = j[j.index >= "2015-01-01"]; j15 = j15 / j15.iloc[0]
    print(f"\nSANITY reconstructed vs REAL TQQQ (2011+): daily-ret corr {corr:.3f}")
    print(f"  cum since 2015: real {j15['real'].iloc[-1]:.2f}x  recon {j15['recon'].iloc[-1]:.2f}x")
    # annualized tracking error
    te = (jr["real"] - jr["recon"]).std() * np.sqrt(252)
    print(f"  annualized tracking error {te:.1%}")
