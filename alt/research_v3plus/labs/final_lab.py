"""Final composition lab — IS ONLY."""
import sys
sys.path.insert(0, "/home/user/bonds/alt")
import numpy as np
import pandas as pd
from sleeve_engine import backtest_weights, sharpe, cagr, max_dd, ann_vol
from research_lab import (van, ori, hel, cry, reversal, tom, bondls,
                          DATES, IS_END, load_etf)
from blend_lab import erc_weights, overlay

IS_END_TS = pd.Timestamp(IS_END)
ERC_START = "2015-01-02"

def rep_is(name, r, extra=""):
    ri = r.loc[:IS_END_TS]
    print(f"{name:34s} IS: SR {sharpe(ri):5.2f}  CAGR {cagr(ri)*100:6.1f}%  "
          f"Vol {ann_vol(ri)*100:5.1f}%  MDD {max_dd(ri)*100:6.1f}%{extra}")

# ---------------- expanded REVERSAL
def reversal_x(z_th=-1.0, hold=5, zwin=60, pairs=None, cost=10.0):
    pairs = pairs or {"QQQ": "TQQQ", "SPY": "UPRO", "SMH": "SOXL",
                      "EEM": "EDC", "XLE": "ERX"}
    closes = pd.DataFrame({u: load_etf(u)["Close"].reindex(DATES).ffill(limit=3) for u in pairs})
    opens = pd.DataFrame({l: load_etf(l)["Open"].reindex(DATES) for l in pairs.values()})
    r5 = closes.pct_change(5)
    z = (r5 - r5.rolling(zwin).mean()) / r5.rolling(zwin).std()
    above = closes > closes.rolling(200).mean()
    trigger = ((z < z_th) & above).shift(1).fillna(False)
    held = trigger.astype(float).rolling(hold, min_periods=1).max()
    tot = held.sum(axis=1)
    W = held.div(tot.where(tot > 0), axis=0).mul(tot.clip(upper=1.0), axis=0).fillna(0.0)
    W = W.rename(columns=pairs)
    return backtest_weights(W, opens, cost)["ret"]

print("== REVERSAL expanded ==")
rep_is("rev 3-name (base)", reversal(-1.0, 5))
rep_is("rev 5-name", reversal_x())
rep_is("rev 5-name z<-1.25", reversal_x(-1.25))

rev = reversal_x()  # will pick based on print
tomr = tom("TQQQ", trend_filter=False)
bnd = bondls(63, None)

# ---------------- weight schemes on final sleeve sets
def blend(sleeves, w, target_vol=0.16):
    df = pd.concat(sleeves, axis=1).reindex(DATES).fillna(0.0)
    raw = df @ pd.Series(w).reindex(df.columns).fillna(0.0)
    net, mult = overlay(raw, target_vol=target_vol)
    return net, mult

def fit_weights(sleeves, scheme="erc", budgets=None, shrink=0.5):
    df = pd.concat(sleeves, axis=1).reindex(DATES).fillna(0.0)
    win = df.loc[ERC_START:IS_END_TS]
    cov = win.cov()
    if scheme == "erc":
        w = erc_weights(cov)
    elif scheme == "budget":
        # risk budgets proportional to IS Sharpe (floored), ERC-style solve
        w = erc_weights(cov)
        b = pd.Series(budgets).reindex(cov.index)
        for _ in range(500):
            mrc = cov.values @ w.values
            rc = w.values * mrc
            ratio = (b.values / b.sum()) / np.maximum(rc / rc.sum(), 1e-12)
            w = pd.Series(np.clip(w.values * ratio ** 0.3, 1e-6, None), index=cov.index)
            w = w / w.sum()
    elif scheme == "mv":
        mu = win.mean() * 252
        S = cov * 252
        Ssh = shrink * np.diag(np.diag(S)) + (1 - shrink) * S.values
        raw = np.linalg.solve(Ssh, mu.values)
        raw = np.clip(raw, 0, None)
        w = pd.Series(raw / raw.sum(), index=cov.index)
    return w

CORE = {"VAN": van, "ORI": ori, "CRY": cry, "REV": rev, "TOM": tomr, "BND": bnd}
CORE_H = {**CORE, "HEL": hel}

is_sr = {k: sharpe(v.loc[:IS_END_TS]) for k, v in CORE.items()}
print("\nIS sleeve Sharpes:", {k: round(v, 2) for k, v in is_sr.items()})

print("\n== composition x weight scheme (tv=16%) ==")
for label, sl in [("core6 (noHEL)", CORE), ("core7 (+HEL)", CORE_H)]:
    for scheme in ["erc", "budget", "mv"]:
        budgets = {k: max(sharpe(v.loc[:IS_END_TS]), 0.3) for k, v in sl.items()}
        w = fit_weights(sl, scheme, budgets)
        net, mult = blend(sl, w)
        rep_is(f"{label} {scheme}", net,
               "  w=" + ",".join(f"{k}:{v:.2f}" for k, v in w.items()))

print("\n== tv grid on core6 budget ==")
budgets = {k: max(sharpe(v.loc[:IS_END_TS]), 0.3) for k, v in CORE.items()}
w6 = fit_weights(CORE, "budget", budgets)
for tv in [0.16, 0.18, 0.20, 0.22]:
    net, mult = blend(CORE, w6, tv)
    rep_is(f"core6 budget tv={tv}", net, f"  avg_mult={mult.loc[:IS_END_TS].mean():.2f}")

print("\n== overlay ablation on core6 budget tv=0.18 ==")
df6 = pd.concat(CORE, axis=1).reindex(DATES).fillna(0.0)
raw6 = df6 @ w6.reindex(df6.columns).fillna(0.0)
rep_is("raw (no overlay)", raw6)
net, mult = blend(CORE, w6, 0.18)
rep_is("full overlay", net)
