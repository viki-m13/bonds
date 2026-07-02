"""Blend assembly lab — IS ONLY. ERC weights, EWMA vol-target overlay,
DD throttle, vol gate; strict shift(2) implementability."""
import sys
sys.path.insert(0, "/home/user/bonds/alt")
import numpy as np
import pandas as pd
from sleeve_engine import backtest_weights, sharpe, cagr, max_dd, ann_vol
from research_lab import (van, ori, hel, cry, reversal, tom, bondls, convex,
                          DATES, IS_END, load_etf)

IS_END_TS = pd.Timestamp(IS_END)
ERC_START = "2015-01-02"

rev = reversal(-1.0, 5)
tomr = tom("TQQQ", trend_filter=False)
bnd = bondls(63, None)


def erc_weights(cov: pd.DataFrame, iters=500) -> pd.Series:
    """Long-only equal-risk-contribution via multiplicative updates."""
    n = len(cov)
    w = np.ones(n) / n
    for _ in range(iters):
        mrc = cov.values @ w
        rc = w * mrc
        target = rc.mean()
        w = w * (target / np.maximum(rc, 1e-12)) ** 0.3
        w = np.clip(w, 1e-6, None)
        w = w / w.sum()
    return pd.Series(w, index=cov.index)


def overlay(raw: pd.Series, target_vol=0.16, lam=0.94, cap=1.0, floor=0.25,
            dd_win=252, dd_floor=-0.10, gate_pct=0.99, gate_look=252,
            tc_bps=10.0):
    """EWMA vol target + DD throttle + vol gate, all applied with shift(2):
    the multiplier scaling the return realized over open[t-1]->open[t] is a
    function of raw returns through t-2 (decidable at close[t-2], traded at
    open[t-1])."""
    ew_var = raw.pow(2).ewm(alpha=1 - lam).mean()
    ew_vol = (ew_var * 252) ** 0.5
    vol_mult = (target_vol / ew_vol).clip(floor, cap)

    scaled = raw * vol_mult.shift(2).fillna(1.0)
    cum = (1 + scaled).cumprod()
    hwm = cum.rolling(dd_win, min_periods=30).max()
    dd = cum / hwm - 1
    dd_mult = (1 + dd / dd_floor).clip(0.0, 1.0)

    sv = scaled.rolling(60).std()
    sv_thr = sv.rolling(gate_look, min_periods=60).quantile(gate_pct)
    gate_mult = pd.Series(np.where(sv <= sv_thr, 1.0, 0.5), index=raw.index)

    total = (vol_mult * dd_mult * gate_mult).shift(2).fillna(1.0)
    tc = total.diff().abs().fillna(0) * (tc_bps / 1e4)
    return raw * total - tc, total


def assemble(sleeves: dict, target_vol=0.16, w_scheme="erc"):
    df = pd.concat(sleeves, axis=1).reindex(DATES)
    # NaN -> 0 only before a sleeve's first valid date is wrong; use 0 fill
    # but note every sleeve here spans the full window (cash-when-inactive).
    df = df.fillna(0.0)
    win = df.loc[ERC_START:IS_END_TS]
    if w_scheme == "erc":
        w = erc_weights(win.cov())
    else:
        iv = 1 / win.std()
        w = iv / iv.sum()
    raw = df @ w
    net, mult = overlay(raw, target_vol=target_vol)
    return net, w, mult, raw


def rep_is(name, net, w=None, mult=None, raw=None):
    r = net.loc[:IS_END_TS]
    extra = ""
    if w is not None:
        extra = "  w=" + ",".join(f"{k[:3]}:{v:.2f}" for k, v in w.items())
    if mult is not None:
        extra += f"  avg_mult={mult.loc[:IS_END_TS].mean():.2f}"
    print(f"{name:26s} IS: SR {sharpe(r):5.2f}  CAGR {cagr(r)*100:6.1f}%  "
          f"Vol {ann_vol(r)*100:5.1f}%  MDD {max_dd(r)*100:6.1f}%{extra}")


S4 = {"VAN": van, "ORI": ori, "HEL": hel, "CRY": cry}
print("== Blend compositions (ERC 2015-2018 cov, EWMA overlay tv=16%) ==")
for name, sl in [
    ("4 fixed", S4),
    ("4+REV", {**S4, "REV": rev}),
    ("4+REV+TOM", {**S4, "REV": rev, "TOM": tomr}),
    ("4+REV+BND", {**S4, "REV": rev, "BND": bnd}),
    ("4+REV+TOM+BND", {**S4, "REV": rev, "TOM": tomr, "BND": bnd}),
    ("3(noHEL)+REV", {k: v for k, v in S4.items() if k != "HEL"} | {"REV": rev}),
]:
    net, w, mult, raw = assemble(sl)
    rep_is(name, net, w, mult)

print("\n== weight scheme comparison on 4+REV ==")
for scheme in ["erc", "invvol"]:
    net, w, mult, _ = assemble({**S4, "REV": rev}, w_scheme=scheme)
    rep_is(f"4+REV {scheme}", net, w, mult)

print("\n== target vol sensitivity on 4+REV (erc) ==")
for tv in [0.14, 0.16, 0.18, 0.20]:
    net, w, mult, _ = assemble({**S4, "REV": rev}, target_vol=tv)
    rep_is(f"tv={tv}", net, None, mult)

print("\n== IS pairwise corr (2015-2018) ==")
allsl = pd.concat({**S4, "REV": rev, "TOM": tomr, "BND": bnd}, axis=1).reindex(DATES).fillna(0.0)
print(allsl.loc[ERC_START:IS_END_TS].corr().round(2).to_string())
