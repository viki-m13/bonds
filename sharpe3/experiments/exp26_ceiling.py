"""Exp26: quantify the ceiling — the fundamental law of active management,
measured rather than asserted.

Sharpe_gross = IC x sqrt(breadth). We measure, on the clean era (2015+):
  1. the realized rank-IC of every signal at h = 1,5,21d
  2. the IC of the best walk-forward linear combination of all of them
  3. the effective breadth of the S&P 500 cross-section (accounting for the
     correlation of residual returns, which is what actually limits bets)
  4. the implied max gross Sharpe, and what IC would be needed for Sharpe 3

This converts the verdict from "we couldn't find it" into "here is the
arithmetic of why it is not there."
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P0 = datalib.load_summit()
P = {k: v.loc["2015":] for k, v in P0.items()}
close, open_, volp, member = P["close"], P["open"], P["volume"], P["member"]
r1 = close.pct_change(fill_method=None)
intraday = close / open_ - 1
overnight = open_ / close.shift(1) - 1
vol20 = r1.rolling(20).std()
dv = (close * volp).rolling(20, min_periods=5).median()

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

sigs = {
    "rev1d": -zs(r1 / vol20),
    "rev5d": -zs(close.pct_change(5, fill_method=None)),
    "rev21d": -zs(close.pct_change(21, fill_method=None)),
    "rev_i5": -zs(intraday.rolling(5).sum()),
    "onmom252": zs(overnight.rolling(252).mean() / overnight.rolling(252).std()),
    "mom12_1": zs(close.shift(21).pct_change(231, fill_method=None)),
    "vol60": -zs(r1.rolling(60).std()),
    "dvrank": zs(dv.rank(axis=1, pct=True)),
    "d200ma": zs(close / close.rolling(200).mean() - 1),
    "skew63": zs(r1.rolling(63).skew()),
}

o = open_
out = {"ic": {}}
for h in (1, 5, 21):
    fwd = (o.shift(-(h + 1)) / o.shift(-1) - 1)
    y = fwd.sub(fwd.where(member).mean(axis=1), axis=0).where(member)
    row = {}
    for nm, s in sigs.items():
        ic = s.where(member).corrwith(y, axis=1, method="spearman").dropna()
        if len(ic) > 100:
            row[nm] = {"ic": round(float(ic.mean()), 4),
                       "t": round(float(ic.mean() / ic.std() * np.sqrt(len(ic))), 1)}
    out["ic"][f"h{h}"] = row
    print(f"--- horizon {h}d rank-IC (2015+) ---")
    for nm, v in sorted(row.items(), key=lambda kv: -abs(kv[1]["ic"])):
        print(f"  {nm:12s} IC={v['ic']:+.4f} t={v['t']:+5.1f}")

# best walk-forward linear combination (annual refit, IC-weighted)
h = 5
fwd = (o.shift(-(h + 1)) / o.shift(-1) - 1)
y = fwd.sub(fwd.where(member).mean(axis=1), axis=0).where(member)
names = list(sigs)
comb = pd.DataFrame(0.0, index=close.index, columns=close.columns)
for yr in range(2017, 2027):
    tr = close.index < pd.Timestamp(f"{yr}-01-01")
    te = (close.index >= pd.Timestamp(f"{yr}-01-01")) & (close.index < pd.Timestamp(f"{yr+1}-01-01"))
    if te.sum() == 0:
        continue
    wts = {}
    for nm in names:
        ic = sigs[nm][tr].where(member[tr]).corrwith(y[tr], axis=1, method="spearman").dropna()
        wts[nm] = float(ic.mean()) if len(ic) > 100 else 0.0
    tot = sum(abs(v) for v in wts.values()) or 1.0
    acc = sum(sigs[nm].loc[te].fillna(0.0) * (wts[nm] / tot) for nm in names)
    comb.loc[te] = acc.values
comb = comb.where(member)
ic_c = comb.corrwith(y, axis=1, method="spearman").dropna()
ic_c = ic_c[ic_c.index >= "2017-01-01"]
out["combo_ic_h5"] = {"ic": round(float(ic_c.mean()), 4),
                      "t": round(float(ic_c.mean() / ic_c.std() * np.sqrt(len(ic_c))), 1)}
print("walk-forward combo IC (h5, 2017+):", out["combo_ic_h5"])

# effective breadth: average pairwise correlation of residual (market-adj) returns
resid = r1.where(member).sub(r1.where(member).mean(axis=1), axis=0)
sub = resid.loc["2018":].dropna(axis=1, thresh=1000)
C = sub.corr().values
n = C.shape[0]
rho = (C.sum() - n) / (n * (n - 1))
N_eff = n / (1 + (n - 1) * rho)
out["breadth"] = {"n_names": int(n), "avg_resid_corr": round(float(rho), 4),
                  "N_eff": round(float(N_eff), 1)}
print("names:", n, "avg residual corr:", round(rho, 4), "N_eff:", round(N_eff, 1))

# Implied Sharpe via the fundamental law. Breadth must be holding-period
# adjusted: a signal held h days does NOT deliver a fresh independent bet every
# day, so bets/year = N_eff x (252/h), not N_eff x 252. This is the THEORETICAL
# ceiling — it assumes a perfect risk model, optimal weights and zero costs, so
# it is an upper bound on what any construction could extract from this IC.
for h in (1, 5, 21):
    row = out["ic"].get(f"h{h}", {})
    if not row:
        continue
    best = max(row, key=lambda k: abs(row[k]["ic"]))
    icv = abs(row[best]["ic"])
    breadth = N_eff * (252.0 / h)
    theo = icv * np.sqrt(breadth)
    out.setdefault("implied", {})[f"h{h}"] = {
        "best_signal": best, "ic_used": round(icv, 4),
        "independent_bets_per_year": round(float(breadth), 0),
        "theoretical_max_gross_sharpe": round(float(theo), 2),
        "ic_needed_for_SR3": round(float(3 / np.sqrt(breadth)), 4),
        "ic_multiple_needed": round(float((3 / np.sqrt(breadth)) / icv), 1)}
    print(f"h{h}: best={best} IC {icv:.4f} x sqrt({breadth:.0f}) -> THEORETICAL max "
          f"gross SR {theo:.2f} | IC needed for SR3: {3/np.sqrt(breadth):.4f} "
          f"({(3/np.sqrt(breadth))/icv:.1f}x what exists)")

json.dump(out, open(os.path.join(ROOT, "results", "exp26_ceiling.json"), "w"), indent=1)
print("saved")
