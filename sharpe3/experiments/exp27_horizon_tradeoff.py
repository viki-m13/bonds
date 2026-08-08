"""Exp27: the horizon trap, quantified.

Exp26 showed the theoretical (zero-cost, perfect-construction) Sharpe ceiling
RISES as the holding period shortens, because breadth = N_eff x (252/h).
But turnover — and therefore cost drag — rises faster.

This experiment computes, for each horizon h, on the clean era (2015+):
  - theoretical max gross Sharpe from the measured IC (fundamental law)
  - the REALIZED gross Sharpe of the actual portfolio
  - realized annual turnover and the resulting cost drag in Sharpe units
  - net Sharpe at 2 / 5 / 10 bps

The result is the trap: horizons with enough breadth to reach 3 cannot pay
their transaction costs; horizons whose costs are affordable lack the breadth.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P0 = datalib.load_summit()
P = {k: v.loc["2015":] for k, v in P0.items()}
close, open_, member = P["close"], P["open"], P["member"]
r1 = close.pct_change(fill_method=None)
o = open_

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

# N_eff from exp26 (recomputed here so the script stands alone)
resid = r1.where(member).sub(r1.where(member).mean(axis=1), axis=0)
sub = resid.loc["2018":].dropna(axis=1, thresh=1000)
C = sub.corr().values
n = C.shape[0]
rho = (C.sum() - n) / (n * (n - 1))
N_eff = n / (1 + (n - 1) * rho)
print(f"N_eff = {N_eff:.1f} (n={n}, avg resid corr={rho:.4f})\n")

rows = {}
for h in (1, 2, 5, 10, 21, 63):
    # signal: h-day reversal (the family with the most consistent IC)
    sig = -zs(close.pct_change(h, fill_method=None))
    # smooth over the holding period so the portfolio actually holds h days
    if h > 1:
        sig = sig.rolling(h).mean()
    fwd = (o.shift(-(h + 1)) / o.shift(-1) - 1)
    y = fwd.sub(fwd.where(member).mean(axis=1), axis=0).where(member)
    ic = sig.where(member).corrwith(y, axis=1, method="spearman").dropna()
    icv = abs(float(ic.mean()))
    breadth = N_eff * (252.0 / h)
    theo = icv * np.sqrt(breadth)

    w = bt.norm_ls(sig, member, 0.1, 0.1, 2.0)
    res0 = bt.run(w, P, mode="open", cost_bps=0.0)
    gross_sr = bt.sharpe(res0["gross"])
    ann_to = float(res0["turnover"].mean()) * 252
    vol = float(res0["gross"].std() * np.sqrt(252))
    nets = {}
    for c in (2.0, 5.0, 10.0):
        r = bt.run(w, P, mode="open", cost_bps=c)["net"]
        nets[c] = round(bt.sharpe(r), 2)
    # cost drag expressed in Sharpe units at 5bps
    drag5 = (ann_to * 5 / 1e4) / vol if vol > 0 else np.nan

    rows[h] = {"ic": round(icv, 4), "t_stat": round(float(ic.mean() / ic.std() * np.sqrt(len(ic))), 1),
               "independent_bets_per_yr": round(float(breadth)),
               "theoretical_max_gross_sharpe": round(float(theo), 2),
               "realized_gross_sharpe": round(gross_sr, 2),
               "annual_turnover_x": round(ann_to, 1),
               "ann_vol": round(vol, 3),
               "cost_drag_sharpe_at_5bps": round(float(drag5), 2),
               "net_sharpe": nets}
    print(f"h={h:2d}d  IC={icv:.4f}(t={rows[h]['t_stat']:+.1f})  "
          f"theoryMax={theo:5.2f}  realizedGross={gross_sr:5.2f}  "
          f"turnover={ann_to:6.0f}x/yr  costDrag@5bps={drag5:5.2f}SR  "
          f"net: 2bp={nets[2.0]:6.2f} 5bp={nets[5.0]:6.2f} 10bp={nets[10.0]:6.2f}")

json.dump({"N_eff": round(float(N_eff), 1), "avg_resid_corr": round(float(rho), 5),
           "horizons": rows},
          open(os.path.join(ROOT, "results", "exp27_horizon_tradeoff.json"), "w"), indent=1)
print("\nsaved")
