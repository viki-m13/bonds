"""ALL-WEATHER + CRISIS-ALPHA TREND OVERLAY on the bond/gold diversifier sleeve.

Thesis: the leveraged risk-parity all-weather (TQQQ + TMF_R + UGL_R) wins the
tech crashes (dot-com, GFC, 2022) but bleeds ~0.5x vs QQQ-DCA over the full
sample because holding leveraged bonds/gold is pure drag in the 2010s / 2023-25
tech bull. Bonds were a great diversifier 2000-2020 (falling yields) and a
DISASTER in 2022 (rising yields). A TREND FILTER on the bond sleeve should have
(a) exited leveraged bonds in the 2022 rising-rate regime, and (b) stayed OUT of
bonds during the 2010s bull, routing that budget back to the tech core -> less
full-sample drag, while KEEPING the dot-com/2008 wins (bonds trended up then).

We reuse the VERIFIED riskparity harness (close, retd, valid_start, month_grid,
run, qqq_dca, maxdd, ERAS). Convention matches riskparity: run() is a forward-buy
loop -- weights at dt are computed from data THROUGH dt (the close at dt), the
trade executes at dt's close, and earns dt->dt+1. All trend signals here are
likewise evaluated at dt using data available at dt (no future data), so there is
NO look-ahead and we do NOT add an extra .shift (matching rp_weights).
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import riskparity as rp
from riskparity import (close, retd, valid_start, month_grid, run, qqq_dca,
                        maxdd, ERAS)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- trend signals on the diversifier sleeve (evaluated at each grid date) ----
def ma_gate(t, mg, win=200):
    """1.0 where close[t] at dt >= its own trailing `win`-day MA at dt, else 0.
    Uses only data through dt (rolling MA of past `win` closes incl. dt)."""
    ma = close[t].rolling(win, min_periods=int(win*0.6)).mean()
    sig = (close[t] >= ma).astype(float)
    return sig.reindex(mg, method="ffill").fillna(1.0)   # default ON before history

def mom_gate(t, mg, lb=252):
    """1.0 where trailing `lb`-day total return of t at dt > 0, else 0."""
    m = close[t]/close[t].shift(lb) - 1.0
    sig = (m > 0).astype(float)
    return sig.reindex(mg, method="ffill").fillna(1.0)

def aw_weights(mg, assets, gated, target=0.14, cap=1.0, volwin=63, mode="invvol",
               fixed=None, gate="ma", gwin=200, route="tech", core="TQQQ",
               gscale=0.0):
    """All-weather weights with a trend gate on the sleeves listed in `gated`.

    Steps: (1) build base risk weights exactly as rp_weights (invvol/momtilt/
    fixed) + vol-target scaling; (2) for each gated sleeve compute a 0/1 (or
    partial via gscale) trend signal; (3) knock the gated fraction down when
    off; (4) route the freed risk budget per `route`:
        'tech' -> proportionally to the non-gated (tech core) risk weights,
                  re-vol-targeted (stay invested in tech during the bull);
        'cash' -> leave it in cash (de-risk);
        'core' -> specifically into `core` (TQQQ).
    gscale in (0,1] keeps a residual bond weight even when gated off (soft gate).
    """
    fixed = fixed or {}
    vol = {t: (retd[t].rolling(volwin, min_periods=int(volwin*0.7)).std()*np.sqrt(252)
               ).reindex(mg, method="ffill") for t in assets}
    vol = pd.DataFrame(vol)
    if mode == "invvol":
        raw = 1.0/vol
    elif mode == "momtilt":
        mom = pd.DataFrame({t: (close[t]/close[t].shift(252)-1).reindex(mg, method="ffill")
                            for t in assets})
        z = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1).replace(0, np.nan), axis=0).fillna(0)
        raw = (1.0/vol) * np.exp(fixed.get("tilt_k", 1.0) * z)
    else:
        raw = pd.DataFrame({t: pd.Series(fixed[t], index=mg) for t in assets})
    for t in assets:
        raw.loc[mg < valid_start[t], t] = np.nan

    # trend gate multiplier per gated sleeve
    gm = pd.DataFrame(1.0, index=mg, columns=assets)
    for t in gated:
        if t not in assets: continue
        g = ma_gate(t, mg, gwin) if gate == "ma" else mom_gate(t, mg, gwin)
        gm[t] = gscale + (1.0 - gscale)*g          # gscale..1

    raw_g = raw * gm                                # gated raw weights
    if route in ("tech", "core"):
        # renormalize risk weights among what's on -> freed budget flows to
        # the ungated (tech) sleeves automatically.
        rw = raw_g.div(raw_g.sum(axis=1), axis=0)
        if route == "core" and core in assets:
            # force freed budget specifically into core rather than pro-rata
            base = raw.div(raw.sum(axis=1), axis=0)
            freed = (base - rw.where(rw.notna(), base)).clip(lower=0).sum(axis=1)
            # simpler: rw already pro-rata; core route only differs if >1 ungated
        bvol = (rw * vol).sum(axis=1)
        scale = (target / bvol).clip(0, cap)
        W = rw.mul(scale, axis=0).fillna(0.0)
    else:  # cash: keep original normalization+target, then zero gated-off budget
        rw = raw.div(raw.sum(axis=1), axis=0)
        bvol = (rw * vol).sum(axis=1)
        scale = (target / bvol).clip(0, cap)
        W = rw.mul(scale, axis=0)
        W = (W * gm).fillna(0.0)                    # freed fraction -> cash residual
    return W

# ---------------------------------- reporting ---------------------------------
def per_era(name, wfn, nth=None):
    out = []
    for st, en in ERAS:
        s, e = pd.Timestamp(st+"-01"), pd.Timestamp(en+"-01")
        W = wfn(month_grid(nth))
        eq = run(s, e, W, nth=nth); b = qqq_dca(s, e, nth=nth)
        out.append(eq["V"].iloc[-1]/b["V"].iloc[-1])
    # explicit 2022
    s, e = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")
    W = wfn(month_grid(nth)); eq = run(s, e, W, nth=nth); b = qqq_dca(s, e, nth=nth)
    out.append(eq["V"].iloc[-1]/b["V"].iloc[-1])
    print(f"{name:26}" + "".join(f"{v:>7.2f}" for v in out))
    return out

def phase(name, wfn, span=("2000-01-01","2026-07-01")):
    s, e = pd.Timestamp(span[0]), pd.Timestamp(span[1])
    print(f"  {name}")
    for nth in [None, 4, 9, 14]:
        W = wfn(month_grid(nth)); eq = run(s, e, W, nth=nth); b = qqq_dca(s, e, nth=nth)
        lab = "ME" if nth is None else "d"+str(nth)
        print(f"    {lab:4}: {eq['V'].iloc[-1]/b['V'].iloc[-1]:.2f}x  maxDD {maxdd(eq):.0%}")

HDR = ["dotcom","00-10","GFC","10-14","15-19","20-26","full00","full06","2022"]

def gate_frac(sleeve="TMF_R", win=200):
    """diagnostic: fraction of each regime the MA-gate would hold the sleeve ON."""
    mg = month_grid(None); g = ma_gate(sleeve, mg, win)
    for a, b, lab in [("2000","2002","dotcom"),("2008","2009","GFC"),
                      ("2010","2020","2010s bull"),("2022","2022","2022"),
                      ("2023","2026","23-26")]:
        m = (g.index >= pd.Timestamp(a+"-01-01")) & (g.index <= pd.Timestamp(b+"-12-31"))
        print(f"    {lab:12} {sleeve} above MA{win}: {g[m].mean():.2f} ON")

if __name__ == "__main__":
    A = ["TQQQ","TMF_R","UGL_R"]
    mk = lambda **kw: (lambda mg: aw_weights(mg, A, ["TMF_R","UGL_R"], **kw))
    mkrp = lambda **kw: (lambda mg: rp.rp_weights(mg, A, **kw))

    print("RATIO vs QQQ-DCA  (all-weather variants)")
    print(f"{'config':26}" + "".join(f"{h:>7}" for h in HDR))
    per_era("BASE momtilt+gold t22", mkrp(target=0.22, mode="momtilt", fixed={"tilt_k":1.0}))
    print("--- H1: trend-GATED bond/gold sleeve (the primary bet) ---")
    per_era("gate-ma200 -> TECH  t22", mk(target=0.22, mode="momtilt", fixed={"tilt_k":1.0}, gate="ma", gwin=200, route="tech"))
    per_era("gate-ma200 -> CASH  t22", mk(target=0.22, mode="momtilt", fixed={"tilt_k":1.0}, gate="ma", gwin=200, route="cash"))
    per_era("gate-mom12 -> TECH  t22", mk(target=0.22, mode="momtilt", fixed={"tilt_k":1.0}, gate="mom", gwin=252, route="tech"))
    print("--- WINNER: crank the CONTINUOUS momentum tilt (same leverage as base) ---")
    per_era("momtilt k2 t22", mkrp(target=0.22, mode="momtilt", fixed={"tilt_k":2.0}))
    per_era("momtilt k3 t22", mkrp(target=0.22, mode="momtilt", fixed={"tilt_k":3.0}))
    print("--- + honest leverage (higher vol target) on top ---")
    per_era("momtilt k2 t26", mkrp(target=0.26, mode="momtilt", fixed={"tilt_k":2.0}))
    per_era("momtilt k3 t30", mkrp(target=0.30, mode="momtilt", fixed={"tilt_k":3.0}))

    print("\nWHY THE GATE FAILS  (bonds trended UP in crises AND the 2010s bull):")
    gate_frac("TMF_R", 200)

    print("\nPHASE ROBUSTNESS (full 2000-2026, ratio/maxDD at ME,d4,d9,d14):")
    phase("BASE momtilt+gold t22", mkrp(target=0.22, mode="momtilt", fixed={"tilt_k":1.0}))
    phase("gate-ma200 tech t22", mk(target=0.22, mode="momtilt", fixed={"tilt_k":1.0}, gate="ma", gwin=200, route="tech"))
    phase("momtilt k3 t22 (winner, same leverage)", mkrp(target=0.22, mode="momtilt", fixed={"tilt_k":3.0}))
    phase("momtilt k2 t26 (winner + leverage)", mkrp(target=0.26, mode="momtilt", fixed={"tilt_k":2.0}))
