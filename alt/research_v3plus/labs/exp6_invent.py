"""Exp6: inventions.
A) HYG-gated GH rotation: exp1 GH book when credit healthy; defensive leg
   (best-momentum of TMF/UGL, or BIL) when HYG < 100dma.
B) Vol-targeted GH book (target 20%/25% ann vol, gross<=1).
C) Rank acceleration: rank change of GH proximity over 21d added to level rank.
D) Dispersion-sized GH: gross scaled by cross-sectional signal dispersion.
"""
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from lib import *

UMAP = {"QQQ":"TQQQ","SPY":"UPRO","SMH":"SOXL","XLK":"TECL","XLF":"FAS","XLE":"ERX",
        "EEM":"EDC","FXI":"YINN","VNQ":"DRN","USO":"UCO","GLD":"UGL","TLT":"TMF"}
UND = list(UMAP); LETF = list(UMAP.values())
closes = panel(UND + ["HYG"], "Close")
opens = panel(LETF + ["BIL"], "Open")
cund = closes[UND]

prox = cund / cund.rolling(252).max()
hyg_ok = (closes["HYG"] > closes["HYG"].rolling(100).mean())

def gh_book(K=3, step=5):
    s = prox.shift(1)
    sel = (s.rank(axis=1, ascending=False) <= K).div(K)
    Wu = hold_between(sel.fillna(0.0), monthly_mask(s.index, step))
    return Wu.rename(columns=UMAP).reindex(columns=LETF).fillna(0.0)

def finish(W, name):
    W = W.copy()
    W["BIL"] = (1.0 - W.sum(axis=1)).clip(lower=0.0)
    return evaluate(W, opens, name)

results = []

# --- A) HYG-gated GH rotation ---
g = hyg_ok.shift(1).fillna(False).astype(float)
Wgh = gh_book(3, 5)
# risk-off -> BIL
results.append(finish(Wgh.mul(g, axis=0), "e6_ghK3s5_hyggate_bil")[0])
# risk-off -> defensive momentum: best 63d mom of TLT/GLD via TMF/UGL (else BIL)
dmom = cund[["TLT","GLD"]].pct_change(63).shift(1)
dmom = dmom.fillna(-9.0)
best = dmom.idxmax(axis=1)
def_ok = dmom.max(axis=1) > 0
Wdef = pd.DataFrame(0.0, index=cund.index, columns=LETF)
Wdef["TMF"] = ((best == "TLT") & def_ok).astype(float)
Wdef["UGL"] = ((best == "GLD") & def_ok).astype(float)
Wdef = hold_between(Wdef, monthly_mask(cund.index, 5))
for dw in (0.5, 1.0):
    W = Wgh.mul(g, axis=0) + Wdef.mul((1 - g) * dw, axis=0)
    results.append(finish(W, f"e6_ghK3s5_hyggate_def{int(dw*10)}")[0])
# gate only the equity-like slots, keep TMF/UGL slots from GH selection
eqL = [l for u, l in UMAP.items() if u not in ("TLT", "GLD")]
Wmix = Wgh.copy(); Wmix[eqL] = Wmix[eqL].mul(g, axis=0)
results.append(finish(Wmix, "e6_ghK3s5_hyggate_eqonly")[0])

# --- B) vol-targeted GH book ---
o2o = opens[LETF].pct_change()
for tgt in (0.20, 0.25):
    Wraw = gh_book(3, 5)
    bookret = (Wraw.shift(1) * o2o).sum(axis=1)          # info to t; shift below
    rv = bookret.rolling(20).std() * np.sqrt(252)
    lev = (tgt / rv).clip(upper=1.0).shift(1).fillna(0.0)  # decision uses ret up to open[t-1]< close[t-1]
    results.append(finish(Wraw.mul(lev, axis=0), f"e6_ghK3s5_vt{int(tgt*100)}")[0])
# vol target + HYG gate combined
Wvt = gh_book(3, 5)
bookret = (Wvt.shift(1) * o2o).sum(axis=1)
lev = (0.25 / (bookret.rolling(20).std() * np.sqrt(252))).clip(upper=1.0).shift(1).fillna(0.0)
results.append(finish(Wvt.mul(lev * g, axis=0), "e6_ghK3s5_vt25_hyg")[0])

# --- C) rank acceleration ---
rk = prox.rank(axis=1, ascending=False)
accel = (rk.shift(21) - rk)          # positive = improving rank
for lam in (0.5, 1.0):
    sig = -rk + lam * accel          # higher better
    s = sig.shift(1)
    sel = (s.rank(axis=1, ascending=False) <= 3).div(3)
    W = hold_between(sel.fillna(0.0), monthly_mask(s.index, 5)).rename(columns=UMAP)
    W = W.reindex(columns=LETF).fillna(0.0)
    results.append(finish(W, f"e6_ghaccel{int(lam*10)}_K3_s5")[0])

# --- D) dispersion-sized GH ---
disp = prox.std(axis=1)
z = ((disp - disp.rolling(252).mean()) / disp.rolling(252).std()).shift(1)
for mode in ("hi", "lo"):
    scale = (0.5 + 0.25 * (z if mode == "hi" else -z)).clip(0.0, 1.0).fillna(0.5)
    W = gh_book(3, 5).mul(scale, axis=0)
    results.append(finish(W, f"e6_ghdisp_{mode}")[0])

pd.DataFrame(results).to_csv(f"{SCRATCH}/e6_results.csv", index=False)
