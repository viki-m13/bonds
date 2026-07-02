"""Regenerate qualifying candidates, save full-window Date,ret CSVs.
Qualification: (IS SR>=0.5 AND |corr|<=0.5) OR IS SR>=0.9. IS stats only."""
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from lib import *

UMAP = {"QQQ":"TQQQ","SPY":"UPRO","SMH":"SOXL","XLK":"TECL","XLF":"FAS","XLE":"ERX",
        "EEM":"EDC","FXI":"YINN","VNQ":"DRN","USO":"UCO","GLD":"UGL","TLT":"TMF"}
UND = list(UMAP); LETF = list(UMAP.values())
closes = panel(UND + ["HYG"], "Close")
opens12 = panel(LETF + ["BIL"], "Open")
cund = closes[UND]
prox = cund / cund.rolling(252).max()
hyg_ok = closes["HYG"] > closes["HYG"].rolling(100).mean()
g = hyg_ok.shift(1).fillna(False).astype(float)

def gh_book(K=3, step=5):
    s = prox.shift(1)
    sel = (s.rank(axis=1, ascending=False) <= K).div(K)
    Wu = hold_between(sel.fillna(0.0), monthly_mask(s.index, step))
    return Wu.rename(columns=UMAP).reindex(columns=LETF).fillna(0.0)

def finish(W, opens, name):
    W = W.copy()
    W["BIL"] = (1.0 - W.sum(axis=1)).clip(lower=0.0)
    return evaluate(W, opens, name, save=True)

saved = {}

# 1. gh52_letf_rot: GH 52w-high proximity rotation, top-3 of 12, weekly, 3x LETFs
_, saved["gh52_letf_rot"] = finish(gh_book(3, 5), opens12, "gh52_letf_rot")

# 2. gh52_hyggate_eq: same book, equity-like slots gated by HYG>100dma (TMF/UGL kept)
eqL = [l for u, l in UMAP.items() if u not in ("TLT", "GLD")]
Wmix = gh_book(3, 5); Wmix[eqL] = Wmix[eqL].mul(g, axis=0)
_, saved["gh52_hyggate_eq"] = finish(Wmix, opens12, "gh52_hyggate_eq")

# 3. gh52_vt25_hyg: GH book, 25% vol target, HYG gate on whole book
o2o = opens12[LETF].pct_change()
Wvt = gh_book(3, 5)
bookret = (Wvt.shift(1) * o2o).sum(axis=1)
lev = (0.25 / (bookret.rolling(20).std() * np.sqrt(252))).clip(upper=1.0).shift(1).fillna(0.0)
_, saved["gh52_vt25_hyg"] = finish(Wvt.mul(lev * g, axis=0), opens12, "gh52_vt25_hyg")

# 4. gh52_displo: GH book sized up when cross-sectional proximity dispersion is LOW
disp = prox.std(axis=1)
z = ((disp - disp.rolling(252).mean()) / disp.rolling(252).std()).shift(1)
scale = (0.5 - 0.25 * z).clip(0.0, 1.0).fillna(0.5)
_, saved["gh52_displo"] = finish(gh_book(3, 5).mul(scale, axis=0), opens12, "gh52_displo")

# 5. hyg_lead_sso: SPY exposure via SSO gated by HYG>100dma
opens_s = panel(["SSO", "TQQQ", "BIL"], "Open")
W = pd.DataFrame(0.0, index=closes.index, columns=["SSO"])
W["SSO"] = g
_, saved["hyg_lead_sso"] = finish(W, opens_s, "hyg_lead_sso")

# 6. smh_lead_tqqq: TQQQ held when SMH 10d momentum > 0 (semis lead QQQ)
smh10 = (closes["SMH"].pct_change(10) > 0).shift(1).fillna(False).astype(float)
W = pd.DataFrame({"TQQQ": smh10})
_, saved["smh_lead_tqqq"] = finish(W, opens_s, "smh_lead_tqqq")

# 7. sector_gh_hedge: sector GH proximity top-3 (TECL/FAS/ERX where levered),
#    SH hedge sized by breadth (frac sectors <200dma), hmax=0.3, monthly
SECT = ["XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLU","XLB"]
LMAP = {"XLK":"TECL","XLF":"FAS","XLE":"ERX"}
cs = panel(SECT, "Close")
TRADE = ["TECL","FAS","ERX","XLV","XLY","XLP","XLI","XLU","XLB","SH","BIL"]
opens_sec = panel(TRADE, "Open")
breadth = (cs < cs.rolling(200).mean()).mean(axis=1)
sp = (cs / cs.rolling(252).max()).shift(1)
b = breadth.shift(1).fillna(0.0)
reb = monthly_mask(sp.index, 21)
sel = (sp.rank(axis=1, ascending=False) <= 3).astype(float)
h = hold_between((0.3 * b).to_frame("h"), reb)["h"]
Wu = hold_between(sel.div(3), reb).mul(1.0 - h, axis=0)
Wsec = Wu.rename(columns=LMAP).reindex(columns=TRADE).fillna(0.0)
Wsec["SH"] = h
Wsec["BIL"] = (1.0 - Wsec.drop(columns="BIL").sum(axis=1)).clip(lower=0.0)
st, r = evaluate(Wsec, opens_sec, "sector_gh_hedge", save=True)
saved["sector_gh_hedge"] = r

# pairwise IS correlations among saved candidates + phoenix
mat = pd.DataFrame({k: v.loc[:IS_END] for k, v in saved.items()})
mat["phoenix"] = PHX_IS
print("\nIS pairwise correlations:")
print(mat.corr().round(2).to_string())
