"""Idea 5: macro state-conditional inventions.
5A Bond-book state machine: rate momentum (DGS10 63d chg) x curve (T10Y2Y vs 63d ago)
   -> {TMF / TYD / IEF / BIL} books. FRED signals get an EXTRA lag day (shift 2 total)
   to be conservative about publication timing.
5B Real-rate gold: 10y real yield (DGS10 - T10YIE) 63d momentum falling -> UGL/GLD.
5C HY-OAS credit regime x breadth tiers sizing an equity LETF sleeve:
   OAS below 200d mean (credit calm) + risk-on breadth count -> tiered TQQQ/QLD size.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from common import *

TICKS = ["TMF","TYD","UBT","IEF","TLT","HYG","UGL","GLD","TQQQ","QLD","SSO","BIL"]
opens = panel(TICKS + ["SPY","EEM"], "Open")
closes = panel(TICKS + ["SPY","EEM"], "Close")

dgs10 = load_fred("DGS10").reindex(CAL).ffill()
t10y2y = load_fred("T10Y2Y").reindex(CAL).ffill()
t10yie = load_fred("T10YIE").reindex(CAL).ffill()
oas = load_fred("BAMLH0A0HYM2").reindex(CAL).ffill()

FLAG = 2  # extra-conservative lag for FRED signals (data <= close[t-2])

# ---------- 5A bond state machine ----------
rate_mom = dgs10.diff(63)          # >0 rates rising
curve_mom = t10y2y.diff(63)        # >0 steepening

def bond_machine(books, mom_thr=0.0, tag=""):
    st = pd.Series(index=CAL, dtype=object)
    fall = rate_mom < -mom_thr
    rise = rate_mom > mom_thr
    steep = curve_mom > 0
    for i, t in enumerate(CAL):
        if np.isnan(rate_mom.get(t, np.nan)): continue
        if fall[t]:
            st[t] = "FS" if steep[t] else "FF"   # falling+steep / falling+flat
        elif rise[t]:
            st[t] = "RS" if steep[t] else "RF"
        else:
            st[t] = "N"
    W = pd.DataFrame(0.0, index=CAL, columns=TICKS)
    for s, book in books.items():
        m = (st == s)
        for tk, w in book.items():
            W.loc[m, tk] = w
    return W.shift(FLAG).fillna(0.0)

books_a = {
  "FS": {"TMF": 0.5, "BIL": 0.5},   # rates falling, curve steepening: long duration lever
  "FF": {"TMF": 0.35, "IEF": 0.3, "BIL": 0.35},
  "RS": {"HYG": 0.4, "BIL": 0.6},   # rates rising + steep = growth reflation: credit
  "RF": {"BIL": 1.0},               # bear flattener: cash
  "N":  {"IEF": 0.5, "BIL": 0.5},
}
books_b = {
  "FS": {"TMF": 0.6, "BIL": 0.4},
  "FF": {"TYD": 0.5, "BIL": 0.5},
  "RS": {"HYG": 0.5, "BIL": 0.5},
  "RF": {"BIL": 1.0},
  "N":  {"TYD": 0.4, "BIL": 0.6},
}
evaluate(bond_machine(books_a), opens, "V5Aa_bondmachine")
evaluate(bond_machine(books_b), opens, "V5Ab_bondmachine_tyd")
evaluate(bond_machine(books_a, mom_thr=0.10), opens, "V5Ac_deadband10bp")
evaluate(bond_machine(books_b, mom_thr=0.10), opens, "V5Ad_tyd_deadband")

# ---------- 5B real-rate gold ----------
rr = (dgs10 - t10yie)
rr_mom = rr.diff(63)
for thr, asset, wt, nm in [(0.0,"UGL",0.5,"V5Ba_UGL50"), (0.0,"GLD",1.0,"V5Bb_GLD100"),
                            (0.10,"UGL",0.5,"V5Bc_UGL_db10"), (0.0,"UGL",0.7,"V5Bd_UGL70")]:
    sig = (rr_mom < -thr).astype(float)
    W = pd.DataFrame(0.0, index=CAL, columns=TICKS)
    W[asset] = sig * wt
    W["BIL"] = 1.0 - sig * wt
    W = W.shift(FLAG).fillna(0.0)
    evaluate(W, opens, nm)

# 5B': require also gold price trend confirm (GLD > 100dma)
gld_tr = closes["GLD"] > closes["GLD"].rolling(100).mean()
for asset, wt, nm in [("UGL",0.5,"V5Be_UGL_trconf"), ("UGL",0.7,"V5Bf_UGL70_trconf")]:
    sig = ((rr_mom < 0) & gld_tr).astype(float)
    W = pd.DataFrame(0.0, index=CAL, columns=TICKS)
    W[asset] = sig * wt
    W["BIL"] = 1.0 - sig * wt
    W = W.shift(FLAG).fillna(0.0)
    evaluate(W, opens, nm)

# ---------- 5C OAS regime x breadth tiers ----------
oas_calm = oas < oas.rolling(200).mean()
spy_tr = closes["SPY"] > closes["SPY"].rolling(200).mean()
hyg_tr = closes["HYG"] > closes["HYG"].rolling(100).mean()
eem_tr = closes["EEM"] > closes["EEM"].rolling(200).mean()
breadth = spy_tr.astype(int) + hyg_tr.astype(int) + eem_tr.astype(int)

def oas_breadth(tiers, asset_hi="TQQQ", asset_mid="QLD"):
    """tiers: dict breadth_count -> weight (applied only when OAS calm)."""
    W = pd.DataFrame(0.0, index=CAL, columns=TICKS)
    for b, w in tiers.items():
        m = oas_calm & (breadth == b)
        a = asset_hi if b == 3 else asset_mid
        W.loc[m, a] = w
    W["BIL"] = 1.0 - W.drop(columns="BIL").sum(axis=1)
    return W.shift(FLAG).fillna(0.0)

evaluate(oas_breadth({3: 0.6, 2: 0.4, 1: 0.2}), opens, "V5Ca_tier642")
evaluate(oas_breadth({3: 0.5, 2: 0.3, 1: 0.0}), opens, "V5Cb_tier530")
evaluate(oas_breadth({3: 0.7, 2: 0.4, 1: 0.0}), opens, "V5Cc_tier740")

# ---------- 5C refinements ----------
def oas_breadth2(tiers, defensive=None, rebal=5, asset_hi="TQQQ", asset_mid="QLD"):
    Wd = pd.DataFrame(0.0, index=CAL, columns=TICKS)
    for b, w in tiers.items():
        m = oas_calm & (breadth == b)
        a = asset_hi if b == 3 else asset_mid
        Wd.loc[m, a] = w
    if defensive:
        m = ~oas_calm
        for tk, w in defensive.items():
            Wd.loc[m, tk] = w
    # weekly snap to reduce turnover
    W = pd.DataFrame(np.nan, index=CAL, columns=TICKS)
    for i in range(0, len(CAL), rebal):
        W.iloc[i] = Wd.iloc[i]
    W = W.ffill().fillna(0.0)
    W["BIL"] = 1.0 - W.drop(columns="BIL").sum(axis=1)
    return W.shift(FLAG).fillna(0.0)

evaluate(oas_breadth2({3:0.5, 2:0.3, 1:0.0}), opens, "V5Cd_weekly")
evaluate(oas_breadth2({3:0.5, 2:0.3, 1:0.0}, defensive={"TMF":0.3}), opens, "V5Ce_wk_defTMF")
evaluate(oas_breadth2({3:0.5, 2:0.3, 1:0.0}, defensive={"TMF":0.25,"UGL":0.1}), opens, "V5Cf_wk_defTMFUGL")
evaluate(oas_breadth2({3:0.6, 2:0.35, 1:0.0}, defensive={"TMF":0.3}), opens, "V5Cg_wk_def_agg")
evaluate(oas_breadth2({3:0.5, 2:0.3, 1:0.0}, defensive={"TMF":0.3}, rebal=10), opens, "V5Ch_10d_defTMF")

evaluate(oas_breadth2({3:0.5, 2:0.3, 1:0.0}, defensive={"TMF":0.3}, rebal=10),
         opens, "SAVE_credit_breadth_def", save="credit_breadth_def")
evaluate(oas_breadth2({3:0.5, 2:0.3, 1:0.0}), opens,
         "SAVE_credit_breadth", save="credit_breadth")
