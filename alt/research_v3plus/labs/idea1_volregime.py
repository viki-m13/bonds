"""Idea 1: Vol-regime state machine.
States from SPY trailing realized vol percentile x trend (close vs 200dma).
A-priori mini-books per state, no per-state optimization.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from common import *

TICKS = ["TQQQ", "QLD", "SSO", "SPY", "QQQ", "TMF", "UGL", "GLD", "IEF", "BIL"]
opens = panel(TICKS, "Open")
closes = panel(TICKS, "Close")

spy_c = closes["SPY"]
ret = spy_c.pct_change()
vol21 = ret.rolling(21).std() * np.sqrt(252)
# vol percentile over trailing 756d (3y)
volpct = vol21.rolling(756).apply(lambda x: (x[-1] >= x).mean(), raw=True)
sma200 = spy_c.rolling(200).mean()
trend = (spy_c > sma200)

def run_state_machine(books, lo=0.5, hi=0.8, dwell=0, tag=""):
    """books: dict state -> {ticker: w}. states: CU, MU, VU, CD, VD (calm/mid/vol x up/down)"""
    st = pd.Series(index=CAL, dtype=object)
    for t in CAL:
        v, tr = volpct.get(t, np.nan), trend.get(t, np.nan)
        if np.isnan(v):
            st[t] = None; continue
        if tr:
            st[t] = "CU" if v < lo else ("MU" if v < hi else "VU")
        else:
            st[t] = "CD" if v < hi else "VD"
    # dwell filter: require state persist `dwell` days before switching book
    if dwell > 0:
        eff = st.copy()
        cur = None; cnt = 0; pend = None
        vals = st.tolist()
        out = []
        for s in vals:
            if s is None:
                out.append(cur); continue
            if s == cur:
                pend = None; cnt = 0
            else:
                if s == pend:
                    cnt += 1
                else:
                    pend = s; cnt = 1
                if cnt >= dwell:
                    cur = s; pend = None; cnt = 0
            out.append(cur)
        eff = pd.Series(out, index=CAL)
    else:
        eff = st
    W = pd.DataFrame(0.0, index=CAL, columns=TICKS)
    for s, book in books.items():
        m = (eff == s)
        for tk, w in book.items():
            W.loc[m, tk] = w
    W = W.shift(1).fillna(0.0)   # decision lag: state known at close t-1 -> hold from open t
    return W

variants = {
 "V1a_base": dict(books={
    "CU": {"TQQQ": 0.5, "BIL": 0.5},
    "MU": {"QLD": 0.5, "BIL": 0.5},
    "VU": {"SSO": 0.3, "IEF": 0.4, "BIL": 0.3},
    "CD": {"BIL": 1.0},
    "VD": {"TMF": 0.3, "UGL": 0.2, "BIL": 0.5}}, ),
 "V1b_aggressive": dict(books={
    "CU": {"TQQQ": 0.7, "BIL": 0.3},
    "MU": {"QLD": 0.7, "BIL": 0.3},
    "VU": {"SSO": 0.5, "IEF": 0.5},
    "CD": {"IEF": 0.5, "BIL": 0.5},
    "VD": {"TMF": 0.4, "UGL": 0.2, "BIL": 0.4}}, ),
 "V1c_dwell3": dict(books={
    "CU": {"TQQQ": 0.5, "BIL": 0.5},
    "MU": {"QLD": 0.5, "BIL": 0.5},
    "VU": {"SSO": 0.3, "IEF": 0.4, "BIL": 0.3},
    "CD": {"BIL": 1.0},
    "VD": {"TMF": 0.3, "UGL": 0.2, "BIL": 0.5}}, dwell=3),
 "V1d_lo40hi75": dict(books={
    "CU": {"TQQQ": 0.5, "BIL": 0.5},
    "MU": {"QLD": 0.5, "BIL": 0.5},
    "VU": {"SSO": 0.3, "IEF": 0.4, "BIL": 0.3},
    "CD": {"BIL": 1.0},
    "VD": {"TMF": 0.3, "UGL": 0.2, "BIL": 0.5}}, lo=0.4, hi=0.75),
 "V1e_dwell5_agg": dict(books={
    "CU": {"TQQQ": 0.7, "BIL": 0.3},
    "MU": {"QLD": 0.7, "BIL": 0.3},
    "VU": {"SSO": 0.5, "IEF": 0.5},
    "CD": {"IEF": 0.5, "BIL": 0.5},
    "VD": {"TMF": 0.4, "UGL": 0.2, "BIL": 0.4}}, dwell=5),
}

res = []
for name, kw in variants.items():
    W = run_state_machine(**kw, tag=name)
    res.append(evaluate(W, opens, name))

variants2 = {
 "V1f_ballast_dwell3": dict(books={
    "CU": {"TQQQ": 0.4, "TMF": 0.15, "UGL": 0.10, "BIL": 0.35},
    "MU": {"QLD": 0.4, "TMF": 0.15, "UGL": 0.10, "BIL": 0.35},
    "VU": {"SSO": 0.25, "TMF": 0.2, "UGL": 0.10, "IEF": 0.2, "BIL": 0.25},
    "CD": {"TMF": 0.15, "UGL": 0.10, "BIL": 0.75},
    "VD": {"TMF": 0.3, "UGL": 0.2, "BIL": 0.5}}, dwell=3),
 "V1g_bondheavy_dwell3": dict(books={
    "CU": {"TQQQ": 0.35, "TMF": 0.25, "UGL": 0.10, "BIL": 0.30},
    "MU": {"QLD": 0.35, "TMF": 0.25, "UGL": 0.10, "BIL": 0.30},
    "VU": {"SSO": 0.2, "TMF": 0.3, "UGL": 0.15, "BIL": 0.35},
    "CD": {"TMF": 0.2, "UGL": 0.1, "BIL": 0.7},
    "VD": {"TMF": 0.35, "UGL": 0.2, "BIL": 0.45}}, dwell=3),
 "V1h_gld_dwell3": dict(books={
    "CU": {"TQQQ": 0.4, "TMF": 0.2, "GLD": 0.15, "BIL": 0.25},
    "MU": {"QLD": 0.4, "TMF": 0.2, "GLD": 0.15, "BIL": 0.25},
    "VU": {"SSO": 0.25, "TMF": 0.25, "GLD": 0.15, "BIL": 0.35},
    "CD": {"TMF": 0.15, "GLD": 0.15, "BIL": 0.7},
    "VD": {"TMF": 0.3, "GLD": 0.2, "BIL": 0.5}}, dwell=3),
}
for name, kw in variants2.items():
    W = run_state_machine(**kw, tag=name)
    res.append(evaluate(W, opens, name))

# save winner
W = run_state_machine(**variants2["V1g_bondheavy_dwell3"], tag="save")
evaluate(W, opens, "SAVE_volregime_sm", save="volregime_sm")
