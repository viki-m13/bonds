"""Reconstruct a long-Treasury TOTAL-RETURN series from constant-maturity yields
(FRED DGS20, back to 2000; DGS10 back to 1962), so we can build TMF (3x long
Treasury) through the dot-com crash for the leveraged risk-parity test.

Daily bond TR (constant-maturity approx):
    r_t = y_{t-1}/252              # carry
        - D * dy_t                 # duration price move
        + 0.5 * C * dy_t^2         # convexity
D (modified duration) and C are CALIBRATED to best-match REAL TLT daily returns
over 2005-2018, then locked and extended backward (same discipline as the TQQQ
reconstruction). Output merged into the panel as TLT_R (recon) / TMF_R (3x).
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

def yseries(name):
    f = f"{REPO}/data/fred/{name}.csv"
    s = pd.read_csv(f); s["Date"] = pd.to_datetime(s["Date"])
    return s.set_index("Date")[name].replace(".", np.nan).astype(float).dropna()/100.0

dgs20 = yseries("DGS20"); dgs10 = yseries("DGS10")
# splice: DGS20 where available, else DGS10 shifted by its avg spread to DGS20
ov = pd.concat([dgs20, dgs10], axis=1).dropna()
spread = (ov["DGS20"] - ov["DGS10"]).mean()
y = dgs20.reindex(dgs20.index.union(dgs10.index)).copy()
y = y.fillna(dgs10 + spread).dropna().sort_index()

def recon_tr(y, D, C, fee=0.0015):
    dy = y.diff()
    r = y.shift(1)/252.0 - D*dy + 0.5*C*(dy**2) - fee/252.0
    px = (1+r.fillna(0)).cumprod()
    return px/px.dropna().iloc[0]*100.0

def load_etf(tk):
    for d in (f"{REPO}/data/etfs", f"{REPO}/data/etfs_extended"):
        f = f"{d}/{tk}.csv"
        if os.path.exists(f):
            df = pd.read_csv(f); df["Date"] = pd.to_datetime(df["Date"])
            return df.set_index("Date")["Close"].sort_index()
    return None

if __name__ == "__main__":
    real = load_etf("TLT")
    # calibrate D (and light C) to match real TLT daily returns 2005-2018
    realr = real.pct_change()
    best = None
    for D in np.arange(10.0, 19.1, 0.5):
        for C in (0.0, 100.0, 200.0):
            tr = recon_tr(y, D, C)
            j = pd.concat([realr.rename("r"), tr.pct_change().rename("x")], axis=1).dropna()
            j = j[(j.index >= "2005-01-01") & (j.index <= "2018-12-31")]
            corr = j["r"].corr(j["x"]); te = (j["r"]-j["x"]).std()*np.sqrt(252)
            score = corr - te
            if best is None or score > best[0]:
                best = (score, D, C, corr, te)
    _, D, C, corr, te = best
    print(f"calibrated  D={D}  C={C}  corr={corr:.3f}  trackErr={te:.1%}")
    tlt_r = recon_tr(y, D, C)
    # validate cumulative 2005-2020
    j = pd.concat([real.rename("real"), tlt_r.rename("recon")], axis=1).dropna()
    j = j[j.index >= "2005-01-01"]; j = j/j.iloc[0]
    print(f"cum 2005->end : real {j['real'].iloc[-1]:.2f}x  recon {j['recon'].iloc[-1]:.2f}x")
    print(f"recon TLT range {tlt_r.index.min().date()} .. {tlt_r.index.max().date()}")
    # dot-com behavior: 20y yield fell 2000->2003; recon TLT should have risen strongly
    d0, d1 = pd.Timestamp("2000-03-24"), pd.Timestamp("2002-10-09")  # nasdaq peak->trough
    seg = tlt_r[(tlt_r.index>=d0)&(tlt_r.index<=d1)]
    print(f"recon long-Tsy over dot-com bust ({d0.date()}->{d1.date()}): {seg.iloc[-1]/seg.iloc[0]-1:+.0%}")
    # save recon TLT + TMF(3x) to a side pickle to merge into the panel
    tmf_r = recon_tr(y, D*3, C*9, fee=0.0106)  # 3x duration & convexity, TMF fee
    # rebuild as 3x daily of the 1x TR (more correct than 3x duration):
    r1 = tlt_r.pct_change()
    r3 = 3*r1 - (0.0106 + 2*0.03)/252.0   # 3x daily - (expense + 2x borrow)
    tmf_r = (1+r3.fillna(0)).cumprod(); tmf_r = tmf_r/tmf_r.dropna().iloc[0]*100.0
    pd.to_pickle({"TLT_R": tlt_r, "TMF_R": tmf_r, "D": D, "C": C},
                 f"{HERE}/_bond_recon.pkl")
    print("saved _bond_recon.pkl  (TLT_R, TMF_R)")
    seg3 = tmf_r[(tmf_r.index>=d0)&(tmf_r.index<=d1)]
    print(f"recon TMF (3x) over dot-com bust: {seg3.iloc[-1]/seg3.iloc[0]-1:+.0%}")
