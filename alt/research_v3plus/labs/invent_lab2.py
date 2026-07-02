"""Invention loop iteration 2 — breadth+credit gate integration. IS ONLY."""
import sys
sys.path.insert(0, "/home/user/bonds/alt")
import numpy as np
import pandas as pd
from sleeve_engine import backtest_weights, sharpe, cagr, max_dd, ann_vol

ETF = "/home/user/bonds/data/etfs/"
IS_END = pd.Timestamp("2018-12-31")

def load(t):
    df = pd.read_csv(ETF + f"{t}.csv", parse_dates=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"]).set_index("Date")
    return df[["Open", "Close"]].apply(pd.to_numeric, errors="coerce")

def rep(name, r):
    ri = r.loc[:IS_END]
    print(f"{name:36s} IS: SR {sharpe(ri):5.2f}  CAGR {cagr(ri)*100:6.1f}%  "
          f"Vol {ann_vol(ri)*100:5.1f}%  MDD {max_dd(ri)*100:6.1f}%")

# --- the breadth+credit gate (raw, unlagged; callers shift) ---
SECTORS = ["XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLU","XLB"]
def bc_gate_raw() -> pd.Series:
    sc = pd.DataFrame({s: load(s)["Close"] for s in SECTORS}).sort_index()
    breadth = (sc > sc.rolling(200).mean()).mean(axis=1)
    hyg = load("HYG")["Close"].reindex(sc.index).ffill(limit=3)
    ief = load("IEF")["Close"].reindex(sc.index).ffill(limit=3)
    credit_ok = (hyg.pct_change(63) - ief.pct_change(63)) > -0.02
    return ((breadth > 0.5) & credit_ok).astype(float)

BC_RAW = bc_gate_raw()

# ================= ORION with BC gate (monkeypatch) =================
import orion_strategy as ORI

def bc_sig_macro_gate(macro, vix_hi=None, hy_hi=None):
    return BC_RAW.shift(1)   # same lag convention as the original gate

ORI_ORIG_GATE = ORI.sig_macro_gate

def run_orion(gate_fn):
    ORI.sig_macro_gate = gate_fn
    try:
        opens, closes = ORI._load_panels()
        macro = ORI.load_macro()
        W_risk = ORI.build_risk_sleeve(opens, closes, macro).loc[ORI.START_DATE:]
        W_safe = ORI.build_safe_sleeve(opens, closes).loc[ORI.START_DATE:]
        W = ORI.RISK_WEIGHT * W_risk + ORI.SAFE_WEIGHT * W_safe
        opens_bt = opens.loc[W.index.min():W.index.max()]
        r, _ = ORI.backtest(W, opens_bt)
        return r
    finally:
        ORI.sig_macro_gate = ORI_ORIG_GATE

print("== ORION gate variants ==")
ori_base = run_orion(ORI_ORIG_GATE)
rep("ORION base (VIX<30 & HY<7)", ori_base)
ori_bc = run_orion(bc_sig_macro_gate)
rep("ORION breadth+credit gate", ori_bc)

# ================= HELIOS with BC gate (monkeypatch) =================
import helios_strategy as HEL
HEL_ORIG_GATE = HEL.build_macro_gate

def bc_build_macro_gate(idx):
    g = BC_RAW.reindex(idx).ffill().shift(1).fillna(0.0)
    return g, None, None

def run_helios(gate_fn):
    HEL.build_macro_gate = gate_fn
    try:
        close_u, opens = HEL.build_panel()
        start = HEL._start_date(opens)
        W, _ = HEL.build_target_weights(close_u, opens)
        W = W.loc[start:]
        bt = HEL.run_backtest(W, opens.loc[start:])
        return bt["ret"]
    finally:
        HEL.build_macro_gate = HEL_ORIG_GATE

print("\n== HELIOS gate variants ==")
hel_base = run_helios(HEL_ORIG_GATE)
rep("HELIOS base (VIXz & HYchg)", hel_base)
hel_bc = run_helios(bc_build_macro_gate)
rep("HELIOS breadth+credit gate", hel_bc)

# ================= VANGUARD: replace VIX trigger with credit trigger =================
import vanguard_strategy as VAN
VAN_ORIG_TRG = VAN.compute_trigger_count

def bc_compute_trigger_count(fred_df, spy):
    hy = fred_df["HY"]
    hy_slope20 = hy - hy.shift(20)
    hy_slope5 = hy - hy.shift(5)
    t10y2y = fred_df["T10Y2Y"]
    t10y2y_s60 = t10y2y - t10y2y.shift(60)
    idx = spy.index
    hyg = load("HYG")["Close"].reindex(idx).ffill(limit=3)
    ief = load("IEF")["Close"].reindex(idx).ffill(limit=3)
    credit_bad = ((hyg.pct_change(63) - ief.pct_change(63)) < -0.02)
    breadth = BC_RAW.reindex(idx).ffill()
    c_hy = (hy_slope20 > 0.30) | (hy_slope5 > 0.25)
    c_credit = credit_bad
    c_curve = (t10y2y < 0.0) & (t10y2y_s60 < 0.0)
    c_spy = spy < spy.rolling(200).mean()
    trg = (c_hy.astype(float).fillna(0) + c_credit.astype(float).fillna(0)
           + c_curve.astype(float).fillna(0) + c_spy.astype(float).fillna(0))
    return trg.rolling(5).mean()

def run_van(trg_fn):
    VAN.compute_trigger_count = trg_fn
    try:
        bt, w, m, trg = VAN.run(verbose=False)
        return bt["net_ret"]
    finally:
        VAN.compute_trigger_count = VAN_ORIG_TRG

print("\n== VANGUARD trigger variants ==")
van_base = run_van(VAN_ORIG_TRG)
rep("VANGUARD base (4 triggers)", van_base)
van_bc = run_van(bc_compute_trigger_count)
rep("VANGUARD credit-for-VIX trigger", van_bc)

# ================= Blend: WF allocator with upgraded sleeves =================
R = "/home/user/bonds/data/results/"
cry = pd.read_csv(R+"crypto_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
rev = pd.read_csv(R+"reversal_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
tom = pd.read_csv(R+"tom_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
bnd = pd.read_csv(R+"bondtrend_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]

import phoenix_production as prod

def wf_blend(sleeves: dict):
    df = pd.concat(sleeves, axis=1)
    df = df.reindex(cry.index).fillna(0.0)
    Wy = pd.DataFrame(0.0, index=df.index, columns=df.columns)
    years = sorted(set(df.loc["2014-01-02":].index.year))
    for y in years:
        end = pd.Timestamp(f"{y-1}-12-31"); start = end - pd.DateOffset(years=4)
        win = df.loc[start:end]
        active = [c for c in df.columns if win[c].std()*np.sqrt(252) >= 0.05]
        srs = {k: sharpe(win[k]) for k in active}
        corr = win[active].corr()
        rho = (corr.sum(axis=1)-1)/max(len(corr)-1,1)
        bud = pd.Series({k: max(srs[k],0.3)*max(1-rho[k],0.05) for k in active})
        w = pd.Series(1/len(active), index=active)
        cov = win[active].cov()
        for _ in range(500):
            rc = w.values*(cov.values@w.values)
            ratio = (bud.values/bud.sum())/np.maximum(rc/max(rc.sum(),1e-18),1e-12)
            w = pd.Series(np.clip(w.values*ratio**0.3,1e-6,None), index=active); w = w/w.sum()
        w = w.clip(upper=0.35); w = w/w.sum()
        Wy.loc[str(y), active] = w.reindex(active).values
    Wy = Wy.loc["2014-01-02":]
    raw = (df.loc[Wy.index]*Wy).sum(axis=1)
    net, _ = prod.apply_overlay(raw)
    return net

print("\n== WF blends (IS window print only) ==")
base = {"VAN": van_base, "ORI": ori_base, "HEL": hel_base, "CRY": cry,
        "REV": rev, "TOM": tom, "BND": bnd}
rep("blend v3 (all base)", wf_blend(base))
up1 = dict(base); up1["ORI"] = ori_bc
rep("blend +ORION-BC", wf_blend(up1))
up2 = dict(up1); up2["HEL"] = hel_bc
rep("blend +ORION-BC +HELIOS-BC", wf_blend(up2))
up3 = dict(up2); up3["VAN"] = van_bc
rep("blend +all three BC", wf_blend(up3))
print("\ndone (IS only)")
