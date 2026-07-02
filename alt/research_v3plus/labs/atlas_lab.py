"""ATLAS + dynamic sizing lab — IS ONLY."""
import sys
sys.path.insert(0, "/home/user/bonds/alt")
import numpy as np
import pandas as pd
from sleeve_engine import backtest_weights, sharpe, cagr, max_dd, ann_vol
from research_lab import (van, ori, hel, cry, reversal, tom, bondls,
                          DATES, IS_END, load_etf, load_fred, SPY)

IS_END_TS = pd.Timestamp(IS_END)
ERC_START = "2015-01-02"

def rep_is(name, r, extra=""):
    ri = r.loc[:IS_END_TS]
    print(f"{name:30s} IS: SR {sharpe(ri):5.2f}  CAGR {cagr(ri)*100:6.1f}%  "
          f"Vol {ann_vol(ri)*100:5.1f}%  MDD {max_dd(ri)*100:6.1f}%{extra}")

# ---------------------------------------------------------------- ATLAS
# Per-asset long/flat trend: each underlying's own momentum blend decides
# whether its LETF slot is on; slots are inverse-vol weighted and normalized
# to gross 1.0 across ACTIVE slots (cash if none).
PAIRS = {"SPY": "UPRO", "QQQ": "TQQQ", "SMH": "SOXL", "XLE": "ERX",
         "XLF": "FAS", "EEM": "EDC", "FXI": "YINN", "VNQ": "DRN",
         "GLD": "UGL", "USO": "UCO", "TLT": "TMF", "IEF": "TYD"}

def atlas(mom_lbs=(63, 126, 252), vol_lb=60, dow=2, cost=7.0, conviction=True):
    closes = pd.DataFrame({u: load_etf(u)["Close"].reindex(DATES).ffill(limit=3) for u in PAIRS})
    opens = pd.DataFrame({l: load_etf(l)["Open"].reindex(DATES) for l in PAIRS.values()})
    sig = sum((closes.pct_change(lb) > 0).astype(float) for lb in mom_lbs) / len(mom_lbs)
    above = (closes > closes.rolling(200).mean()).astype(float)
    score = (sig * above).shift(1)              # 0..1 conviction, decided close t-1
    if not conviction:
        score = (score >= 0.99).astype(float)   # all horizons agree
    vol = closes.pct_change().rolling(vol_lb).std().shift(1)
    iv = (1 / vol).replace([np.inf, -np.inf], np.nan)
    raw_w = (score * iv)
    tot = raw_w.sum(axis=1)
    W = raw_w.div(tot.where(tot > 0), axis=0).fillna(0.0)
    # weekly freeze (Wednesday)
    reb = (pd.Series(DATES.dayofweek, index=DATES) == dow)
    reb.iloc[0] = True
    mask = pd.DataFrame(np.broadcast_to(reb.values[:, None], W.shape), index=W.index, columns=W.columns)
    W = W.where(mask, np.nan).ffill().fillna(0.0)
    W = W.rename(columns=PAIRS)
    return backtest_weights(W, opens, cost)["ret"]

print("== ATLAS variants ==")
for lbs in [(63, 126, 252), (126, 252), (21, 63, 126, 252)]:
    rep_is(f"atlas {lbs} conv", atlas(lbs))
rep_is("atlas (63,126,252) binary", atlas((63, 126, 252), conviction=False))

# ---------------------------------------------------------------- CRYPTO speed blend
def crypto_speed(lbs=(21, 63, 126), vol_target=0.60):
    from phoenix_v2_crypto import build_chained_panel, build_regime, CASH
    o2o, c2c, closes, proxy_era, _ = build_chained_panel(DATES)
    regime = build_regime(DATES)
    bil = load_etf("BIL")["Open"].reindex(DATES).ffill(limit=5)
    conv = sum((closes.pct_change(lb) > 0).astype(float) for lb in lbs) / len(lbs)
    conv = conv.shift(1)
    vol = (c2c.rolling(30).std() * np.sqrt(252)).shift(1)
    size = (vol_target / vol).clip(upper=1.0)
    raw_w = conv * size
    raw_w = raw_w.where(raw_w.notna(), 0.0)
    raw_w[~regime.astype(bool)] = 0.0
    tot = raw_w.sum(axis=1)
    scale = tot.where(tot <= 1.0, 1.0) / tot.where(tot > 0)
    Wc = raw_w.mul(scale.fillna(0.0), axis=0)
    # weekly Friday freeze
    reb = (pd.Series(DATES.dayofweek, index=DATES) == 4)
    reb.iloc[0] = True
    mask = pd.DataFrame(np.broadcast_to(reb.values[:, None], Wc.shape), index=Wc.index, columns=Wc.columns)
    Wc = Wc.where(mask, np.nan).ffill().fillna(0.0)
    Wcash = (1 - Wc.sum(axis=1)).clip(lower=0)
    # returns
    ret_panel = o2o.copy()
    ret_panel[CASH] = bil.pct_change(fill_method=None)
    W = Wc.copy(); W[CASH] = Wcash
    w_prev = W.shift(1).fillna(0.0)
    gross = (w_prev[ret_panel.columns] * ret_panel).sum(axis=1)
    dw = (W - w_prev).abs()
    bps = pd.DataFrame(10.0, index=DATES, columns=W.columns)
    for name in ["BTC", "ETH"]:
        if name in proxy_era.columns:
            bps.loc[proxy_era[name].astype(bool), name] = 30.0
    bps[CASH] = 2.0
    return gross - (dw * bps / 1e4).sum(axis=1)

print("\n== CRYPTO speed-blend variants ==")
rep_is("crypto orig (63 binary)", cry)
for lbs in [(21, 63, 126), (63, 126)]:
    for vt in [0.4, 0.6, 0.8]:
        rep_is(f"crypto {lbs} vt={vt}", crypto_speed(lbs, vt))

# ---------------------------------------------------------------- dynamic sizing
def dyn_blend(sleeves: dict, lam=0.97, base=None, tail_overlay=True,
              dd_floor=-0.10, gate_pct=0.99, tc_bps=10.0):
    """Dynamic inverse-vol capital allocation at gross 1.0 (no leverage):
    w_i(t) ∝ base_i / EWMAvol_i(t-2), renormalized daily to sum 1."""
    df = pd.concat(sleeves, axis=1).reindex(DATES).fillna(0.0)
    ew = df.pow(2).ewm(alpha=1 - lam).mean().pow(0.5) * np.sqrt(252)
    iv = 1.0 / ew.clip(lower=0.05)
    if base is not None:
        iv = iv.mul(pd.Series(base), axis=1)
    w = iv.div(iv.sum(axis=1), axis=0).shift(2)
    raw = (df * w).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(0)
    raw = raw - turn * (tc_bps / 1e4) * 0.5  # sleeve-level reallocation cost (approx)
    if not tail_overlay:
        return raw
    cum = (1 + raw).cumprod()
    hwm = cum.rolling(252, min_periods=30).max()
    dd = cum / hwm - 1
    dd_mult = (1 + dd / dd_floor).clip(0.0, 1.0)
    sv = raw.rolling(60).std()
    sv_thr = sv.rolling(252, min_periods=60).quantile(gate_pct)
    gate = pd.Series(np.where(sv <= sv_thr, 1.0, 0.5), index=raw.index)
    total = (dd_mult * gate).shift(2).fillna(1.0)
    tc = total.diff().abs().fillna(0) * (tc_bps / 1e4)
    return raw * total - tc

rev = reversal(-1.0, 5)
tomr = tom("TQQQ", trend_filter=False)
bnd = bondls(63, None)
atl = atlas((63, 126, 252))
cry2 = crypto_speed((21, 63, 126), 0.6)

print("\n== dynamic gross-1 risk-parity blends ==")
S = {"VAN": van, "ORI": ori, "HEL": hel, "CRY": cry, "REV": rev, "TOM": tomr, "BND": bnd}
rep_is("dyn 4+REV+TOM+BND", dyn_blend(S))
S2 = {"VAN": van, "ORI": ori, "ATL": atl, "CRY": cry2, "REV": rev, "TOM": tomr, "BND": bnd}
rep_is("dyn VAN,ORI,ATL,CRY2,REV,TOM,BND", dyn_blend(S2))
S3 = {"ATL": atl, "CRY": cry2, "REV": rev, "TOM": tomr, "BND": bnd}
rep_is("dyn ATL,CRY2,REV,TOM,BND", dyn_blend(S3))
S4b = {"VAN": van, "ATL": atl, "CRY": cry2, "REV": rev, "TOM": tomr, "BND": bnd}
rep_is("dyn VAN,ATL,CRY2,REV,TOM,BND", dyn_blend(S4b))
print("\ncorr (2015-2018):")
allsl = pd.concat(S2, axis=1).reindex(DATES).fillna(0.0)
print(allsl.loc[ERC_START:IS_END_TS].corr().round(2).to_string())
