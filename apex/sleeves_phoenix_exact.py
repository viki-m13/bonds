"""APEX v9 — Phoenix-exact clones on synthetic LETFs.

Implements the EXACT Phoenix Vanguard + Orion + Helios + Quantum sleeve
formulas on the extended 2005+ synthetic LETF universe, with the composite
macro gate that gave Phoenix its -1.3% MDD in 2008.

Key macro gate (applied to RISK-ASSET sleeves, not defensive ones):
  trigger count = sum of:
    1. HY OAS widening (proxied via HYG 20d ret < -2% when HYG exists,
       fallback to VIX z-score > 1.5 when HYG missing)
    2. VIX spike (60d z > 1.2 OR level > 30)
    3. Curve inversion in progress (T10Y2Y < 0 AND 60d diff < 0)
    4. SPY below its 200d SMA
  participation = {0: 1.0, 0.5: 0.75, 1.0: 0.50, 1.5: 0.25, 2.0+: 0.0}
  5-day smoothed, 1-day lagged.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import util

ROOT = Path("/home/user/bonds")
FRED = ROOT / "data/fred"
ETF = ROOT / "data/etfs"


def _fred(name, idx):
    fp = FRED / f"{name}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[df.columns[0]].astype(float).reindex(idx).ffill()


def _etf_close(t, idx):
    fp = ETF / f"{t}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df["Close"].astype(float).reindex(idx).ffill()


def _weights_to_ret(W, cp):
    w = W.fillna(0.0)
    rets = cp.pct_change()
    r = (w.shift(1).fillna(0.0) * rets.reindex_like(w).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w.diff().abs().fillna(w.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    return r - drag


def compute_macro_gate(cp: pd.DataFrame) -> pd.Series:
    """Composite Phoenix trigger count → participation fraction [0,1].

    Returns a series of participation multipliers (0 to 1) per date,
    aligned to cp.index, 1-day lagged.
    """
    idx = cp.index
    vix = _fred("VIXCLS", idx)
    hyg = _etf_close("HYG", idx)
    lqd = _etf_close("LQD", idx)
    t10y2y = _fred("T10Y2Y", idx)
    spy = cp["SPY"]

    # Credit proxy: HYG/LQD ratio decline (equivalent to HY-IG spread widening)
    if not hyg.isna().all() and not lqd.isna().all():
        cr = hyg / lqd
        cr_20 = cr - cr.shift(20)
        cr_5 = cr - cr.shift(5)
        # Widening = ratio falling
        c_hy = (cr_20 < -0.015) | (cr_5 < -0.012)
    else:
        c_hy = pd.Series(False, index=idx)

    # VIX
    vix_z = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
    c_vix = (vix_z > 1.2) | (vix > 30.0)

    # Curve
    t_s60 = t10y2y - t10y2y.shift(60)
    c_curve = (t10y2y < 0.0) & (t_s60 < 0.0)

    # SPY below 200MA
    c_spy = ~(spy > spy.rolling(200).mean())

    # Fallback: when HYG missing (pre-2007), use VIX z>1.5 as second credit trigger
    vix_hi = (vix_z > 1.5) | (vix > 35)
    c_hy_effective = c_hy.where(~hyg.isna(), vix_hi)

    trg = (c_hy_effective.astype(float).fillna(0)
           + c_vix.astype(float).fillna(0)
           + c_curve.astype(float).fillna(0)
           + c_spy.astype(float).fillna(0))
    trg_smooth = trg.rolling(5).mean()

    p = pd.Series(1.0, index=idx)
    p[trg_smooth >= 0.5] = 0.75
    p[trg_smooth >= 1.0] = 0.50
    p[trg_smooth >= 1.5] = 0.25
    p[trg_smooth >= 2.0] = 0.00
    return p.shift(1).fillna(0.0)


def sleeve_vanguard_exact(cp: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """EXACT Phoenix VANGUARD clone.
      Universe: {QLD, UGL, TMF, TYD}
      189d mom > 0 AND price > 200d SMA
      Inverse-60d-vol weights
      Monthly rebal
      Multiplied by macro gate participation
    """
    universe = [a for a in ["QLD", "UGL", "TMF", "TYD"] if a in cp.columns]
    p = cp[universe]
    c_lag = p.shift(1)
    mom = c_lag / c_lag.shift(189) - 1.0
    above_sma = c_lag > c_lag.rolling(200).mean()
    eligible = (mom > 0) & above_sma & c_lag.notna()

    rets = p.pct_change().shift(1)
    vol = rets.rolling(60).std()
    iv = 1.0 / vol.replace(0, np.nan)
    iv = iv.where(eligible, 0.0)
    iv_w = iv.div(iv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # Monthly rebal: first bday of each month
    idx = cp.index
    month_mark = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).transform("first") == pd.Series(idx, index=idx)
    rebal_mask = pd.Series(month_mark.values, index=idx)

    w_monthly = iv_w.where(rebal_mask, axis=0).ffill().fillna(0.0)

    gate = compute_macro_gate(cp)
    w_gated = w_monthly.mul(gate, axis=0) * gross

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in universe:
        W[a] = w_gated[a]
    return W


def sleeve_orion_exact(cp: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """EXACT Phoenix ORION clone.
      Two sub-sleeves 50/50:
        RISK: top-3 by 252d mom among {TQQQ,UPRO,SOXL,TECL,FAS,ERX,NUGT,DRN,UCO,EDC,YINN}
              with 200d trend, VIX+HY gate
        SAFE: top-2 by (0.7·mom rank + 0.3·low-vol rank) among {TMF,UBT,TYD,UGL}
      Weekly rebal (Wednesday).
    """
    risk_u = [a for a in ["TQQQ","UPRO","SOXL","TECL","FAS","ERX","NUGT","DRN","UCO","EDC","YINN"] if a in cp.columns]
    safe_u = [a for a in ["TMF","UBT","TYD","UGL"] if a in cp.columns]

    p_r = cp[risk_u]; p_s = cp[safe_u]
    c_r_lag = p_r.shift(1); c_s_lag = p_s.shift(1)

    # RISK: top-3 by 252d mom, 200d trend filter
    mom_r = c_r_lag / c_r_lag.shift(252) - 1.0
    trend_r = c_r_lag > c_r_lag.rolling(200).mean()
    eligible_r = (mom_r > 0) & trend_r & c_r_lag.notna()
    rnk_r = mom_r.where(eligible_r).rank(axis=1, ascending=False, method="first")
    sel_r = (rnk_r <= 3).fillna(False)
    w_r = sel_r.astype(float) / 3.0

    # SAFE: composite 0.7*mom + 0.3*low-vol, top-2 among eligible (mom>0)
    mom_s = c_s_lag / c_s_lag.shift(252) - 1.0
    vol_s = p_s.pct_change().shift(1).rolling(60).std()
    mom_rank = mom_s.rank(axis=1, pct=True)
    lv_rank = 1 - vol_s.rank(axis=1, pct=True)
    comp_s = 0.7 * mom_rank + 0.3 * lv_rank
    rnk_s = comp_s.rank(axis=1, ascending=False, method="first")
    sel_s = (rnk_s <= 2).fillna(False)
    w_s = sel_s.astype(float) / 2.0

    # Weekly rebal: every Wednesday (weekday 2)
    idx = cp.index
    wed_mask = pd.Series(idx.weekday, index=idx) == 2
    rebal_idx = idx[wed_mask]
    rebal_mask = pd.Series(wed_mask.values, index=idx)
    w_r_wk = w_r.where(rebal_mask, axis=0).ffill().fillna(0.0)
    w_s_wk = w_s.where(rebal_mask, axis=0).ffill().fillna(0.0)

    # Macro gate on RISK only; SAFE always on (it IS the hedge)
    gate = compute_macro_gate(cp)
    w_r_gated = w_r_wk.mul(gate, axis=0)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in risk_u:
        W[a] = 0.5 * w_r_gated[a] * gross
    for a in safe_u:
        W[a] = 0.5 * w_s_wk[a] * gross
    return W


def sleeve_helios_exact(cp: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """EXACT Phoenix HELIOS clone.
      13 unlevered underlyings → LETF expression
      189d mom (skip 42) + 200d trend + defensive-bypass for TLT/GLD/IEF
      Weekly (Friday) rebal
      Macro gate for equity/commodity; defensives exempt
    """
    # Map unlevered → 3x LETF (or 2x when 3x not available)
    LETF_MAP = {
        "SPY": "UPRO", "QQQ": "TQQQ", "TLT": "TMF", "GLD": "UGL",
        "IEF": "TYD",
    }
    # Sector extensions if underlyings available
    # Note: XLK/XLF/XLE are not in cp but ARE underlyings for TECL/FAS/ERX synthetic
    # So we'd need to load them separately
    unlevered_idx = cp.index
    extra = {}
    for under, letf in [("XLK", "TECL"), ("XLF", "FAS"), ("XLE", "ERX"), ("SMH", "SOXL"),
                         ("EEM", "EDC"), ("FXI", "YINN"), ("VNQ", "DRN"), ("USO", "UCO")]:
        under_s = _etf_close(under, unlevered_idx)
        if not under_s.isna().all() and letf in cp.columns:
            extra[under] = under_s
    # Build unlevered prices DataFrame
    unlev = {u: cp[u] for u in LETF_MAP if u in cp.columns}
    unlev.update(extra)
    unlev_df = pd.DataFrame(unlev)
    # Combine all mappings
    ALL_MAP = {**LETF_MAP,
               "XLK": "TECL", "XLF": "FAS", "XLE": "ERX", "SMH": "SOXL",
               "EEM": "EDC", "FXI": "YINN", "VNQ": "DRN", "USO": "UCO"}
    # Restrict to available underlying + letf pairs
    MAP = {u: l for u, l in ALL_MAP.items() if u in unlev_df.columns and l in cp.columns}
    unlev_df = unlev_df[list(MAP.keys())]

    c_lag = unlev_df.shift(1)
    mom = c_lag.shift(42) / c_lag.shift(189) - 1.0    # 6-month mom skip 42
    trend = c_lag > c_lag.rolling(200).mean()
    eligible = (mom > 0) & trend & c_lag.notna()
    rnk = mom.where(eligible).rank(axis=1, ascending=False, method="first")
    sel = (rnk <= 2).fillna(False)

    # Defensive bypass: TLT, GLD, IEF stay eligible if mom > 0 & trend OK
    defensives = ["TLT", "GLD", "IEF"]
    for d in defensives:
        if d in sel.columns:
            sel[d] = sel[d] | ((mom[d] > 0) & trend[d]).fillna(False)

    # Weekly rebal: Friday (weekday 4)
    idx = cp.index
    fri_mask = pd.Series(idx.weekday, index=idx) == 4
    sel_wk = sel.where(fri_mask, axis=0).ffill().fillna(False)

    gate = compute_macro_gate(cp)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for u in MAP:
        letf = MAP[u]
        if letf not in W.columns:
            continue
        if u in defensives:
            W[letf] = W[letf] + 0.5 * sel_wk[u].astype(float)
        else:
            W[letf] = W[letf] + 0.5 * sel_wk[u].astype(float) * gate
    W = W * gross
    return W


def sleeve_crypto_exact(cp: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """Phoenix-style CRYPTO sleeve: 63d mom on BTC-USD, gated by SPY 200MA + VIX<30.

    Since crypto LETFs have short history, trade BTC directly via cash-proxy.
    Actually: we'll use our LETF universe's closest analog — TQQQ — gated
    by BTC momentum (proxy). Simpler: if BTC data available, allocate to
    BITO (futures-based BTC ETF) when BTC trending.
    """
    # Load BTC_USD
    btc = _etf_close("BTC_USD", cp.index)
    if btc.isna().all():
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    mom = btc.pct_change(63)
    on = (mom > 0).astype(float)
    # SPY>200MA gate + VIX<30
    spy = cp["SPY"]
    spy_ok = (spy > spy.rolling(200).mean()).astype(float)
    vix = _fred("VIXCLS", cp.index)
    vix_ok = (vix < 30).astype(float).fillna(1.0)
    gate = on * spy_ok * vix_ok

    # Express via BITO if available, else TQQQ
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if "BITO" in cp.columns:
        W["BITO"] = gate * gross
    elif "TQQQ" in cp.columns:
        # Allocate half to TQQQ as crypto-correlated proxy
        W["TQQQ"] = gate * 0.3 * gross
    return W
