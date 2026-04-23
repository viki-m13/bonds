"""APEX v3 sleeves — Phoenix-inspired architecture with regime gates.

Key upgrades vs v2:
  • Each sleeve has a MACRO GATE that shuts it down in unfavorable regimes
    (vol spike, credit blowout, bond-equity correlation flip).
  • Multi-lookback consensus momentum (1m + 3m + 6m averaged) — Keller/Keuning
    PAA style.
  • Sub-sleeve structure for some (RISK/SAFE blend).
  • Weekly rebalance for momentum sleeves (not daily) — matches Phoenix ORION.
  • Crisis correlation filter: when corr(TLT, SPY, 20d) > 0 for 10d → swap
    TMF to cash.
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


def _weights_to_returns(W, cp):
    w = W.fillna(0.0)
    rets = cp.pct_change()
    r = (w.shift(1).fillna(0.0) * rets.reindex_like(w).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w.diff().abs().fillna(w.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    return r - drag


def _scale_down(W, r, target_vol=0.15, win=60):
    rv = r.rolling(win, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return W.mul(m, axis=0)


def _finalize(W, cp, target_vol=0.15):
    r = _weights_to_returns(W, cp)
    return _scale_down(W, r, target_vol=target_vol)


# ---- Macro gates -----------------------------------------------------------

def macro_gates(cp: pd.DataFrame) -> dict:
    """Return dict of boolean pd.Series: vix_ok, corr_ok, spy_ok, cred_ok, overall."""
    idx = cp.index
    spy = cp["SPY"]

    vix = _fred("VIXCLS", idx)
    # VIX < 30 and not in top 1% of 3y rolling
    vix_hi = vix.rolling(756, min_periods=60).quantile(0.99)
    vix_ok = ((vix < 30) & (vix < vix_hi)).astype(float).ffill().fillna(1.0)

    # SPY > 200MA
    spy_ok = (spy > spy.rolling(200).mean()).astype(float).fillna(0.0)

    # Credit proxy: HYG/LQD ratio should not be below its 6m min-3σ
    hyg = _etf_close("HYG", idx)
    lqd = _etf_close("LQD", idx)
    cr = (hyg / lqd)
    cr_ma = cr.rolling(126).mean()
    cr_sd = cr.rolling(126).std()
    cred_ok = (cr > (cr_ma - 2 * cr_sd)).astype(float).fillna(1.0)

    # Bond-equity correlation filter (2022 protection)
    spy_r = spy.pct_change()
    tlt_r = cp["TLT"].pct_change() if "TLT" in cp.columns else spy_r * 0
    corr_20d = spy_r.rolling(20).corr(tlt_r)
    # Cumulative: 10-day count of positive corr days
    pos_streak = (corr_20d > 0).rolling(10).sum()
    corr_ok = (pos_streak < 8).astype(float).fillna(1.0)  # if 8/10 last days had positive corr → flag

    return {
        "vix_ok": vix_ok,
        "spy_ok": spy_ok,
        "cred_ok": cred_ok,
        "corr_ok": corr_ok,
    }


# ---- S1: Vanguard-like — momentum on 4 select LETFs with vol gate ----------

def sleeve_vanguard(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Monthly: rank {QLD, UGL, TMF, TYD} by 189d momentum, top-2 above 200MA,
    inv-60d-vol weight. Gated by VIX < 30."""
    universe = ["QLD", "UGL", "TMF", "TYD"]
    universe = [a for a in universe if a in cp.columns]
    p = cp[universe]
    mom189 = p.pct_change(189)
    ma200 = p.rolling(200).mean()
    # Skip last 42d for reversal
    mom_eligible = (p > ma200) & (mom189 > 0)
    rnk = mom189.where(mom_eligible).rank(axis=1, ascending=False, method="first")
    sel = (rnk <= 2).fillna(False)
    vol60 = p.pct_change().rolling(60).std()
    iv = 1.0 / vol60.replace(0, np.nan)
    raw = sel.astype(float) * iv
    rowsum = raw.sum(axis=1).replace(0, np.nan)
    w = raw.div(rowsum, axis=0).fillna(0.0)

    # Gate
    g = macro_gates(cp)["vix_ok"]

    # Monthly rebal: every 21 days
    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_rebal = mask % 21 == 0
    w_monthly = w.where(is_rebal).ffill().fillna(0.0)
    w_monthly = w_monthly.mul(g, axis=0)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in universe:
        W[a] = w_monthly[a]
    return _finalize(W, cp, target_vol=target_vol)


# ---- S2: Orion-like — RISK + SAFE sub-sleeves, weekly, macro gate ---------

def sleeve_orion(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """50/50 blend of RISK (top-3 252d mom on 7 equity LETFs, gated) +
    SAFE (top-2 on 4 bond/gold LETFs, always on). Weekly rebal."""
    risk_u = [a for a in ["UPRO","TQQQ","TECL","SOXL","FAS","EDC","YINN"] if a in cp.columns]
    safe_u = [a for a in ["TMF","UBT","TYD","UGL"] if a in cp.columns]
    p_r = cp[risk_u]; p_s = cp[safe_u]
    mom_r = p_r.pct_change(252).shift(21)
    mom_s = p_s.pct_change(252).shift(21)
    vol_r = p_r.pct_change().rolling(60).std()
    vol_s = p_s.pct_change().rolling(60).std()

    # RISK: top-3 by mom, require mom > 0
    rnk_r = mom_r.rank(axis=1, ascending=False, method="first")
    sel_r = (rnk_r <= 3) & (mom_r > 0)
    w_r = sel_r.astype(float) / 3.0

    # SAFE: top-2 by composite 0.7*mom_rank + 0.3*low_vol_rank
    mom_rank = mom_s.rank(axis=1, pct=True)
    lv_rank = (1 - vol_s.rank(axis=1, pct=True))
    comp = 0.7 * mom_rank + 0.3 * lv_rank
    rnk_s = comp.rank(axis=1, ascending=False, method="first")
    sel_s = (rnk_s <= 2)
    w_s = sel_s.astype(float) / 2.0

    # Weekly rebal (every 5 days)
    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_rebal = mask % 5 == 0
    w_r_w = w_r.where(is_rebal).ffill().fillna(0.0)
    w_s_w = w_s.where(is_rebal).ffill().fillna(0.0)

    # Macro gate on RISK only
    g = macro_gates(cp)
    risk_gate = (g["vix_ok"] * g["cred_ok"])
    w_r_w = w_r_w.mul(risk_gate, axis=0)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in risk_u:
        W[a] = 0.5 * w_r_w[a]
    for a in safe_u:
        W[a] = 0.5 * w_s_w[a]
    return _finalize(W, cp, target_vol=target_vol)


# ---- S3: Helios-like — 6m momentum on UNLEVERED underlying assets ----------

def sleeve_helios(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Top-2 by 189d-42d momentum on unlevered ETFs. Express via LETF."""
    # Map unlevered → LETF expression
    LETF_MAP = {
        "SPY": ("UPRO", 3), "QQQ": ("TQQQ", 3), "TLT": ("TMF", 3),
        "GLD": ("UGL", 2),  "IEF": ("TYD", 3),
    }
    # Optional adds if available
    if "EEM" in cp.columns:
        LETF_MAP["EEM"] = ("EDC", 3)
    universe = [u for u in LETF_MAP if u in cp.columns]
    p = cp[universe]
    mom = p.shift(42).pct_change(189 - 42)
    ma200 = p.rolling(200).mean()
    eligible = (p > ma200) & (mom > 0)
    rnk = mom.where(eligible).rank(axis=1, ascending=False, method="first")
    sel = (rnk <= 2).fillna(False)

    # Weekly rebal
    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_rebal = mask % 5 == 0
    sel_w = sel.where(is_rebal).ffill().fillna(False)

    # VIX gate
    g = macro_gates(cp)["vix_ok"]

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for u in universe:
        letf, _ = LETF_MAP[u]
        if letf in cp.columns:
            W[letf] = 0.5 * sel_w[u].astype(float) * g
    return _finalize(W, cp, target_vol=target_vol)


# ---- S4: PAA (Keller) — accelerating momentum -----------------------------

def sleeve_paa(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Score = avg of 21d, 63d, 126d returns. Top-2 equal weight.
    Apply protection: for each of the top-2, if its score <= 0, replace with cash.
    """
    risk_u = [a for a in ["UPRO","TQQQ","TECL","SOXL","FAS","EDC","YINN",
                           "TMF","UBT","UGL","UCO","DRN"] if a in cp.columns]
    p = cp[risk_u]
    score = (p.pct_change(21) + p.pct_change(63) + p.pct_change(126)) / 3.0
    rnk = score.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= 2) & (score > 0)

    # Monthly rebal
    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_rebal = mask % 21 == 0
    sel_m = sel.where(is_rebal).ffill().fillna(False)
    n_sel = sel_m.sum(axis=1)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in risk_u:
        W[a] = (sel_m[a].astype(float) / n_sel.replace(0, np.nan)).fillna(0.0) * 0.5
    return _finalize(W, cp, target_vol=target_vol)


# ---- S5: Trend-vol (per-asset vol-targeted trend) --------------------------

def sleeve_trend_vol(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Per LETF: long if close > 200d MA AND 20d return > 0; size by inv-vol,
    cap 0.33, sum ≤ 1."""
    universe = [a for a in ["UPRO","TQQQ","TECL","SOXL","FAS","EDC",
                             "TMF","UBT","UGL","UCO","DRN"] if a in cp.columns]
    p = cp[universe]
    ma200 = p.rolling(200).mean()
    r20 = p.pct_change(20)
    on = ((p > ma200) & (r20 > 0)).astype(float)
    vol = p.pct_change().rolling(60).std() * np.sqrt(util.DPY)
    raw = on * (0.20 / vol.replace(0, np.nan))
    raw = raw.fillna(0.0).clip(upper=0.33)
    s = raw.sum(axis=1)
    scale = np.minimum(1.0, 1.0 / s.replace(0, np.nan)).fillna(0.0)
    w = raw.mul(scale, axis=0)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in universe:
        W[a] = w[a]
    return _finalize(W, cp, target_vol=target_vol)


# ---- S6: Risk Parity with correlation filter ------------------------------

def sleeve_rpar_cf(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Inverse-vol UPRO/TMF/UGL BUT replace TMF with cash when TLT-SPY 20d
    correlation goes persistently positive (2022 protection)."""
    assets = [a for a in ["UPRO","TMF","UGL"] if a in cp.columns]
    rv = cp[assets].pct_change().rolling(60).std()
    iv = 1.0 / rv.replace(0, np.nan)
    iv = iv.div(iv.sum(axis=1), axis=0).fillna(0.0)

    # Correlation filter
    if "TLT" in cp.columns:
        sr = cp["SPY"].pct_change()
        tr = cp["TLT"].pct_change()
        corr = sr.rolling(20).corr(tr)
        pos_streak = (corr > 0).rolling(10).sum()
        tmf_ok = (pos_streak < 8).astype(float).fillna(1.0)
    else:
        tmf_ok = pd.Series(1.0, index=cp.index)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in assets:
        if a == "TMF":
            W[a] = iv[a] * tmf_ok
        else:
            W[a] = iv[a]
    # re-normalize after gating TMF
    s = W[assets].sum(axis=1).replace(0, np.nan)
    W[assets] = W[assets].div(s, axis=0).fillna(0.0)
    return _finalize(W, cp, target_vol=target_vol)


# ---- S7: Short-term MR on winners only ------------------------------------

def sleeve_mrev_winners(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Among LETFs above 200MA: buy the 2 worst 5-day performers weekly, hold 5d."""
    universe = [a for a in ["UPRO","TQQQ","TECL","SOXL","FAS","EDC",
                             "UGL","DRN"] if a in cp.columns]
    p = cp[universe]
    ma200 = p.rolling(200).mean()
    r5 = p.pct_change(5)
    # Only eligible if above 200MA
    eligible = (p > ma200) & (r5 < 0)
    rnk = r5.where(eligible).rank(axis=1, ascending=True, method="first")   # worst = rank 1
    sel = (rnk <= 2).fillna(False)

    # Weekly entry, 5-day hold
    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_rebal = mask % 5 == 0
    sel_w = sel.where(is_rebal).ffill(limit=5).fillna(False)

    # Macro gate: SPY > 200MA
    g = macro_gates(cp)["spy_ok"]

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in universe:
        W[a] = 0.5 * sel_w[a].astype(float) * g
    return _finalize(W, cp, target_vol=target_vol)


# ---- S8: Calendar / turn-of-month -----------------------------------------

def sleeve_calendar(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Long top-2 momentum equity LETFs on (last 2 + first 3 days of month)."""
    idx = cp.index
    day = pd.Series(idx.day, index=idx)
    month = pd.Series(idx.month, index=idx)
    year = pd.Series(idx.year, index=idx)
    group = year.astype(str) + "-" + month.astype(str).str.zfill(2)
    g = pd.DataFrame({"g": group.values}, index=idx)
    fwd_rank = g.groupby("g").cumcount()
    back_rank = g.groupby("g").cumcount(ascending=False)
    tom = ((fwd_rank < 3) | (back_rank < 2)).astype(bool)

    equity_u = [a for a in ["UPRO","TQQQ","TECL","SOXL","FAS"] if a in cp.columns]
    p = cp[equity_u]
    mom = p.pct_change(63)
    rnk = mom.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= 2) & (mom > 0)

    # Only active on TOM days
    sel_tom = sel.mul(tom, axis=0).astype(bool)

    # Macro gate
    gates = macro_gates(cp)
    gate = gates["vix_ok"] * gates["spy_ok"]

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in equity_u:
        W[a] = 0.5 * sel_tom[a].astype(float) * gate
    return _finalize(W, cp, target_vol=target_vol)


# ---- S9: ML (Phoenix Quantum style) — placeholder, loaded externally -----

def sleeve_ml_load(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Load the pre-computed ML weights from disk."""
    fp = Path("/home/user/bonds/data/apex/ml_v2_weights.csv")
    if not fp.exists():
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    W = pd.read_csv(fp, parse_dates=["Date"], index_col="Date")
    W = W.reindex(cp.index).fillna(0.0)
    # Align columns
    full = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for c in W.columns:
        if c in full.columns:
            full[c] = W[c]
    return _finalize(full, cp, target_vol=target_vol)
