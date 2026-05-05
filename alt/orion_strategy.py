"""
ORION — Orthogonal Signal Ensemble
==================================

A two-sleeve / four-signal portfolio over a broad leveraged-ETF universe.

Sleeves (construction orthogonality):
    RISK sleeve  — leveraged long-equity / sector / commodity LETFs
    SAFE sleeve  — leveraged long-bond LETFs and leveraged gold
These two baskets have close-to-zero return correlation in the realised
period and are the primary source of the ensemble's diversification.

Signals (information orthogonality — computed, evaluated, and correlated
separately):
    S1. Cross-sectional 12-month momentum (252d log-return, no skip)
        -> rank within the RISK sleeve.
    S2. Time-series trend filter (price > 200-day MA, lagged 1 day)
        -> acts as an "eligibility" mask before ranking.
    S3. Low-vol tilt on the SAFE sleeve (rank by -rolling 60d vol)
        -> orthogonal selector within the safe basket.
    S4. Macro-regime gate (VIX < 30 AND HY OAS < 7.0)
        -> binary risk-on flag. When OFF, the RISK sleeve goes to CASH;
        the SAFE sleeve is ALWAYS ON.

Portfolio construction (weekly, executed next-day OPEN):
    w_risk = top-K by S1 among RISK names passing S2, normalised to 1;
             multiplied by S4 gate.
    w_safe = top-K by S1 among SAFE names passing S2 (always on).
    W      = 0.5 * w_risk + 0.5 * w_safe.
    Rebalance:  every Wednesday.  Freeze in between.
    Fills:      at the next day's OPEN price.  Signal uses close[t-1] only.
    TC:         5 bps one-way on |dw| per symbol per rebalance.

No daily vol scaling.  No sigma-targeting.  Signals strictly lagged >= 1 day.

Deliverables produced by this file:
    data/results/orion_metrics.json
    data/results/orion_returns.csv
    alt/ORION_DESIGN.md  (human summary; written separately)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ETF_DIR = ROOT / "data" / "etfs"
FRED_DIR = ROOT / "data" / "fred"
OUT_DIR = ROOT / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Universe (>= 10 leveraged names, equity / sector / international / REIT /
# commodity / bond).
# ---------------------------------------------------------------------------
RISK_UNIVERSE = [
    # Broad equity 3x
    "TQQQ", "UPRO",
    # Broad equity 2x
    "QLD", "SSO",
    # Sector 3x
    "SOXL", "TECL", "FAS", "ERX",
    # International 3x
    "EDC", "YINN",
    # Real estate 3x
    "DRN",
    # Commodities 2x
    "UCO",
]
SAFE_UNIVERSE = [
    "TMF",  # 3x long Treasury 20+
    "UBT",  # 2x long Treasury 20+
    "TYD",  # 3x long Treasury 7-10
    "UGL",  # 2x long gold
]
UNIVERSE = RISK_UNIVERSE + SAFE_UNIVERSE

# Parameters (set by inspection of IS only)
MOM_LOOKBACK    = 252   # 12-month log-return
MOM_SKIP        = 0     # no skip - the 200d MA already removes some noise
TREND_MA        = 200   # trend eligibility
VIX_HI          = 30.0  # VIX must be below this in risk-on
HY_HI           = 7.0   # HY OAS must be below this in risk-on
K_RISK          = 4     # top-K names in risk sleeve
K_SAFE          = 2     # top-K names in safe sleeve
RISK_WEIGHT     = 0.50  # fraction of the book to RISK sleeve
SAFE_WEIGHT     = 0.50  # fraction to SAFE sleeve
REBAL_DOW       = 2     # Wednesday
COST_BPS        = 5.0

START_DATE = "2010-03-11"   # all RISK LETFs available by this date
END_DATE   = None  # extend to latest available data
IS_END     = "2018-12-31"
OOS_START  = "2019-01-01"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _read_etf(sym):
    df = pd.read_csv(ETF_DIR / f"{sym}.csv", parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"])
    return df.set_index("Date")[["Open", "Close"]].rename(
        columns={"Open": f"{sym}_Open", "Close": f"{sym}_Close"}
    )


def load_prices(symbols):
    frames = [_read_etf(s) for s in symbols]
    px = pd.concat(frames, axis=1, sort=True).sort_index()
    opens = px[[f"{s}_Open" for s in symbols]]
    closes = px[[f"{s}_Close" for s in symbols]]
    opens.columns = symbols
    closes.columns = symbols
    return opens, closes


def load_macro():
    def rd(name):
        d = pd.read_csv(FRED_DIR / f"{name}.csv", parse_dates=["Date"])
        d = d.sort_values("Date").drop_duplicates(subset=["Date"]).set_index("Date")
        return d[name]
    return pd.DataFrame({
        "VIX":    rd("VIXCLS"),
        "HY":     rd("BAMLH0A0HYM2"),
        "T10Y3M": rd("T10Y3M"),
    })


# ---------------------------------------------------------------------------
# Weekly rebalance helper - freeze weights between rebal days
# ---------------------------------------------------------------------------
def _weekly_freeze(weights, dayofweek=REBAL_DOW):
    idx = weights.index
    dow = pd.Series(idx.dayofweek, index=idx)
    rebal = (dow == dayofweek).copy()
    rebal.iloc[0] = True
    out = weights.copy()
    last = None
    for i in range(len(out)):
        if rebal.iloc[i]:
            last = out.iloc[i].values.copy()
        else:
            if last is None:
                last = out.iloc[i].values.copy()
            out.iloc[i] = last
    return out


# ---------------------------------------------------------------------------
# Signal definitions (all use close[t-1]; outputs are shifted/lagged)
# ---------------------------------------------------------------------------
def sig_momentum(close_ret, lookback=MOM_LOOKBACK, skip=MOM_SKIP):
    """S1: 12-month log-return, lagged >= 1 day (shift skip+1)."""
    lr = np.log1p(close_ret)
    raw = lr.rolling(lookback - skip).sum()
    return raw.shift(skip + 1)


def sig_trend_filter(close, ma=TREND_MA):
    """S2: +1 if close > 200d MA else 0 (lagged 1 day)."""
    return (close > close.rolling(ma).mean()).astype(float).shift(1)


def sig_low_vol(close_ret, window=60):
    """S3 score: higher is better (i.e. negative realised vol), lagged 1 day."""
    vol = close_ret.rolling(window).std().shift(1)
    return -vol


def sig_macro_gate(macro, vix_hi=VIX_HI, hy_hi=HY_HI):
    """S4: 1 when risk-on (VIX low & HY spread low), 0 otherwise.
    Inputs are lagged via shift(1)."""
    m = macro.ffill()
    risk_off = (m["VIX"] > vix_hi) | (m["HY"] > hy_hi)
    return (~risk_off).astype(float).shift(1)


# ---------------------------------------------------------------------------
# Sleeve portfolio construction
# ---------------------------------------------------------------------------
def _topk_equal(score, k):
    """Top-K equal weight, renormalised to 1. If no names, all zeros."""
    ranks = score.rank(axis=1, ascending=False, method="first")
    m = (ranks <= k).astype(float)
    s = m.sum(axis=1).replace(0, np.nan)
    return m.div(s, axis=0).fillna(0.0)


def build_risk_sleeve(opens, closes, macro):
    """w_risk = top-K12-mo-momentum among RISK-universe names above 200d MA,
    zeroed when macro gate = 0."""
    cr = closes.pct_change()
    mom = sig_momentum(cr)
    trend = sig_trend_filter(closes)
    # restrict to RISK columns
    mom_r = mom[RISK_UNIVERSE]
    trend_r = trend[RISK_UNIVERSE]
    score = mom_r.where(trend_r > 0.5)
    w = _topk_equal(score, K_RISK)
    W = pd.DataFrame(0.0, index=opens.index, columns=opens.columns)
    W[RISK_UNIVERSE] = w.reindex(opens.index).fillna(0.0).values
    W = _weekly_freeze(W)
    gate = sig_macro_gate(macro).reindex(opens.index).ffill().fillna(0.0)
    return W.mul(gate, axis=0)


def build_safe_sleeve(opens, closes):
    """w_safe = top-K momentum among SAFE-universe names above 200d MA.
    Low-vol tilt is used as a 10% secondary score to break ties among bond LETFs."""
    cr = closes.pct_change()
    mom = sig_momentum(cr)
    trend = sig_trend_filter(closes)
    lv = sig_low_vol(cr)
    # Standardise each across SAFE universe per day
    mom_s = mom[SAFE_UNIVERSE]
    lv_s = lv[SAFE_UNIVERSE]
    trend_s = trend[SAFE_UNIVERSE]

    def xs_z(df):
        mu = df.mean(axis=1)
        sd = df.std(axis=1).replace(0, np.nan)
        return df.sub(mu, axis=0).div(sd, axis=0)

    composite = 0.7 * xs_z(mom_s) + 0.3 * xs_z(lv_s)
    composite = composite.where(trend_s > 0.5)
    w = _topk_equal(composite, K_SAFE)
    W = pd.DataFrame(0.0, index=opens.index, columns=opens.columns)
    W[SAFE_UNIVERSE] = w.reindex(opens.index).fillna(0.0).values
    return _weekly_freeze(W)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def backtest(weights, opens, cost_bps=COST_BPS):
    """w[t] set at close[t-1], held open[t] -> open[t+1]."""
    w = weights.reindex(opens.index).fillna(0.0)
    o2o = opens.pct_change().shift(-1)         # return from open[t] to open[t+1]
    gross = (w * o2o).sum(axis=1)
    dw = w.diff().abs().sum(axis=1).fillna(0.0)
    tc = dw * (cost_bps / 1e4)
    net = (gross - tc).fillna(0.0)
    return net, dw


def sharpe(r):
    r = r.dropna()
    if len(r) == 0 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def cagr(r):
    r = r.dropna()
    if len(r) == 0:
        return 0.0
    return float((1 + r).prod() ** (252 / len(r)) - 1)


def max_drawdown(r):
    r = r.dropna()
    if len(r) == 0:
        return 0.0
    c = (1 + r).cumprod()
    return float((c / c.cummax() - 1).min())


def annual_vol(r):
    return float(r.std() * np.sqrt(252))


# ---------------------------------------------------------------------------
# Orthogonality diagnostics: standalone signal returns
# ---------------------------------------------------------------------------
def standalone_signal_portfolios(opens, closes, macro):
    """For each of S1..S4, produce a standalone top-K portfolio for
    correlation/orthogonality analysis. Weekly rebalance, same TC.
    Universe for these diagnostics = RISK (S1..S3) and a regime portfolio for S4."""
    cr = closes.pct_change()
    lr = np.log1p(cr)

    idx = opens.index
    out = {}

    # S1: 12-m momentum on RISK, top-K=4, no trend, no gate
    mom = lr.rolling(MOM_LOOKBACK).sum().shift(1)[RISK_UNIVERSE]
    w = _topk_equal(mom, 4)
    W = pd.DataFrame(0.0, index=idx, columns=opens.columns)
    W[RISK_UNIVERSE] = w.reindex(idx).fillna(0.0).values
    W = _weekly_freeze(W)
    out["S1_momentum"], _ = backtest(W, opens)

    # S2: trend-filter only — equal weight among names above 200d MA (from WHOLE universe)
    trend = (closes > closes.rolling(TREND_MA).mean()).astype(float).shift(1)
    cnt = trend.sum(axis=1).replace(0, np.nan)
    W = trend.div(cnt, axis=0).fillna(0.0)
    W = _weekly_freeze(W)
    out["S2_trend"], _ = backtest(W, opens)

    # S3: low-vol tilt on SAFE only - top-2
    lv = (-cr.rolling(60).std().shift(1))[SAFE_UNIVERSE]
    w = _topk_equal(lv, 2)
    W = pd.DataFrame(0.0, index=idx, columns=opens.columns)
    W[SAFE_UNIVERSE] = w.reindex(idx).fillna(0.0).values
    W = _weekly_freeze(W)
    out["S3_lowvol_safe"], _ = backtest(W, opens)

    # S4: macro-regime pair switch - UPRO when risk-on, TMF when risk-off
    gate = sig_macro_gate(macro).reindex(idx).ffill().fillna(0.0)
    W = pd.DataFrame(0.0, index=idx, columns=opens.columns)
    W["UPRO"] = gate
    W["TMF"] = 1 - gate
    # regime switches are not weekly-locked
    regime_change = gate.diff().abs() > 0
    rebal_dow = pd.Series(idx.dayofweek, index=idx) == REBAL_DOW
    rebal = rebal_dow | regime_change.fillna(False)
    rebal.iloc[0] = True
    # custom freeze using this rebal mask
    W2 = W.copy()
    last = None
    for i in range(len(W2)):
        if rebal.iloc[i]:
            last = W2.iloc[i].values.copy()
        else:
            if last is None:
                last = W2.iloc[i].values.copy()
            W2.iloc[i] = last
    out["S4_regime"], _ = backtest(W2, opens)

    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Live-signal-friendly weight builder (single source of truth)
# ---------------------------------------------------------------------------
def build_weights(live_extend: bool = False) -> pd.DataFrame:
    """Compute the canonical ORION daily target-weight DataFrame.

    Index: trading dates from START_DATE.
    Columns: UNIVERSE (RISK + SAFE LETFs). Weights sum to <= 1.0 per day.

    live_extend: If True, extend the date index by one BDay forward
        (ffilling closes & opens) so the LAST row is W[t+1] computed from
        close[t] info — i.e., the weight to hold at next-day open. The
        signal layer's shift(1) automatically advances the lookback by one
        day. Used by alt/live_signal.py only.
    """
    opens, closes = load_prices(UNIVERSE)
    opens = opens.dropna(how="any")
    closes = closes.loc[opens.index]
    opens = opens.loc[START_DATE:END_DATE]
    closes = closes.loc[START_DATE:END_DATE]
    if live_extend and len(opens) > 0:
        next_day = opens.index[-1] + pd.tseries.offsets.BDay()
        opens.loc[next_day] = opens.iloc[-1]
        closes.loc[next_day] = closes.iloc[-1]
        opens = opens.sort_index()
        closes = closes.sort_index()
    macro = load_macro()
    W_risk = build_risk_sleeve(opens, closes, macro)
    W_safe = build_safe_sleeve(opens, closes)
    return RISK_WEIGHT * W_risk + SAFE_WEIGHT * W_safe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    opens, closes = load_prices(UNIVERSE)
    opens = opens.dropna(how="any")
    closes = closes.loc[opens.index]
    opens = opens.loc[START_DATE:END_DATE]
    closes = closes.loc[START_DATE:END_DATE]

    macro = load_macro()

    W_risk = build_risk_sleeve(opens, closes, macro)
    W_safe = build_safe_sleeve(opens, closes)
    W = RISK_WEIGHT * W_risk + SAFE_WEIGHT * W_safe

    returns, dw = backtest(W, opens)
    returns.name = "orion"

    # Sub-sleeve returns for diagnostics
    r_risk, _ = backtest(W_risk, opens)
    r_safe, _ = backtest(W_safe, opens)

    # Orthogonality diagnostics
    standalone = standalone_signal_portfolios(opens, closes, macro).dropna(how="all")
    corr = standalone.corr().round(3)
    per_signal_sharpe = {k: sharpe(standalone[k]) for k in standalone.columns}

    # Sleeves
    sleeve_sharpe = {"RISK_sleeve": sharpe(r_risk),
                     "SAFE_sleeve": sharpe(r_safe)}
    sleeve_corr = pd.DataFrame({"RISK": r_risk, "SAFE": r_safe}).corr().round(3)

    # Metrics
    def block(r, label):
        return {
            f"{label}_sharpe": sharpe(r),
            f"{label}_cagr":   cagr(r),
            f"{label}_vol":    annual_vol(r),
            f"{label}_mdd":    max_drawdown(r),
        }
    r_is  = returns.loc[:IS_END]
    r_oos = returns.loc[OOS_START:]

    metrics = {}
    metrics.update(block(r_is,  "IS"))
    metrics.update(block(r_oos, "OOS"))
    metrics.update(block(returns, "Full"))
    metrics["avg_turnover_annualised"] = float(dw.mean() * 252)
    metrics["sharpe_gap_IS_OOS"] = float(abs(metrics["IS_sharpe"] - metrics["OOS_sharpe"]))
    metrics["per_signal_sharpe"] = per_signal_sharpe
    metrics["signal_return_corr"] = corr.to_dict()
    metrics["sleeve_sharpe"] = sleeve_sharpe
    metrics["sleeve_corr"] = sleeve_corr.to_dict()
    metrics["params"] = dict(
        MOM_LOOKBACK=MOM_LOOKBACK, MOM_SKIP=MOM_SKIP, TREND_MA=TREND_MA,
        VIX_HI=VIX_HI, HY_HI=HY_HI, K_RISK=K_RISK, K_SAFE=K_SAFE,
        RISK_WEIGHT=RISK_WEIGHT, SAFE_WEIGHT=SAFE_WEIGHT,
        REBAL_DOW=REBAL_DOW, COST_BPS=COST_BPS,
        universe_size=len(UNIVERSE),
        start=START_DATE, end=END_DATE, is_end=IS_END, oos_start=OOS_START,
    )

    # Requirements
    req = {
        "sharpe_full_ge_2.0": metrics["Full_sharpe"] >= 2.0,
        "cagr_full_ge_0.20":  metrics["Full_cagr"]   >= 0.20,
        "IS_sharpe_ge_1.5":   metrics["IS_sharpe"]   >= 1.5,
        "OOS_sharpe_ge_1.5":  metrics["OOS_sharpe"]  >= 1.5,
        "IS_OOS_gap_le_0.5":  metrics["sharpe_gap_IS_OOS"] <= 0.5,
    }
    metrics["requirements"] = req

    # Reports
    print("\n=== ORION — Orthogonal Signal Ensemble ===")
    print(f"Universe: {len(UNIVERSE)} LETFs (RISK={len(RISK_UNIVERSE)}, SAFE={len(SAFE_UNIVERSE)})")
    print(f"Window:   {returns.index[0].date()}  ->  {returns.index[-1].date()}")
    print(f"Params:   K_risk={K_RISK}, K_safe={K_SAFE}, weekly DOW={REBAL_DOW}, "
          f"VIX<{VIX_HI}, HY<{HY_HI}, weights={RISK_WEIGHT}/{SAFE_WEIGHT}")

    print("\n-- Individual signal standalone Sharpe --")
    for k, v in per_signal_sharpe.items():
        print(f"  {k:20s}: {v:6.3f}")

    print("\n-- Signal return correlation matrix (orthogonality proof) --")
    print(corr)

    n = len(corr)
    off = corr.values[np.triu_indices(n, 1)]
    print(f"  avg pair-corr        = {off.mean():.3f}")
    print(f"  min / max pair-corr  = {off.min():.3f} / {off.max():.3f}")

    print("\n-- Sleeve diagnostics --")
    for k, v in sleeve_sharpe.items():
        print(f"  {k:14s}: Sharpe={v:.3f}")
    print("  Sleeve corr:")
    print(sleeve_corr)

    print("\n-- ORION portfolio metrics --")
    for lbl in ["IS", "OOS", "Full"]:
        print(f"  {lbl:4s}  Sharpe={metrics[f'{lbl}_sharpe']:5.2f}  "
              f"CAGR={metrics[f'{lbl}_cagr']*100:6.2f}%  "
              f"Vol={metrics[f'{lbl}_vol']*100:5.2f}%  "
              f"MDD={metrics[f'{lbl}_mdd']*100:6.2f}%")
    print(f"  IS/OOS Sharpe gap                : {metrics['sharpe_gap_IS_OOS']:.3f}")
    print(f"  Avg turnover (annualised sum|dw|): {metrics['avg_turnover_annualised']:.2f}")

    print("\n-- Hard requirements --")
    for k, v in req.items():
        print(f"  {k:25s}: {'PASS' if v else 'FAIL'}")

    # Persist
    returns.to_csv(OUT_DIR / "orion_returns.csv", header=True)
    with open(OUT_DIR / "orion_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nSaved: {OUT_DIR/'orion_metrics.json'}")
    print(f"Saved: {OUT_DIR/'orion_returns.csv'}")

    return metrics


if __name__ == "__main__":
    main()
