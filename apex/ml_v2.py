"""APEX — ML sleeve v2, Phoenix-Quantum-style architecture.

Key differences vs my earlier attempt:
  1. Multi-horizon CV: train per N ∈ {5, 10, 21, 42} days; score via rank-IC
     (cross-sectional Spearman correlation of predicted vs realized return).
     Select best (N, K) on IS only.
  2. No early-stopping/val-split leakage: full IS used to train the final model
     after CV hyperparameter selection.
  3. Cross-sectional rank features (key for Phoenix quality).
  4. Features shifted by 1 so close[t-1] data is used for signal date t.
  5. Tight regularization: depth=4, min_child_weight=20, reg_lambda=5.
  6. Uses VIXCLS (1990+) and proxy-credit (HYG/LQD price-spread) instead
     of the truncated BAMLH OAS.

Returns a weights DataFrame (T x N_assets). Rebalance every N days, equal-
weight top-K, weights sum to K·(1/K) = 1 on rebal days (no cash fallback).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

import xgboost as xgb
import util

ROOT = Path("/home/user/bonds")
FRED = ROOT / "data/fred"
ETF = ROOT / "data/etfs"

UNIVERSE = ["UPRO", "TQQQ", "TECL", "SOXL", "FAS", "EDC", "YINN",
            "TMF", "UBT", "TYD", "UGL", "UCO", "DRN"]

IS_START = "2005-01-03"
IS_END = "2018-12-31"


def _fred(name, idx):
    fp = FRED / f"{name}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[df.columns[0]].astype(float).reindex(idx).ffill()


def _load_etf_close(ticker, idx):
    fp = ETF / f"{ticker}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df["Close"].astype(float).reindex(idx).ffill()


def build_features(cp: pd.DataFrame) -> pd.DataFrame:
    """Long-format DataFrame: rows = (Date, Ticker), columns = features."""
    spy = cp["SPY"]
    vix = _fred("VIXCLS", cp.index)    # 1990+
    curve = _fred("T10Y2Y", cp.index)  # 2000+
    # Credit proxy: HYG close / LQD close spread ratio (HYG starts 2007)
    hyg = _load_etf_close("HYG", cp.index)
    lqd = _load_etf_close("LQD", cp.index)
    cred_ratio = (hyg / lqd).rename("cred_ratio")

    mac = pd.DataFrame(index=cp.index)
    mac["vix"] = vix
    mac["vix_chg21"] = vix.diff(21)
    mac["curve"] = curve
    mac["curve_chg60"] = curve.diff(60)
    mac["cred_ratio"] = cred_ratio
    mac["cred_chg21"] = cred_ratio.diff(21)
    mac["spy_rv21"] = spy.pct_change().rolling(21).std() * np.sqrt(util.DPY)
    mac["spy_ma_spread"] = spy.rolling(21).mean() / spy.rolling(63).mean() - 1
    mac["spy_dist200"] = spy / spy.rolling(200).mean() - 1
    mac = mac.ffill()

    rows = []
    for tic in UNIVERSE:
        if tic not in cp.columns:
            continue
        p = cp[tic]
        lr = np.log(p / p.shift(1))
        df = pd.DataFrame(index=cp.index)
        df["mom_5"] = np.log(p / p.shift(5))
        df["mom_21"] = np.log(p / p.shift(21))
        df["mom_63"] = np.log(p / p.shift(63))
        df["mom_126"] = np.log(p / p.shift(126))
        df["mom_252"] = np.log(p / p.shift(252))
        df["vol_21"] = lr.rolling(21).std()
        df["vol_63"] = lr.rolling(63).std()
        df["sr_21"] = lr.rolling(21).mean() / lr.rolling(21).std().replace(0, np.nan)
        df["sr_63"] = lr.rolling(63).mean() / lr.rolling(63).std().replace(0, np.nan)
        df["rel_spy_21"] = df["mom_21"] - np.log(spy / spy.shift(21))
        df["rel_spy_63"] = df["mom_63"] - np.log(spy / spy.shift(63))
        df["d_200dma"] = p / p.rolling(200).mean() - 1
        # Shift ticker features by 1 (use close[t-1] for signal at t)
        df = df.shift(1)
        # Macro also shifted by 1
        for c in mac.columns:
            df[c] = mac[c].shift(1)
        df["Ticker"] = tic
        df["Date"] = cp.index
        rows.append(df.reset_index(drop=True))

    long = pd.concat(rows, ignore_index=True)
    # Cross-sectional ranks per Date for ticker features (pct rank 0..1)
    rank_cols = ["mom_5", "mom_21", "mom_63", "mom_126", "mom_252",
                 "vol_21", "vol_63", "sr_21", "sr_63",
                 "rel_spy_21", "rel_spy_63", "d_200dma"]
    for c in rank_cols:
        long[f"rk_{c}"] = long.groupby("Date")[c].rank(pct=True)
    return long


def build_targets(cp: pd.DataFrame, N: int) -> pd.DataFrame:
    rows = []
    for tic in UNIVERSE:
        if tic not in cp.columns:
            continue
        p = cp[tic]
        fwd = np.log(p.shift(-N) / p)
        rows.append(pd.DataFrame({"Date": cp.index, "Ticker": tic, f"fwd_{N}": fwd.values}))
    return pd.concat(rows, ignore_index=True)


def make_model(seed=42):
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        min_child_weight=20, subsample=0.7, colsample_bytree=0.7,
        reg_lambda=5.0, n_jobs=4, verbosity=0,
        random_state=seed, tree_method="hist",
    )


def rank_ic(y_true, y_pred, dates):
    """Cross-sectional Spearman between pred and actual, averaged across dates."""
    df = pd.DataFrame({"y": y_true, "yp": y_pred, "d": dates})
    ics = []
    for _, g in df.groupby("d"):
        if len(g) < 5:
            continue
        c = g["y"].rank().corr(g["yp"].rank())
        if np.isfinite(c):
            ics.append(c)
    return float(np.mean(ics)) if ics else float("nan")


def cv_select(feats: pd.DataFrame, feat_cols: list, targets: dict) -> dict:
    """4-fold walk-forward CV inside IS, picks best (N, K) by mean rank-IC."""
    dates_sorted = sorted(feats["Date"].unique())
    n_folds = 4
    fold_size = len(dates_sorted) // (n_folds + 1)

    best = None
    best_ic = -np.inf
    scores = {}
    for N in (5, 10, 21, 42):
        y_col = f"fwd_{N}"
        data = feats.merge(targets[N], on=["Date", "Ticker"], how="left")
        fold_ics = []
        for f in range(n_folds):
            tr_end_i = fold_size * (f + 1)
            va_start_i = tr_end_i + N   # embargo = N days
            va_end_i = min(tr_end_i + fold_size, len(dates_sorted) - 1)
            if va_start_i >= va_end_i:
                continue
            tr_end = dates_sorted[tr_end_i - 1]
            va_start = dates_sorted[va_start_i]
            va_end = dates_sorted[va_end_i - 1]
            tr = data[data["Date"] <= tr_end].dropna(subset=feat_cols + [y_col])
            va = data[(data["Date"] >= va_start) & (data["Date"] <= va_end)].dropna(
                subset=feat_cols + [y_col])
            if len(tr) < 2000 or len(va) < 200:
                continue
            m = make_model()
            m.fit(tr[feat_cols].values, tr[y_col].values)
            pred = m.predict(va[feat_cols].values)
            ic = rank_ic(va[y_col].values, pred, va["Date"].values)
            fold_ics.append(ic)
        mean_ic = float(np.nanmean(fold_ics)) if fold_ics else float("nan")
        scores[N] = {"mean_ic": mean_ic, "folds": fold_ics}
        if np.isfinite(mean_ic) and mean_ic > best_ic:
            best_ic = mean_ic
            best = N
    if best is None:
        best = 21
    return {"N": best, "ic": best_ic, "scores": scores}


def train_full_is(feats: pd.DataFrame, feat_cols: list, targets: pd.DataFrame, N: int):
    """Train on all IS data."""
    data = feats.merge(targets, on=["Date", "Ticker"], how="left")
    is_mask = ((data["Date"] >= pd.Timestamp(IS_START)) &
               (data["Date"] <= pd.Timestamp(IS_END) - pd.Timedelta(days=N + 5)))
    tr = data[is_mask].dropna(subset=feat_cols + [f"fwd_{N}"])
    print(f"[ML] training on {len(tr)} rows for N={N}")
    m = make_model()
    m.fit(tr[feat_cols].values, tr[f"fwd_{N}"].values)
    return m


def predict_all(m, feats: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    valid = feats.dropna(subset=feat_cols).copy()
    valid["pred"] = m.predict(valid[feat_cols].values)
    wide = valid.pivot(index="Date", columns="Ticker", values="pred")
    return wide


def sleeve_ml_weights(cp: pd.DataFrame, K: int = 3, rebal_every: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Returns (T x N_assets) weights and the CV results dict."""
    feats = build_features(cp)
    feat_cols = [c for c in feats.columns if c not in ("Date", "Ticker")]

    # Build targets for all horizons
    tgt_by_N = {N: build_targets(cp, N) for N in (5, 10, 21, 42)}

    # Restrict features to IS for CV
    feats_is = feats[(feats["Date"] >= pd.Timestamp(IS_START)) &
                      (feats["Date"] <= pd.Timestamp(IS_END))]
    cv = cv_select(feats_is, feat_cols, tgt_by_N)
    N = cv["N"]
    if rebal_every is None:
        rebal_every = N
    print(f"[ML] CV selected N={N}, ic={cv['ic']:.4f}, rebal_every={rebal_every}")

    # Train on full IS data
    m = train_full_is(feats, feat_cols, tgt_by_N[N], N)

    # Predict on full data
    preds = predict_all(m, feats, feat_cols)
    preds = preds.reindex(cp.index)

    # Top-K selection on rebal days
    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_rebal = mask % rebal_every == 0
    rnk = preds.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= K)
    sel_m = sel.where(is_rebal).ffill().fillna(False)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for tic in UNIVERSE:
        if tic not in cp.columns or tic not in sel_m.columns:
            continue
        W[tic] = sel_m[tic].astype(float) / K
    return W, cv


if __name__ == "__main__":
    op, cp = util.load_prices()
    W, cv = sleeve_ml_weights(cp, K=3)
    print(f"CV scores: {cv['scores']}")
    # compute returns
    rets = cp.pct_change()
    r = (W.shift(1).fillna(0.0) * rets.reindex_like(W).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = W.diff().abs().fillna(W.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in W.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    r = r - drag

    util.summarize(r, "ML FULL")
    util.summarize(util.regime_slice(r, "2005-01-01", "2018-12-31"), "IS 05-18")
    util.summarize(util.regime_slice(r, "2019-01-02", "2027-12-31"), "OOS 19+")
    util.summarize(util.regime_slice(r, "2022-01-01", "2022-12-31"), "2022")
    util.summarize(util.regime_slice(r, "2007-01-01", "2009-12-31"), "GFC")
    util.summarize(util.regime_slice(r, "2020-01-01", "2020-12-31"), "COVID")
    r.to_frame("ml_v2").to_csv("/home/user/bonds/data/apex/ml_v2_returns.csv")
    W.to_csv("/home/user/bonds/data/apex/ml_v2_weights.csv")
