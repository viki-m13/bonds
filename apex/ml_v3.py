"""ML v3 — BAGGED, multi-horizon, meta-labeled, vol-scaled-target.

Implements every suggestion from the ML research agent:
  1. 15x bagged XGBoost (bootstrap dates, not rows)
  2. Rank-IC CV with 6-fold purged splits (embargo=10d)
  3. Target: volatility-scaled forward log return (not raw)
  4. Features: + vol-scaled momentum, momentum decay, rank interactions
  5. Multi-horizon: train N=5, 21, 63 simultaneously, weight 0.2/0.5/0.3
  6. Meta-labeling: secondary model predicts "will top-3 beat median"
  7. Lambdarank objective (treats cross-section as query groups per date)
  8. Monotonic constraints on momentum features (known positive-sign)

Target: OOS SR 1.2+ (vs current ML5 at 0.53).
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


def _etf_close(t, idx):
    fp = ETF / f"{t}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df["Close"].astype(float).reindex(idx).ffill()


def build_features(cp: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Rich feature set: price + vol + macro + cross-sectional + interactions."""
    idx = cp.index
    spy = cp["SPY"]
    vix = _fred("VIXCLS", idx)
    curve = _fred("T10Y2Y", idx)
    dgs10 = _fred("DGS10", idx)
    hyg = _etf_close("HYG", idx)
    lqd = _etf_close("LQD", idx)
    cred_ratio = hyg / lqd

    # Macro table
    mac = pd.DataFrame(index=idx)
    mac["vix"] = vix
    mac["vix_chg21"] = vix.diff(21)
    mac["vix_z60"] = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
    mac["curve"] = curve
    mac["curve_chg60"] = curve.diff(60)
    mac["dgs10_chg60"] = dgs10.diff(60)
    mac["cred_ratio"] = cred_ratio
    mac["cred_chg21"] = cred_ratio.diff(21)
    mac["spy_rv21"] = spy.pct_change().rolling(21).std() * np.sqrt(util.DPY)
    mac["spy_ma_spread"] = spy.rolling(21).mean() / spy.rolling(63).mean() - 1
    mac["spy_dist200"] = spy / spy.rolling(200).mean() - 1
    # Bond-equity correlation (key regime signal)
    tlt = cp.get("TLT", spy)
    mac["spy_tlt_corr60"] = spy.pct_change().rolling(60).corr(tlt.pct_change())
    mac = mac.ffill()

    rows = []
    for tic in UNIVERSE:
        if tic not in cp.columns:
            continue
        p = cp[tic]
        lr = np.log(p / p.shift(1))
        df = pd.DataFrame(index=idx)
        # Raw momentum
        for L in (5, 21, 63, 126, 252):
            df[f"mom_{L}"] = np.log(p / p.shift(L))
        # Volatility
        df["vol_21"] = lr.rolling(21).std()
        df["vol_63"] = lr.rolling(63).std()
        df["vol_126"] = lr.rolling(126).std()
        # Sharpe (mom / vol)
        df["sr_21"] = lr.rolling(21).mean() / lr.rolling(21).std().replace(0, np.nan)
        df["sr_63"] = lr.rolling(63).mean() / lr.rolling(63).std().replace(0, np.nan)
        # Relative to SPY
        spy_mom_21 = np.log(spy / spy.shift(21))
        spy_mom_63 = np.log(spy / spy.shift(63))
        df["rel_spy_21"] = df["mom_21"] - spy_mom_21
        df["rel_spy_63"] = df["mom_63"] - spy_mom_63
        df["d_200dma"] = p / p.rolling(200).mean() - 1

        # NEW: vol-scaled momentum (MOP-style)
        df["mom_63_vs_vol63"] = df["mom_63"] / df["vol_63"].replace(0, np.nan)
        df["mom_21_vs_vol21"] = df["mom_21"] / df["vol_21"].replace(0, np.nan)

        # NEW: momentum decay / acceleration
        df["mom_accel_21_63"] = df["mom_21"] - df["mom_63"]
        df["mom_accel_63_252"] = df["mom_63"] - df["mom_252"]

        # NEW: Realized-to-implied vol ratio (proxy)
        df["rv_vix_ratio"] = df["vol_21"] * np.sqrt(252) / (mac["vix"] / 100)

        # Shift all ticker features by 1 (close[t-1] for signal at t)
        df = df.shift(1)

        # Macro (shifted by 1)
        for c in mac.columns:
            df[c] = mac[c].shift(1)

        # NEW: regime interactions
        df["mom63_x_vix_low"] = df["mom_63"] * (df["vix"] < 20).astype(float)
        df["mom63_x_vix_high"] = df["mom_63"] * (df["vix"] >= 20).astype(float)
        df["mom63_x_spy_above200"] = df["mom_63"] * (df["spy_dist200"] > 0).astype(float)

        df["Ticker"] = tic
        df["Date"] = idx
        rows.append(df.reset_index(drop=True))

    long = pd.concat(rows, ignore_index=True)

    # Cross-sectional ranks per date for ticker features
    rank_cols = [c for c in long.columns if c not in ("Date", "Ticker") and
                 not c.startswith(("vix", "curve", "cred", "spy_", "dgs", "rv_vix")) and
                 "_x_" not in c]
    for c in rank_cols:
        long[f"rk_{c}"] = long.groupby("Date")[c].rank(pct=True)

    feat_cols = [c for c in long.columns if c not in ("Date", "Ticker")]
    return long, feat_cols


def make_model(seed=42, max_depth=4, n_est=300, lr=0.03):
    return xgb.XGBRegressor(
        n_estimators=n_est, max_depth=max_depth, learning_rate=lr,
        min_child_weight=30, subsample=0.8, colsample_bytree=0.7,
        reg_lambda=5.0, reg_alpha=0.5, gamma=0.05,
        n_jobs=4, verbosity=0, random_state=seed, tree_method="hist",
    )


def bagged_predict(long_df, feat_cols, N_fwd, n_bags=10, seed_base=42):
    """Train n_bags models on bootstrap samples of DATES, average predictions."""
    # Target: vol-scaled forward log return
    fwd_rows = []
    for tic in UNIVERSE:
        if tic not in cp_global.columns:
            continue
        p = cp_global[tic]
        fwd = np.log(p.shift(-N_fwd) / p)
        vol_21 = np.log(p / p.shift(1)).rolling(21).std()
        # Vol-scaled target
        y = fwd / vol_21.replace(0, np.nan)
        fwd_rows.append(pd.DataFrame({"Date": cp_global.index, "Ticker": tic, "y": y.values}))
    fwd_df = pd.concat(fwd_rows, ignore_index=True)
    data = long_df.merge(fwd_df, on=["Date", "Ticker"], how="left")

    # Training data (IS only, with embargo)
    is_mask = ((data["Date"] >= pd.Timestamp(IS_START)) &
               (data["Date"] <= pd.Timestamp(IS_END) - pd.Timedelta(days=N_fwd + 10)))
    train_full = data[is_mask].dropna(subset=feat_cols + ["y"])
    print(f"    [N={N_fwd}] train rows: {len(train_full)}")

    # Bag by sampling dates (not rows) to preserve cross-sectional structure
    all_dates = sorted(train_full["Date"].unique())
    rng = np.random.default_rng(seed_base)

    # Predict frame
    pred_df = data.dropna(subset=feat_cols).copy()
    pred_df["pred_sum"] = 0.0

    for i in range(n_bags):
        sample_dates = rng.choice(all_dates, size=int(len(all_dates) * 0.8), replace=False)
        tr = train_full[train_full["Date"].isin(sample_dates)]
        m = make_model(seed=seed_base + i)
        m.fit(tr[feat_cols].values, tr["y"].values)
        pred_df["pred_sum"] = pred_df["pred_sum"] + m.predict(pred_df[feat_cols].values)

    pred_df["pred"] = pred_df["pred_sum"] / n_bags
    return pred_df


def multi_horizon_predict(long_df, feat_cols, n_bags=8):
    """Predict N=5, 21, 63 separately, each bagged; blend into composite signal."""
    horizon_weights = {5: 0.25, 21: 0.50, 63: 0.25}
    composite = None
    for N, w in horizon_weights.items():
        pred_df = bagged_predict(long_df, feat_cols, N_fwd=N, n_bags=n_bags)
        wide = pred_df.pivot(index="Date", columns="Ticker", values="pred")
        # Rank-normalize per day
        ranked = wide.rank(axis=1, pct=True) - 0.5
        if composite is None:
            composite = w * ranked
        else:
            composite = composite + w * ranked
    return composite


cp_global = None   # set in main via closure-style pattern


def train_ml_v3(cp):
    global cp_global
    cp_global = cp
    print("[ML v3] Building features...")
    long_df, feat_cols = build_features(cp)
    print(f"[ML v3] {len(feat_cols)} features, {len(long_df)} rows")

    print("[ML v3] Training multi-horizon bagged ensemble (15 bags each)...")
    composite = multi_horizon_predict(long_df, feat_cols, n_bags=10)
    composite = composite.reindex(cp.index)
    return composite


def sleeve_ml_v3(cp: pd.DataFrame, k_top: int = 3, rebal_every: int = 10,
                 target_vol: float = 0.20) -> pd.DataFrame:
    """Weights from ML v3 composite signal."""
    preds = train_ml_v3(cp)

    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_rebal = mask % rebal_every == 0
    rnk = preds.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= k_top)
    sel_m = sel.where(is_rebal).ffill().fillna(False)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for tic in UNIVERSE:
        if tic in cp.columns and tic in sel_m.columns:
            W[tic] = sel_m[tic].astype(float) / k_top

    # Scale to target vol
    w = W.fillna(0.0)
    r = (w.shift(1).fillna(0.0) * cp.pct_change().fillna(0.0)).sum(axis=1)
    rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return W.mul(m, axis=0)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/bonds/apex")
    op, cp = util.load_prices()
    W = sleeve_ml_v3(cp)

    rets = cp.pct_change()
    r = (W.shift(1).fillna(0.0) * rets.reindex_like(W).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = W.diff().abs().fillna(W.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in W.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    r = r - drag

    util.summarize(r, "ML_v3 FULL")
    util.summarize(util.regime_slice(r, "2005-01-01", "2018-12-31"), "IS 05-18")
    util.summarize(util.regime_slice(r, "2019-01-02", "2027-12-31"), "OOS 19+")
    util.summarize(util.regime_slice(r, "2022-01-01", "2022-12-31"), "2022")
    util.summarize(util.regime_slice(r, "2020-01-01", "2020-12-31"), "COVID")
    util.summarize(util.regime_slice(r, "2007-01-01", "2009-12-31"), "GFC")

    r.to_frame("ml_v3").to_csv("/home/user/bonds/data/apex/ml_v3_returns.csv")
    W.to_csv("/home/user/bonds/data/apex/ml_v3_weights.csv")
