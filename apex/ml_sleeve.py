"""APEX — ML sleeve v2 (fixed).

Trains XGBoost on IS (2005-2018) to predict each LETF's next-21-day log return.
Fixed issues: stack long-format properly, use Date as primary key, predict on
full data (including OOS) from a single IS-trained model.
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

UNIVERSE = ["UPRO", "TQQQ", "TECL", "SOXL", "FAS", "EDC", "YINN",
            "TMF", "UBT", "TYD", "UGL", "UCO", "DRN"]

IS_START = "2005-01-03"
IS_END = "2018-12-31"

N_FORWARD = 21
K_TOP = 3
REBAL_EVERY = 21


def _fred(name, idx):
    fp = FRED / f"{name}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[df.columns[0]].astype(float).reindex(idx).ffill()


def build_features_long(cp: pd.DataFrame) -> pd.DataFrame:
    """Build long-format DataFrame: rows = (Date, tic), cols = features + y."""
    spy = cp["SPY"]
    slope = _fred("T10Y2Y", cp.index)

    # Macro features (skip HY — data only from 2023 in this repo)
    mac = pd.DataFrame(index=cp.index)
    mac["curve"] = slope
    mac["spy_rv21"] = spy.pct_change().rolling(21).std() * np.sqrt(util.DPY)
    mac["spy_rv63"] = spy.pct_change().rolling(63).std() * np.sqrt(util.DPY)
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
        df["dist_200"] = p / p.rolling(200).mean() - 1
        for c in mac.columns:
            df[c] = mac[c]
        df["y"] = np.log(p.shift(-N_FORWARD) / p)
        df["tic"] = tic
        df = df.reset_index().rename(columns={"index": "Date"})
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def train_predict(cp: pd.DataFrame) -> pd.DataFrame:
    """Returns wide (Date x tic) predicted next-21d log return."""
    long_df = build_features_long(cp)
    # Cross-sectional ranks per Date for each feature
    feat_base = [c for c in long_df.columns if c not in ("Date", "tic", "y")]
    for c in feat_base:
        long_df[c + "_r"] = long_df.groupby("Date")[c].rank(pct=True)

    feat_cols = [c for c in long_df.columns if c not in ("Date", "tic", "y")]

    train_mask = ((long_df["Date"] >= pd.Timestamp(IS_START)) &
                  (long_df["Date"] <= pd.Timestamp(IS_END) - pd.Timedelta(days=N_FORWARD + 5)))
    train = long_df[train_mask].dropna(subset=feat_cols + ["y"])
    print(f"[ML] train rows: {len(train)}, features: {len(feat_cols)}")

    # Split train into train/validation (last 15% of train dates for val)
    train_dates_sorted = sorted(train["Date"].unique())
    cutoff = train_dates_sorted[int(len(train_dates_sorted) * 0.85)]
    tr = train[train["Date"] <= cutoff]
    va = train[train["Date"] > cutoff]
    # Heavily regularized, small ensemble, early stopping
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.02,
        min_child_weight=50, subsample=0.7, colsample_bytree=0.6,
        reg_lambda=10.0, reg_alpha=1.0, gamma=0.1,
        n_jobs=4, verbosity=0,
        early_stopping_rounds=20,
    )
    model.fit(tr[feat_cols], tr["y"],
              eval_set=[(va[feat_cols], va["y"])], verbose=False)
    print(f"[ML] best_iter: {model.best_iteration}")

    # Predict across ALL rows (including OOS)
    valid = long_df.dropna(subset=feat_cols).copy()
    valid["pred"] = model.predict(valid[feat_cols])

    wide = valid.pivot(index="Date", columns="tic", values="pred")
    wide = wide.reindex(cp.index)
    return wide


def sleeve_ml(cp: pd.DataFrame, target_vol: float = 0.20,
              k_top: int = K_TOP, rebal: int = REBAL_EVERY) -> pd.DataFrame:
    """Returns weights DataFrame (T x N_assets)."""
    preds = train_predict(cp)
    idx = cp.index

    mask = pd.Series(range(len(idx)), index=idx)
    is_rebal = mask % rebal == 0

    rnk = preds.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= k_top)
    sel_m = sel.where(is_rebal).ffill().fillna(False)
    n_sel = sel_m.sum(axis=1)

    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    for tic in UNIVERSE:
        if tic not in cp.columns or tic not in sel_m.columns:
            continue
        W[tic] = (sel_m[tic].astype(float) / n_sel.replace(0, np.nan)).fillna(0.0)

    return W


if __name__ == "__main__":
    op, cp = util.load_prices()
    W = sleeve_ml(cp)
    # Diagnose
    print(f"Total days with non-zero weights: {(W.sum(axis=1) > 0).sum()}")
    print(f"Mean gross weight: {W.sum(axis=1).mean():.3f}")
    # Compute naive return
    rets = cp.pct_change()
    r = (W.shift(1).fillna(0.0) * rets.reindex_like(W).fillna(0.0)).sum(axis=1)
    util.summarize(r, "ML FULL")
    util.summarize(util.regime_slice(r, "2005-01-01", "2018-12-31"), "IS 05-18")
    util.summarize(util.regime_slice(r, "2019-01-02", "2027-12-31"), "OOS 19+")
    util.summarize(util.regime_slice(r, "2022-01-01", "2022-12-31"), "2022")
    util.summarize(util.regime_slice(r, "2007-01-01", "2009-12-31"), "GFC")
    r.to_frame("ml").to_csv("/home/user/bonds/data/apex/ml_sleeve_v2.csv")
    W.to_csv("/home/user/bonds/data/apex/ml_sleeve_v2_weights.csv")
