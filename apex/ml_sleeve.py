"""APEX — ML sleeve (XGBoost regressor on per-asset features).

Trains on IS (2005-2018) to predict each LETF's next-21-day log return.
Rebalances every 21 days; holds top K by predicted return.
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
OOS_START = "2019-01-02"

N_FORWARD = 21
K_TOP = 3
REBAL_EVERY = 21


def _fred(name, idx):
    fp = FRED / f"{name}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[df.columns[0]].astype(float).reindex(idx).ffill()


def feature_frame_for(tic: str, cp: pd.DataFrame,
                      macro: pd.DataFrame) -> pd.DataFrame:
    p = cp[tic]
    lr = np.log(p / p.shift(1))
    spy = cp["SPY"]
    spy_lr = np.log(spy / spy.shift(1))
    feat = pd.DataFrame(index=cp.index)
    feat["mom_5"] = np.log(p / p.shift(5))
    feat["mom_21"] = np.log(p / p.shift(21))
    feat["mom_63"] = np.log(p / p.shift(63))
    feat["mom_126"] = np.log(p / p.shift(126))
    feat["mom_252"] = np.log(p / p.shift(252))
    feat["vol_21"] = lr.rolling(21).std()
    feat["vol_63"] = lr.rolling(63).std()
    feat["sr_21"] = lr.rolling(21).mean() / lr.rolling(21).std().replace(0, np.nan)
    feat["sr_63"] = lr.rolling(63).mean() / lr.rolling(63).std().replace(0, np.nan)
    feat["rel_spy_21"] = feat["mom_21"] - np.log(spy / spy.shift(21))
    feat["rel_spy_63"] = feat["mom_63"] - np.log(spy / spy.shift(63))
    feat["dist_200"] = p / p.rolling(200).mean() - 1
    for c in macro.columns:
        feat[c] = macro[c]
    return feat


def train_predict(cp: pd.DataFrame) -> pd.DataFrame:
    """Returns a (Date, Ticker) wide DF of predicted next-21d return."""
    spy = cp["SPY"]
    hy = _fred("BAMLH0A0HYM2", cp.index)
    slope = _fred("T10Y2Y", cp.index)

    macro = pd.DataFrame(index=cp.index)
    macro["hy_level"] = hy
    macro["hy_slope"] = hy - hy.rolling(21).mean()
    macro["curve"] = slope
    macro["spy_rv21"] = spy.pct_change().rolling(21).std() * np.sqrt(util.DPY)
    macro["spy_ma_spread"] = spy.rolling(21).mean() / spy.rolling(63).mean() - 1
    macro = macro.ffill()

    # Stack all tickers' feature frames into one long DF
    long_rows = []
    for tic in UNIVERSE:
        if tic not in cp.columns:
            continue
        f = feature_frame_for(tic, cp, macro)
        p = cp[tic]
        f["y"] = np.log(p.shift(-N_FORWARD) / p)
        f["tic"] = tic
        long_rows.append(f.reset_index().rename(columns={"index": "Date"}))

    long_df = pd.concat(long_rows, ignore_index=True)
    # Cross-sectional ranks per Date for each feature
    feat_cols = [c for c in long_df.columns if c not in ("Date", "tic", "y")]
    for c in feat_cols:
        long_df[c + "_r"] = long_df.groupby("Date")[c].rank(pct=True)

    all_feats = [c for c in long_df.columns if c not in ("Date", "tic", "y")]
    train = long_df[(long_df["Date"] >= pd.Timestamp(IS_START)) &
                    (long_df["Date"] <= pd.Timestamp(IS_END) - pd.Timedelta(days=N_FORWARD + 5))]
    train = train.dropna(subset=all_feats + ["y"])
    X_train = train[all_feats]
    y_train = train["y"]

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        min_child_weight=20, subsample=0.7, colsample_bytree=0.7,
        reg_lambda=5.0, n_jobs=4, verbosity=0,
    )
    model.fit(X_train, y_train)

    # Predict on ALL (drop any rows with missing features)
    pred_df = long_df.dropna(subset=all_feats).copy()
    pred_df["pred"] = model.predict(pred_df[all_feats])

    # Pivot to Date x tic
    pred_wide = pred_df.pivot(index="Date", columns="tic", values="pred")
    pred_wide = pred_wide.reindex(cp.index)
    return pred_wide


def sleeve_ml(cp: pd.DataFrame, target_vol: float = 0.10,
              k_top: int = K_TOP, rebal_every: int = REBAL_EVERY) -> pd.Series:
    preds = train_predict(cp)
    idx = cp.index

    mask = pd.Series(range(len(idx)), index=idx)
    is_rebal = mask % rebal_every == 0

    rnk = preds.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= k_top)
    sel_m = sel.where(is_rebal).ffill().fillna(False)
    n_sel = sel_m.sum(axis=1)

    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    for tic in UNIVERSE:
        if tic not in cp.columns or tic not in sel_m.columns:
            continue
        W[tic] = (sel_m[tic].astype(float) / n_sel.replace(0, np.nan)).fillna(0.0)

    rets = cp.pct_change()
    w_eff = W.shift(1).fillna(0.0)
    pr = (w_eff * rets.reindex_like(W).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = W.diff().abs().fillna(W.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in W.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    pr = pr - drag

    rv = pr.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(lower=0.2, upper=2.5).shift(1).fillna(1.0)
    return pr * m


if __name__ == "__main__":
    op, cp = util.load_prices()
    print("Training ML sleeve...")
    r = sleeve_ml(cp, target_vol=0.10)
    util.summarize(r, "ML FULL")
    util.summarize(util.regime_slice(r, "2005-01-01", "2018-12-31"), "IS 05-18")
    util.summarize(util.regime_slice(r, "2019-01-02", "2027-12-31"), "OOS 19+")
    util.summarize(util.regime_slice(r, "2022-01-01", "2022-12-31"), "2022")
    util.summarize(util.regime_slice(r, "2007-01-01", "2009-12-31"), "GFC")
    r.to_frame("ml").to_csv("/home/user/bonds/data/apex/ml_sleeve.csv")
