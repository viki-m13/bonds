"""ML v4 — walk-forward retraining on rolling 5-year window.

Phoenix's Quantum trains ONCE on 2010-2018 and freezes. That works because
the 2010-2018 training set is large, clean (real LETFs), and matches the
post-QE regime that persisted into OOS.

My IS 2005-2018 mixes synthetic pre-2010 data with real post-2010, creating
regime distribution shift. Solution: WALK-FORWARD RETRAINING on a rolling
5-year window. At each year boundary, retrain on [year-5, year-1] and predict
for year.

This should reduce the IS/OOS gap by keeping the model on recent data.
Combined with HEAVY regularization (smaller model, more dropout).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
import util
from ml_v3 import build_features, UNIVERSE, FRED

ETF = Path("/home/user/bonds/data/etfs")


def make_small_model(seed=42):
    return xgb.XGBRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.02,
        min_child_weight=80, subsample=0.7, colsample_bytree=0.5,
        reg_lambda=20.0, reg_alpha=2.0, gamma=0.2,
        n_jobs=4, verbosity=0, random_state=seed, tree_method="hist",
    )


def walk_forward_predict(cp: pd.DataFrame, N_fwd: int = 21,
                         window_years: int = 5, bags: int = 5,
                         min_train_year: int = 2010) -> pd.DataFrame:
    """Walk-forward: for each year starting from min_train_year + window_years,
    train on prior `window_years` years, predict for that year."""
    long_df, feat_cols = build_features(cp)

    # Build target (vol-scaled)
    fwd_rows = []
    for tic in UNIVERSE:
        if tic not in cp.columns:
            continue
        p = cp[tic]
        lr = np.log(p / p.shift(1))
        vol_21 = lr.rolling(21).std()
        fwd = np.log(p.shift(-N_fwd) / p)
        y = fwd / vol_21.replace(0, np.nan)
        fwd_rows.append(pd.DataFrame({"Date": cp.index, "Ticker": tic, "y": y.values}))
    fwd_df = pd.concat(fwd_rows, ignore_index=True)
    data = long_df.merge(fwd_df, on=["Date", "Ticker"], how="left")

    all_years = sorted(data["Date"].dt.year.unique())
    preds_wide = pd.DataFrame(np.nan, index=cp.index, columns=UNIVERSE)

    for y in all_years:
        if y < min_train_year + window_years:
            continue
        tr_start = pd.Timestamp(f"{y - window_years}-01-01")
        tr_end = pd.Timestamp(f"{y - 1}-12-31") - pd.Timedelta(days=N_fwd + 10)
        tr = data[(data["Date"] >= tr_start) & (data["Date"] <= tr_end)].dropna(
            subset=feat_cols + ["y"])
        if len(tr) < 2000:
            continue
        # Bag models
        bag_preds = None
        rng = np.random.default_rng(y)
        dates = sorted(tr["Date"].unique())
        for i in range(bags):
            sample_dates = rng.choice(dates, int(len(dates) * 0.8), replace=False)
            tr_b = tr[tr["Date"].isin(sample_dates)]
            m = make_small_model(seed=y * 100 + i)
            m.fit(tr_b[feat_cols].values, tr_b["y"].values)
            # Predict for year y
            year_mask = (data["Date"] >= pd.Timestamp(f"{y}-01-01")) & (data["Date"] <= pd.Timestamp(f"{y}-12-31"))
            pred_df = data[year_mask].dropna(subset=feat_cols).copy()
            if len(pred_df) == 0:
                continue
            p = m.predict(pred_df[feat_cols].values)
            if bag_preds is None:
                bag_preds = pd.DataFrame({"Date": pred_df["Date"].values,
                                          "Ticker": pred_df["Ticker"].values,
                                          "pred": p})
            else:
                bag_preds["pred"] = bag_preds["pred"].values + p
        if bag_preds is None:
            continue
        bag_preds["pred"] = bag_preds["pred"] / bags
        for _, row in bag_preds.iterrows():
            preds_wide.loc[row["Date"], row["Ticker"]] = row["pred"]

    return preds_wide


def multi_horizon_wf(cp: pd.DataFrame, window_years=5, bags=5):
    """Walk-forward, multiple horizons, blend."""
    weights = {5: 0.25, 21: 0.50, 63: 0.25}
    composite = None
    for N, w in weights.items():
        preds = walk_forward_predict(cp, N_fwd=N, window_years=window_years, bags=bags)
        # Rank-normalize per day
        ranked = preds.rank(axis=1, pct=True) - 0.5
        if composite is None:
            composite = w * ranked
        else:
            composite = composite + w * ranked
    return composite


def sleeve_ml_v4(cp: pd.DataFrame, k_top=3, rebal_every=10,
                 target_vol=0.20) -> pd.DataFrame:
    print("[ML v4] Walk-forward multi-horizon training...")
    preds = multi_horizon_wf(cp, window_years=5, bags=5)

    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_rebal = mask % rebal_every == 0
    rnk = preds.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= k_top)
    sel_m = sel.where(is_rebal).ffill().fillna(False)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for tic in UNIVERSE:
        if tic in cp.columns and tic in sel_m.columns:
            W[tic] = sel_m[tic].astype(float) / k_top

    # Scale
    w = W.fillna(0.0)
    r = (w.shift(1).fillna(0.0) * cp.pct_change().fillna(0.0)).sum(axis=1)
    rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return W.mul(m, axis=0)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/bonds/apex")
    op, cp = util.load_prices()
    W = sleeve_ml_v4(cp)
    rets = cp.pct_change()
    r = (W.shift(1).fillna(0.0) * rets.reindex_like(W).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = W.diff().abs().fillna(W.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in W.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    r = r - drag

    util.summarize(r, "ML_v4 FULL")
    util.summarize(util.regime_slice(r, "2015-01-01", "2018-12-31"), "IS 15-18")
    util.summarize(util.regime_slice(r, "2019-01-02", "2027-12-31"), "OOS 19+")
    util.summarize(util.regime_slice(r, "2022-01-01", "2022-12-31"), "2022")
    util.summarize(util.regime_slice(r, "2020-01-01", "2020-12-31"), "COVID")

    r.to_frame("ml_v4").to_csv("/home/user/bonds/data/apex/ml_v4_returns.csv")
    W.to_csv("/home/user/bonds/data/apex/ml_v4_weights.csv")
