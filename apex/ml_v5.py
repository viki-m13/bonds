"""ML v5 — train STRICTLY on 2010+ clean data (matches Phoenix exactly).

By training on 2010-2018 only, we get the clean data regime Phoenix used.
The pre-2008 synthetic data only used for STRESS TESTING, not training.
This matches Phoenix's approach exactly.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
import util
from ml_v3 import build_features, UNIVERSE

# NEW training window — matches Phoenix exactly
IS_START_10 = "2010-03-11"
IS_END = "2018-12-31"


def make_model(seed=42):
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        min_child_weight=20, subsample=0.7, colsample_bytree=0.7,
        reg_lambda=5.0, reg_alpha=0.0,
        n_jobs=4, verbosity=0, random_state=seed, tree_method="hist",
    )


def train_predict_2010_only(cp: pd.DataFrame, N_fwd: int = 21,
                              n_bags: int = 10) -> pd.DataFrame:
    """Train ONLY on 2010-2018, predict on all dates."""
    long_df, feat_cols = build_features(cp)

    # Vol-scaled target
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

    # IS: 2010-2018 ONLY (matches Phoenix)
    is_mask = ((data["Date"] >= pd.Timestamp(IS_START_10)) &
               (data["Date"] <= pd.Timestamp(IS_END) - pd.Timedelta(days=N_fwd + 5)))
    train_full = data[is_mask].dropna(subset=feat_cols + ["y"])
    print(f"    [N={N_fwd}] train rows (2010-2018 only): {len(train_full)}")

    all_dates = sorted(train_full["Date"].unique())
    rng = np.random.default_rng(42)

    pred_df = data.dropna(subset=feat_cols).copy()
    pred_sum = np.zeros(len(pred_df))

    for i in range(n_bags):
        sample_dates = rng.choice(all_dates, int(len(all_dates) * 0.80), replace=False)
        tr = train_full[train_full["Date"].isin(sample_dates)]
        m = make_model(seed=42 + i)
        m.fit(tr[feat_cols].values, tr["y"].values)
        pred_sum = pred_sum + m.predict(pred_df[feat_cols].values)

    pred_df["pred"] = pred_sum / n_bags
    wide = pred_df.pivot(index="Date", columns="Ticker", values="pred")
    return wide.reindex(cp.index)


def multi_horizon(cp: pd.DataFrame, n_bags: int = 10) -> pd.DataFrame:
    weights = {5: 0.25, 21: 0.50, 63: 0.25}
    composite = None
    for N, w in weights.items():
        preds = train_predict_2010_only(cp, N_fwd=N, n_bags=n_bags)
        ranked = preds.rank(axis=1, pct=True) - 0.5
        if composite is None:
            composite = w * ranked
        else:
            composite = composite + w * ranked
    return composite


def sleeve_ml_v5(cp: pd.DataFrame, k_top: int = 3, rebal_every: int = 10,
                 target_vol: float = 0.20) -> pd.DataFrame:
    """ML sleeve trained strictly on 2010-2018."""
    print("[ML v5] Training on 2010-2018 only (Phoenix-matching)...")
    preds = multi_horizon(cp, n_bags=10)

    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_rebal = mask % rebal_every == 0
    rnk = preds.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= k_top)
    sel_m = sel.where(is_rebal).ffill().fillna(False)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for tic in UNIVERSE:
        if tic in cp.columns and tic in sel_m.columns:
            W[tic] = sel_m[tic].astype(float) / k_top

    w = W.fillna(0.0)
    r = (w.shift(1).fillna(0.0) * cp.pct_change().fillna(0.0)).sum(axis=1)
    rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return W.mul(m, axis=0)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/bonds/apex")
    op, cp = util.load_prices()
    W = sleeve_ml_v5(cp)

    rets = cp.pct_change()
    r = (W.shift(1).fillna(0.0) * rets.reindex_like(W).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = W.diff().abs().fillna(W.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in W.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    r = r - drag

    util.summarize(r, "ML_v5 FULL")
    util.summarize(util.regime_slice(r, IS_START_10, IS_END), "IS 2010-2018 (training window)")
    util.summarize(util.regime_slice(r, "2019-01-02", "2027-12-31"), "OOS 19+")
    util.summarize(util.regime_slice(r, "2022-01-01", "2022-12-31"), "2022")
    util.summarize(util.regime_slice(r, "2020-01-01", "2020-12-31"), "COVID")
    util.summarize(util.regime_slice(r, "2008-01-01", "2008-12-31"), "2008 (stress test only)")

    r.to_frame("ml_v5").to_csv("/home/user/bonds/data/apex/ml_v5_returns.csv")
    W.to_csv("/home/user/bonds/data/apex/ml_v5_weights.csv")
