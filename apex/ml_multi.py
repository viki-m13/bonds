"""APEX — Multi-horizon ML: train separate models for N=5, 21, 42, 63-day
forward returns. Each is a different sleeve. Lower correlation than one model.
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
    spy = cp["SPY"]
    vix = _fred("VIXCLS", cp.index)
    curve = _fred("T10Y2Y", cp.index)
    hyg = _etf_close("HYG", cp.index)
    lqd = _etf_close("LQD", cp.index)
    cred_ratio = (hyg / lqd)

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
        df = df.shift(1)
        for c in mac.columns:
            df[c] = mac[c].shift(1)
        df["Ticker"] = tic
        df["Date"] = cp.index
        rows.append(df.reset_index(drop=True))

    long = pd.concat(rows, ignore_index=True)
    rank_cols = ["mom_5", "mom_21", "mom_63", "mom_126", "mom_252",
                 "vol_21", "vol_63", "sr_21", "sr_63",
                 "rel_spy_21", "rel_spy_63", "d_200dma"]
    for c in rank_cols:
        long[f"rk_{c}"] = long.groupby("Date")[c].rank(pct=True)
    feat_cols = [c for c in long.columns if c not in ("Date", "Ticker")]
    return long, feat_cols


def train_model(N: int, feats: pd.DataFrame, feat_cols: list, cp: pd.DataFrame,
                depth=3, n_est=200, lam=10.0, mcw=50):
    """Train XGBoost on IS to predict N-day forward log return. Return wide preds."""
    # Target
    fwd_rows = []
    for tic in UNIVERSE:
        if tic not in cp.columns:
            continue
        p = cp[tic]
        fwd = np.log(p.shift(-N) / p)
        fwd_rows.append(pd.DataFrame({"Date": cp.index, "Ticker": tic, "y": fwd.values}))
    fwd = pd.concat(fwd_rows, ignore_index=True)

    data = feats.merge(fwd, on=["Date", "Ticker"], how="left")
    is_mask = ((data["Date"] >= pd.Timestamp(IS_START)) &
               (data["Date"] <= pd.Timestamp(IS_END) - pd.Timedelta(days=N + 5)))
    tr = data[is_mask].dropna(subset=feat_cols + ["y"])

    m = xgb.XGBRegressor(
        n_estimators=n_est, max_depth=depth, learning_rate=0.03,
        min_child_weight=mcw, subsample=0.7, colsample_bytree=0.6,
        reg_lambda=lam, reg_alpha=1.0, gamma=0.1,
        n_jobs=4, verbosity=0, tree_method="hist", random_state=42,
    )
    m.fit(tr[feat_cols].values, tr["y"].values)

    valid = data.dropna(subset=feat_cols).copy()
    valid["pred"] = m.predict(valid[feat_cols].values)
    wide = valid.pivot(index="Date", columns="Ticker", values="pred")
    return wide.reindex(cp.index)


def make_weights(preds: pd.DataFrame, cp: pd.DataFrame, K: int, rebal: int) -> pd.DataFrame:
    idx = cp.index
    mask = pd.Series(range(len(idx)), index=idx)
    is_rebal = mask % rebal == 0
    rnk = preds.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= K)
    sel_m = sel.where(is_rebal).ffill().fillna(False)
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    for tic in UNIVERSE:
        if tic not in cp.columns or tic not in sel_m.columns:
            continue
        W[tic] = sel_m[tic].astype(float) / K
    return W


def compute_sleeve_ret(W, cp):
    rets = cp.pct_change()
    r = (W.shift(1).fillna(0.0) * rets.reindex_like(W).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = W.diff().abs().fillna(W.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in W.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    return r - drag


def main():
    op, cp = util.load_prices()
    feats, feat_cols = build_features(cp)
    print(f"Features built. Rows: {len(feats)}, cols: {len(feat_cols)}")

    # Build multiple ML sleeves at different horizons
    configs = [
        ("ML5",  5,  3, 5),    # N=5 horizon, K=3, rebal 5d
        ("ML21", 21, 3, 21),
        ("ML63", 63, 3, 21),
    ]

    results = {}
    for name, N, K, rebal in configs:
        print(f"\nTraining {name} (N={N}, K={K}, rebal={rebal})...")
        preds = train_model(N, feats, feat_cols, cp)
        W = make_weights(preds, cp, K, rebal)
        r = compute_sleeve_ret(W, cp)
        results[name] = r
        W.to_csv(f"/home/user/bonds/data/apex/{name.lower()}_weights.csv")
        r.to_frame(name).to_csv(f"/home/user/bonds/data/apex/{name.lower()}_returns.csv")

    R = pd.DataFrame(results)
    print("\nCorrelations:")
    print(R.corr().round(2))

    print("\nSleeve metrics (FULL):")
    for n in R.columns:
        util.summarize(R[n], f"  {n}")
    print("\nSleeve metrics (OOS 19+):")
    for n in R.columns:
        util.summarize(util.regime_slice(R[n], "2019-01-02", "2027-12-31"), f"  {n}")

    # Equal-weight blend
    blend = R.mean(axis=1)
    print("\n=== ML BLEND (EW) ===")
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", "2018-12-31")),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31")),
                        ("GFC", ("2007-01-01", "2009-12-31"))]:
        util.summarize(util.regime_slice(blend, s, e), f"  {lbl}")


if __name__ == "__main__":
    main()
