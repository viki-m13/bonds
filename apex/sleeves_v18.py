"""APEX v18 sleeves — truly novel approaches.

  SL_BREAKOUT52     — 52-week high breakout (Jegadeesh-Titman style)
  SL_HMM            — Hidden Markov Model regime detection
  SL_HETEROML       — Ensemble of XGBoost + RandomForest + GradientBoosting
  SL_DIVERGENCE     — Long/short based on SPY-QQQ relative strength divergence
  SL_VOLUME_THRUST  — Volume-confirmed breakout signals
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

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


def _weights_to_ret(W, cp):
    w = W.fillna(0.0)
    rets = cp.pct_change()
    r = (w.shift(1).fillna(0.0) * rets.reindex_like(w).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w.diff().abs().fillna(w.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    return r - drag


def _scale_to_vol(W, cp, target_vol=0.15):
    r = _weights_to_ret(W, cp)
    rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return W.mul(m, axis=0)


# ==========================================================================
# 52-WEEK BREAKOUT — Jegadeesh-Titman / George-Hwang
# ==========================================================================

def sleeve_breakout52(cp: pd.DataFrame, target_vol: float = 0.18) -> pd.DataFrame:
    """Long each LETF when its underlying hits a 52-week high.

    George-Hwang (2004 JoF) shows that stocks near their 52-week high
    systematically outperform. The effect is STRONGER than momentum because
    it incorporates price psychology.

    Universe: Underlyings. When underlying > rolling 252d max × 0.98 AND
    trending up, go long its LETF. Equal-weight breakouts.
    """
    # Underlyings paired with LETFs
    pairs = {"SPY": "UPRO", "QQQ": "TQQQ", "TLT": "TMF", "GLD": "UGL"}
    for u in ["EEM", "SMH", "XLK", "XLF", "XLE", "VNQ", "USO", "FXI"]:
        pass  # we only have SPY/QQQ/TLT/GLD/etc in cp

    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    for under, letf in pairs.items():
        if under not in cp.columns or letf not in cp.columns:
            continue
        p = cp[under]
        # Is price within 2% of 252d max?
        max_252 = p.rolling(252, min_periods=30).max()
        near_high = (p / max_252 > 0.98).astype(float)
        # AND trending up (above 200MA)
        above_200 = (p > p.rolling(200).mean()).astype(float)
        # AND momentum positive
        mom_63 = (p.pct_change(63) > 0).astype(float)
        signal = near_high * above_200 * mom_63
        W[letf] = signal.shift(1).fillna(0.0) * 0.25

    # Cap total weight at 1
    s = W.sum(axis=1).clip(upper=1.0)
    scale = (s / W.sum(axis=1).replace(0, np.nan)).fillna(1.0).clip(upper=1.0)
    W = W.mul(scale, axis=0)

    return _scale_to_vol(W, cp, target_vol=target_vol)


# ==========================================================================
# HMM REGIME DETECTION
# ==========================================================================

def sleeve_hmm(cp: pd.DataFrame, target_vol: float = 0.15, n_states: int = 3) -> pd.DataFrame:
    """Hidden Markov Model on SPY returns identifies hidden regimes.

    Train on IS (2005-2018). Each state has different (mu, sigma). The highest-mu
    state = "bull"; others = "bear/sideways". Allocate to UPRO in bull, cash
    otherwise.
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)

    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    if "UPRO" not in cp.columns:
        return W

    spy_r = cp["SPY"].pct_change().dropna()

    # Fit HMM on IS (2005-2018) only
    is_r = spy_r.loc["2005-01-01":"2018-12-31"].dropna()
    if len(is_r) < 500:
        return W

    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full",
                             n_iter=100, random_state=42)
    X_is = is_r.values.reshape(-1, 1)
    model.fit(X_is)

    # Classify ALL dates
    all_r = spy_r.dropna()
    X_all = all_r.values.reshape(-1, 1)
    states = model.predict(X_all)
    state_series = pd.Series(states, index=all_r.index)

    # Identify "bull state" = state with highest mean
    state_means = [model.means_[i][0] for i in range(n_states)]
    bull_state = int(np.argmax(state_means))
    # And "bear state" = lowest mean
    bear_state = int(np.argmin(state_means))

    # Signal
    in_bull = (state_series == bull_state).astype(float)
    in_bear = (state_series == bear_state).astype(float)

    # Reindex
    in_bull = in_bull.reindex(idx).fillna(0).shift(1).fillna(0.0)
    in_bear = in_bear.reindex(idx).fillna(0).shift(1).fillna(0.0)

    W["UPRO"] = in_bull * 0.50
    if "UGL" in cp.columns:
        W["UGL"] = in_bear * 0.30   # gold in bear
    if "TMF" in cp.columns:
        W["TMF"] = in_bear * 0.20

    return _scale_to_vol(W, cp, target_vol=target_vol)


# ==========================================================================
# HETEROGENEOUS ML ENSEMBLE — XGB + RandomForest + GradientBoosting
# ==========================================================================

def sleeve_hetero_ml(cp: pd.DataFrame, target_vol: float = 0.20,
                      k_top: int = 3, rebal_every: int = 10,
                      train_start: str = "2010-03-11",
                      train_end: str = "2018-12-31") -> pd.DataFrame:
    """Three different ML algorithms, average predictions.

    Different algorithms capture different biases. Averaging reduces
    model-specific overfit.
    """
    try:
        import xgboost as xgb
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
    except ImportError:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)

    from ml_v3 import build_features, UNIVERSE
    long_df, feat_cols = build_features(cp)

    N = 21
    fwd_rows = []
    for tic in UNIVERSE:
        if tic not in cp.columns:
            continue
        p = cp[tic]
        lr = np.log(p / p.shift(1))
        vol_21 = lr.rolling(21).std()
        fwd = np.log(p.shift(-N) / p)
        y = fwd / vol_21.replace(0, np.nan)
        fwd_rows.append(pd.DataFrame({"Date": cp.index, "Ticker": tic, "y": y.values}))
    fwd_df = pd.concat(fwd_rows, ignore_index=True)
    data = long_df.merge(fwd_df, on=["Date", "Ticker"], how="left")

    is_mask = ((data["Date"] >= pd.Timestamp(train_start)) &
               (data["Date"] <= pd.Timestamp(train_end) - pd.Timedelta(days=N + 5)))
    train = data[is_mask].dropna(subset=feat_cols + ["y"])
    print(f"    [hetero ML] train rows: {len(train)}")

    pred_df = data.dropna(subset=feat_cols).copy()
    pred_sum = np.zeros(len(pred_df))

    # Model 1: XGBoost
    m1 = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                           min_child_weight=20, subsample=0.7, reg_lambda=5,
                           n_jobs=4, verbosity=0, random_state=42, tree_method="hist")
    m1.fit(train[feat_cols].values, train["y"].values)
    pred_sum = pred_sum + m1.predict(pred_df[feat_cols].values)

    # Model 2: Random Forest
    m2 = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=30,
                                 n_jobs=4, random_state=43)
    m2.fit(train[feat_cols].values, train["y"].values)
    pred_sum = pred_sum + m2.predict(pred_df[feat_cols].values)

    # Model 3: Gradient Boosting (different algo than XGBoost)
    m3 = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                    min_samples_leaf=50, subsample=0.7, random_state=44)
    m3.fit(train[feat_cols].values, train["y"].values)
    pred_sum = pred_sum + m3.predict(pred_df[feat_cols].values)

    pred_df["pred"] = pred_sum / 3
    wide = pred_df.pivot(index="Date", columns="Ticker", values="pred")
    wide = wide.reindex(cp.index)

    # Weights
    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_rebal = mask % rebal_every == 0
    rnk = wide.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= k_top)
    sel_m = sel.where(is_rebal).ffill().fillna(False)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for tic in UNIVERSE:
        if tic in cp.columns and tic in sel_m.columns:
            W[tic] = sel_m[tic].astype(float) / k_top

    return _scale_to_vol(W, cp, target_vol=target_vol)


# ==========================================================================
# DIVERGENCE — SPY/QQQ relative-strength breakdown
# ==========================================================================

def sleeve_divergence(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """When SPY-QQQ relative strength diverges (one going up, one going down),
    pick the winner. Diversifies from pure momentum by focusing on relative.
    """
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    if "SPY" not in cp.columns or "QQQ" not in cp.columns:
        return W

    spy_m63 = cp["SPY"].pct_change(63)
    qqq_m63 = cp["QQQ"].pct_change(63)
    # Relative strength: QQQ - SPY
    rel = qqq_m63 - spy_m63
    # Z-score
    rel_z = (rel - rel.rolling(252, min_periods=60).mean()) / rel.rolling(252, min_periods=60).std()

    # When rel_z > 1: QQQ dominant, long TQQQ
    # When rel_z < -1: SPY dominant, long UPRO
    qqq_lead = (rel_z > 1.0).astype(float).shift(1).fillna(0)
    spy_lead = (rel_z < -1.0).astype(float).shift(1).fillna(0)

    # Market filter
    spy_ok = (cp["SPY"] > cp["SPY"].rolling(200).mean()).astype(float)

    if "TQQQ" in cp.columns:
        W["TQQQ"] = qqq_lead * spy_ok * 0.7
    if "UPRO" in cp.columns:
        W["UPRO"] = spy_lead * spy_ok * 0.7

    return _scale_to_vol(W, cp, target_vol=target_vol)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/bonds/apex")
    op, cp = util.load_prices()

    sleeves = {
        "BREAKOUT52":   sleeve_breakout52(cp),
        "HMM":          sleeve_hmm(cp),
        "DIVERGENCE":   sleeve_divergence(cp),
    }
    print("Fast sleeves (no ML):")
    print(f"  {'Sleeve':15s}  {'SR':>5}  {'CAGR':>7}  {'Vol':>6}  {'MDD':>7}  {'OOS':>5}  {'2022':>7}  {'2008':>7}")
    for name, W in sleeves.items():
        r = _weights_to_ret(W, cp)
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, "2019-01-02", "2027-12-31"))
        r22 = util.regime_slice(r, "2022-01-01", "2022-12-31")
        m22 = util.metrics(r22) if len(r22) > 20 else {"sharpe": 0}
        r08 = util.regime_slice(r, "2008-01-01", "2008-12-31")
        m08 = util.metrics(r08) if len(r08) > 20 else {"sharpe": 0}
        print(f"  {name:15s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['vol']*100:>5.1f}%  {m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>5.2f}  "
              f"{m22.get('sharpe',0):>7.2f}  {m08.get('sharpe',0):>7.2f}")

    # Hetero ML (slower)
    print("\nTraining heterogeneous ML...")
    W_hetero = sleeve_hetero_ml(cp)
    r = _weights_to_ret(W_hetero, cp)
    m = util.metrics(r)
    om = util.metrics(util.regime_slice(r, "2019-01-02", "2027-12-31"))
    r22 = util.regime_slice(r, "2022-01-01", "2022-12-31")
    m22 = util.metrics(r22) if len(r22) > 20 else {"sharpe": 0}
    print(f"  HETERO_ML         {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
          f"{m['vol']*100:>5.1f}%  {m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>5.2f}  {m22.get('sharpe',0):>7.2f}")

    r.to_frame("hetero_ml").to_csv("/home/user/bonds/data/apex/hetero_ml_returns.csv")
