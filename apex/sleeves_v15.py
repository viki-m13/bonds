"""APEX v15 — New sleeves: inverse-LETF (synthetic short), multi-crypto, walk-forward ML.

The inverse-LETF sleeve gives us "short equity/bond" exposure without margin/shorting:
  Long SH when SPY trending DOWN (SPY<200MA AND 63d-return<0).
  Long PSQ when QQQ trending down.
  Long SDS when SPY crashing (20d<-5% AND VIX>25).
  Long TBF when TLT trending down (2022-killer for the bond sleeve).

Why this works: in 2022, SH gained +20%, PSQ gained +35%, TBF gained +45%.
These ADD alpha when trend-followers hit bear markets.
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
# INVERSE-LETF SLEEVE — synthetic short via inverse ETFs (long-only, no margin)
# ==========================================================================

def sleeve_inverse(cp: pd.DataFrame) -> pd.DataFrame:
    """Long inverse LETFs when their underlyings are trending DOWN.

    Gives us short-equity and short-bond exposure within the no-margin
    constraint (we are LONG inverse LETFs; the inverse LETF itself is
    bearish-positioned internally).

    Triggers:
    - Long SH (1x inverse SPY) when: SPY < 200dMA AND SPY 63d ret < 0
    - Long PSQ (1x inverse QQQ) when: QQQ < 200dMA AND QQQ 63d ret < 0
    - Long SDS (2x inverse SPY) when: SPY 20d < -5% AND VIX > 25 (crash mode)
    - Long TBF (1x inverse TLT) when: TLT < 200dMA AND TLT 63d ret < 0
       (THE 2022 bond-rout defense)

    Each trigger gets 0.25 allocation when on. Max gross ~1.0.
    """
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    # Load inverse LETFs into cp if not present
    for t in ["SH", "PSQ", "SDS", "TBF", "SQQQ", "SPXU"]:
        if t not in cp.columns:
            s = _etf_close(t, idx)
            if not s.isna().all():
                cp[t] = s

    spy = cp["SPY"]
    qqq = cp.get("QQQ", spy)
    tlt = cp.get("TLT", spy)
    vix = _fred("VIXCLS", idx)

    # Signal 1: SH when SPY in downtrend
    sh_on = ((spy < spy.rolling(200).mean()) & (spy.pct_change(63) < 0)).astype(float)
    if "SH" in cp.columns:
        W["SH"] = sh_on * 0.25

    # Signal 2: PSQ when QQQ in downtrend
    psq_on = ((qqq < qqq.rolling(200).mean()) & (qqq.pct_change(63) < 0)).astype(float)
    if "PSQ" in cp.columns:
        W["PSQ"] = psq_on * 0.25

    # Signal 3: SDS when SPY in crash mode
    sds_on = ((spy.pct_change(20) < -0.05) & (vix > 25)).astype(float).fillna(0)
    if "SDS" in cp.columns:
        W["SDS"] = sds_on * 0.25

    # Signal 4: TBF when TLT in downtrend (2022-killer)
    tbf_on = ((tlt < tlt.rolling(200).mean()) & (tlt.pct_change(63) < 0)).astype(float)
    if "TBF" in cp.columns:
        W["TBF"] = tbf_on * 0.25

    return _scale_to_vol(W, cp, target_vol=0.15)


# ==========================================================================
# MULTI-CRYPTO SLEEVE — BTC + ETH + SOL
# ==========================================================================

def multi_crypto_returns(idx: pd.DatetimeIndex, target_vol: float = 0.25) -> pd.Series:
    """BTC + ETH + SOL equal-weighted when each is trending, with safeguards.

    Each coin gets 1/3 allocation when its own trend is up (63d return > 0
    AND spot > 200d MA). Gated by SPY > 200MA AND VIX < 30.
    """
    btc = _etf_close("BTC_USD", idx)
    eth = _etf_close("ETH_USD", idx)
    sol = _etf_close("SOL_USD", idx)
    spy = _etf_close("SPY", idx)
    vix = _fred("VIXCLS", idx)

    # Market gates
    spy_ok = (spy > spy.rolling(200).mean()).astype(float)
    vix_ok = (vix < 30).astype(float).fillna(1.0)
    gate = spy_ok * vix_ok

    def signal(p):
        if p.isna().all():
            return pd.Series(0.0, index=idx)
        on = ((p > p.rolling(200).mean()) & (p.pct_change(63) > 0)).astype(float)
        return on.shift(1).fillna(0.0)

    btc_on = signal(btc)
    eth_on = signal(eth)
    sol_on = signal(sol)

    btc_r = btc.pct_change().fillna(0)
    eth_r = eth.pct_change().fillna(0)
    sol_r = sol.pct_change().fillna(0)

    # Each coin gets 1/3 of sleeve. When ON, invest; when OFF, 0.
    r_btc = btc_on * btc_r * gate / 3.0
    r_eth = eth_on * eth_r * gate / 3.0
    r_sol = sol_on * sol_r * gate / 3.0

    r = r_btc + r_eth + r_sol
    # TC — 20 bps per trade per coin
    tc = (btc_on.diff().abs().fillna(btc_on.abs())
          + eth_on.diff().abs().fillna(eth_on.abs())
          + sol_on.diff().abs().fillna(sol_on.abs())) * 0.002 / 3
    r = r - tc

    # Scale to target vol
    rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return r * m


# ==========================================================================
# WALK-FORWARD ML — re-train yearly on expanding window
# ==========================================================================

def walkforward_ml_weights(cp: pd.DataFrame, top_k: int = 3,
                            rebal_every: int = 21) -> pd.DataFrame:
    """Walk-forward XGBoost: retrain every year on expanding window [2005..prev_year].
    Predict next-21d forward log return. Rank top-K LETFs.
    """
    try:
        import xgboost as xgb
    except ImportError:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)

    # Reuse feature-build from ml_v2
    import ml_v2 as MLV

    UNIVERSE = MLV.UNIVERSE
    idx = cp.index

    print("  Building features for walk-forward ML...")
    long_df, feat_cols = MLV.build_features(cp), None
    if isinstance(long_df, tuple):
        long_df, feat_cols = long_df  # may not be tuple in my ml_v2
    else:
        # build_features only returns long frame; need to compute feat_cols
        feat_cols = [c for c in long_df.columns if c not in ("Date", "Ticker")]

    N = 21
    # Build target
    fwd_rows = []
    for tic in UNIVERSE:
        if tic not in cp.columns:
            continue
        p = cp[tic]
        fwd = np.log(p.shift(-N) / p)
        fwd_rows.append(pd.DataFrame({"Date": cp.index, "Ticker": tic, "y": fwd.values}))
    fwd = pd.concat(fwd_rows, ignore_index=True)
    data = long_df.merge(fwd, on=["Date", "Ticker"], how="left")

    all_preds = pd.DataFrame(np.nan, index=idx, columns=UNIVERSE)

    train_start = "2007-01-01"
    # Yearly retrains
    years = sorted(set(data["Date"].dt.year.unique()))
    years = [y for y in years if y >= 2010]
    for y in years:
        tr_end = pd.Timestamp(f"{y-1}-12-31") - pd.Timedelta(days=N + 5)
        tr = data[(data["Date"] >= pd.Timestamp(train_start)) &
                  (data["Date"] <= tr_end)].dropna(subset=feat_cols + ["y"])
        if len(tr) < 3000:
            continue
        m = xgb.XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.02,
            min_child_weight=40, subsample=0.7, colsample_bytree=0.6,
            reg_lambda=10.0, reg_alpha=1.0,
            n_jobs=4, verbosity=0, random_state=42, tree_method="hist",
        )
        m.fit(tr[feat_cols].values, tr["y"].values)

        # Predict for year y
        year_mask = (data["Date"] >= pd.Timestamp(f"{y}-01-01")) & (data["Date"] <= pd.Timestamp(f"{y}-12-31"))
        pred_df = data[year_mask].dropna(subset=feat_cols).copy()
        if len(pred_df) == 0:
            continue
        pred_df["pred"] = m.predict(pred_df[feat_cols].values)
        for _, row in pred_df.iterrows():
            all_preds.loc[row["Date"], row["Ticker"]] = row["pred"]

    # Build weights
    mask = pd.Series(range(len(idx)), index=idx)
    is_rebal = mask % rebal_every == 0
    rnk = all_preds.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= top_k)
    sel_m = sel.where(is_rebal).ffill().fillna(False)

    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    for tic in UNIVERSE:
        if tic in cp.columns and tic in sel_m.columns:
            W[tic] = sel_m[tic].astype(float) / top_k
    return W


def sleeve_wf_ml(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    W = walkforward_ml_weights(cp)
    return _scale_to_vol(W, cp, target_vol=target_vol)
