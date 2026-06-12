"""Walk-forward LightGBM cross-sectional ranker for the biweekly DCA picker.

Causality contract (see RESEARCH_PROTOCOL.md):
  * All features are trailing-only transforms of OHLCV through close of d.
  * Cross-sectional rank transforms are within-row (date) only, members only.
  * Labels (forward 126d return, rank-transformed within date) are used ONLY
    for training, and a model fit at trading-day position t may only train on
    feature dates at positions <= t - 127, so every label is fully realized
    strictly before the fit date.  Predictions are made strictly after t.
  * Refit every ~6 months (126 trading days).  First fit requires >= 3 years
    (756 trading days) of labeled history -> first prediction ~ 2008-01.

`build_scores()` runs the whole walk-forward and caches the scores matrix to
research/ml_scores.parquet (+ ml_meta.json with per-refit feature importances
and rank-IC diagnostics) so it never recomputes unless force=True.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import data as data_mod  # noqa: E402

SCORES_PQ = os.path.join(_HERE, "ml_scores.parquet")
META_JSON = os.path.join(_HERE, "ml_meta.json")

HORIZON = 126          # forward label horizon (trading days)
EMBARGO = HORIZON + 1  # fit at t may use feature dates <= t - EMBARGO
TRAIN_EVERY = 10       # sample training dates every 10 trading days
REFIT_EVERY = 126      # ~6 months
MIN_LABELED_SPAN = 756 # >= 3y of labeled training dates before first fit
MAX_TRAIN_ROWS = 300_000
SEED = 7

LGBM_PARAMS = dict(
    objective="regression",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=200,
    colsample_bytree=0.8,
    subsample=1.0,
    n_jobs=1,   # box is oversubscribed; OMP spin-wait makes -1 pathological
    random_state=SEED,
    verbose=-1,
)


# ---------------------------------------------------------------------------
# features (all wide DataFrames, trailing-only)
# ---------------------------------------------------------------------------

def build_features(P: dict) -> dict:
    """Return {name: wide DataFrame} of trailing-only raw features."""
    c, h, l, v = P["close"], P["high"], P["low"], P["volume"]
    ret = c.pct_change(fill_method=None)

    F = {}
    # --- returns / momentum ---
    for lb in (21, 63, 126, 252):
        F[f"ret_{lb}"] = c / c.shift(lb) - 1.0
    F["mom_12_1"] = c.shift(21) / c.shift(252) - 1.0

    # --- 52w high ---
    rmax = c.rolling(252, min_periods=252).max()
    F["dist_high_252"] = c / rmax - 1.0
    is_high = (c >= rmax - 1e-12) & rmax.notna()
    pos = pd.DataFrame(
        np.broadcast_to(np.arange(len(c), dtype=float)[:, None], c.shape).copy(),
        index=c.index, columns=c.columns)
    F["days_since_high_252"] = (pos - pos.where(is_high).ffill()).where(rmax.notna())

    # --- realized vol & ratios ---
    vol = {w: ret.rolling(w, min_periods=w).std() for w in (20, 60, 120)}
    F["vol_20"], F["vol_60"], F["vol_120"] = vol[20], vol[60], vol[120]
    F["vol_ratio_20_60"] = vol[20] / vol[60]
    F["vol_ratio_20_120"] = vol[20] / vol[120]

    # --- path quality ---
    up = (ret > 0).where(ret.notna())
    F["up_share_63"] = up.rolling(63, min_periods=63).mean()
    F["max_ret_21"] = ret.rolling(21, min_periods=21).max()
    F["skew_60"] = ret.rolling(60, min_periods=60).skew()

    # --- volume ---
    F["volm_ratio_20_120"] = (v.rolling(20, min_periods=20).mean()
                              / v.rolling(120, min_periods=120).mean())
    upv = v.where(ret > 0, 0.0).where(ret.notna())
    dnv = v.where(ret < 0, 0.0).where(ret.notna())
    su = upv.rolling(63, min_periods=63).sum()
    sd = dnv.rolling(63, min_periods=63).sum()
    F["updown_volm_63"] = su / (su + sd)

    # --- range contraction ---
    rng = ((h - l) / c).where(c.notna())
    F["range_contr_10_60"] = (rng.rolling(10, min_periods=10).mean()
                              / rng.rolling(60, min_periods=60).mean())

    # --- beta & idiosyncratic momentum vs SPY (rolling cov/var, 252d) ---
    spy = data_mod.load_benchmark("SPY")["Close"].reindex(c.index)
    sret = spy.pct_change(fill_method=None)
    W = 252
    m_s = sret.rolling(W, min_periods=W).mean()
    m_r = ret.rolling(W, min_periods=W).mean()
    m_rs = ret.mul(sret, axis=0).rolling(W, min_periods=W).mean()
    cov = m_rs - m_r.mul(m_s, axis=0)
    var_s = (sret ** 2).rolling(W, min_periods=W).mean() - m_s ** 2
    beta = cov.div(var_s, axis=0)
    F["beta_252"] = beta
    spy_252 = spy / spy.shift(252) - 1.0
    F["idio_mom_252"] = (c / c.shift(252) - 1.0).sub(
        beta.mul(spy_252, axis=0))
    return F


FEATURES = [
    "ret_21", "ret_63", "ret_126", "ret_252", "mom_12_1",
    "dist_high_252", "days_since_high_252",
    "vol_20", "vol_60", "vol_120", "vol_ratio_20_60", "vol_ratio_20_120",
    "up_share_63", "max_ret_21", "skew_60",
    "volm_ratio_20_120", "updown_volm_63", "range_contr_10_60",
    "beta_252", "idio_mom_252",
]


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------

def build_scores(force: bool = False, verbose: bool = True):
    """Run the full walk-forward, cache scores to parquet.  Returns scores."""
    if not force and os.path.exists(SCORES_PQ):
        return pd.read_parquet(SCORES_PQ)

    import lightgbm as lgb

    P = data_mod.build_panel()
    c, member = P["close"], P["member"]
    idx, cols = c.index, c.columns
    n, m = len(idx), len(cols)

    t0 = time.time()
    raw = build_features(P)
    # cross-sectional rank transform within each date, members only
    memb = member.to_numpy(bool)
    cube = np.full((n, m, len(FEATURES)), np.nan, dtype=np.float32)
    for j, f in enumerate(FEATURES):
        ranked = raw[f].where(member).rank(axis=1, pct=True)
        cube[:, :, j] = ranked.to_numpy(np.float32)
        del raw[f]
    if verbose:
        print(f"features+ranks built in {time.time()-t0:.0f}s", flush=True)

    # label: forward 126d return, rank within date among members
    fwd = c.shift(-HORIZON) / c - 1.0
    y_rank = fwd.where(member).rank(axis=1, pct=True).to_numpy(np.float32)

    # a training/prediction row is usable if member & core feature present
    core = ~np.isnan(cube[:, :, FEATURES.index("ret_126")])
    row_ok = memb & core
    label_ok = row_ok & ~np.isnan(y_rank)

    sampled = np.arange(0, n, TRAIN_EVERY)
    labeled = sampled[label_ok[sampled].sum(axis=1) >= 50]
    first_fit = int(labeled[0]) + MIN_LABELED_SPAN + EMBARGO
    fit_positions = list(range(first_fit, n, REFIT_EVERY))
    if verbose:
        print(f"first labeled date {idx[labeled[0]].date()}, "
              f"first fit {idx[fit_positions[0]].date()}, "
              f"{len(fit_positions)} refits", flush=True)

    rng = np.random.default_rng(SEED)
    scores = np.full((n, m), np.nan, dtype=np.float32)
    importances, fit_dates = [], []

    for fi, t in enumerate(fit_positions):
        tr_pos = labeled[labeled <= t - EMBARGO]
        Xs, ys = [], []
        for p in tr_pos:
            ok = label_ok[p]
            Xs.append(cube[p][ok])
            ys.append(y_rank[p][ok])
        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        if len(X) > MAX_TRAIN_ROWS:
            sel = rng.choice(len(X), MAX_TRAIN_ROWS, replace=False)
            X, y = X[sel], y[sel]
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X, y, feature_name=FEATURES)
        gain = model.booster_.feature_importance(importance_type="gain")
        importances.append((gain / gain.sum()).tolist())
        fit_dates.append(str(idx[t].date()))

        # predict all dates in (t, t + REFIT_EVERY] (last fit: through end)
        p_end = min(t + REFIT_EVERY, n - 1) if fi + 1 < len(fit_positions) else n - 1
        p_rng = np.arange(t + 1, p_end + 1)
        if len(p_rng) == 0:
            continue
        block = cube[p_rng].reshape(-1, len(FEATURES))
        ok = row_ok[p_rng].reshape(-1)
        pred = np.full(len(block), np.nan, dtype=np.float32)
        if ok.any():
            pred[ok] = model.predict(block[ok])
        scores[p_rng] = pred.reshape(len(p_rng), m)
        if verbose and (fi % 5 == 0 or fi == len(fit_positions) - 1):
            print(f"fit {fi+1}/{len(fit_positions)} at {idx[t].date()} "
                  f"rows={len(X)} ({time.time()-t0:.0f}s)", flush=True)

    S = pd.DataFrame(scores, index=idx, columns=cols)
    S.to_parquet(SCORES_PQ)

    # diagnostics: rank-IC (Spearman) of score vs realized fwd return by year
    ic = _rank_ic(S, fwd, member)
    meta = {
        "features": FEATURES,
        "lgbm_params": {k: v for k, v in LGBM_PARAMS.items()},
        "fit_dates": fit_dates,
        "importances_gain_share": importances,
        "first_prediction": str(S.dropna(how="all").index[0].date()),
        "ic_by_year": {str(k): float(v) for k, v in
                       ic.groupby(ic.index.year).mean().items()},
        "ic_overall_mean": float(ic.mean()),
        "ic_overall_t": float(ic.mean() / ic.std() * np.sqrt(len(ic))),
    }
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=1)
    if verbose:
        print(f"done in {time.time()-t0:.0f}s; cached {SCORES_PQ}", flush=True)
    return S


def _rank_ic(scores, fwd, member, every=10):
    """Per-date Spearman IC of score vs realized forward return (members)."""
    out = {}
    pos = np.arange(0, len(scores), every)
    sv, fv, mv = scores.to_numpy(), fwd.to_numpy(), member.to_numpy(bool)
    for p in pos:
        ok = mv[p] & ~np.isnan(sv[p]) & ~np.isnan(fv[p])
        if ok.sum() < 50:
            continue
        s = pd.Series(sv[p][ok]).rank()
        f = pd.Series(fv[p][ok]).rank()
        out[scores.index[p]] = s.corr(f)
    return pd.Series(out).dropna()


def builder(panels):  # audit-compatible entry point (uses cached scores)
    return build_scores()


if __name__ == "__main__":
    import protocol

    S = build_scores(force="--force" in sys.argv)
    meta = json.load(open(META_JSON))
    print("first prediction:", meta["first_prediction"])
    print("IC by year:", {k: round(v, 3) for k, v in meta["ic_by_year"].items()})
    for k in (1, 2, 3, 5):
        protocol.evaluate_signal(S, f"ml_lgbm_rank_k{k}", k=k)
