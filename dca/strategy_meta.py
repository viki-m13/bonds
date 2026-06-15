"""META — meta-labeling layer on top of 12-1 momentum.

The four return-prediction models (CNN, PatchTST, LightGBM-ranker, Chronos) all
hit IC~0: a model cannot say *which* large-cap will outperform. This asks the
*easier, different* question that meta-labeling (Lopez de Prado, AFML ch.3) is
designed for:

    given that 12-1 momentum already nominated a name, will THIS bet beat QQQ
    over the next quarter -- and is the regime right to make it at all?

So the base signal does the selecting; the model only decides *which of the
momentum leaders to keep* (be selective) and *when to stand down* (abstain ->
hold the cash leg). Its edge, if any, comes from REGIME context the
cross-sectional models never saw: SPY trend, breadth, VIX, term spread,
cross-sectional dispersion.

Causality: every feature at date d is trailing-only (rolling / shift); the
binary meta-label (beat QQQ over the next 63d) is used in TRAINING only and is
fully realised before each walk-forward fit date. LightGBM refit every 126
trading days on labels closed <= fit date; predicts strictly after.

Run:  python strategy_meta.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data as data_mod          # noqa: E402

HORIZON = 63          # meta-label / forward evaluation horizon (~1 quarter)
POOL_M = 15           # candidate pool: top-M eligible names by 12-1 momentum
TRAIN_STRIDE = 10     # sample training rows every 10 td
SCORE_STRIDE = 5
REFIT_EVERY = 126     # ~6-month walk-forward refit
FIRST_FIT = "2009-01-01"
SEED = 7

FEATS_XS = ["mom121", "mom61", "mom31", "vol63", "vol126", "beta252",
            "dist_high", "dd", "volvol", "voltrend"]   # cross-sectional (ranked)
FEATS_REG = ["spy_gap", "breadth", "vix", "vix_chg", "term", "disp"]  # regime
FEATS = FEATS_XS + FEATS_REG


def _bench_close(idx, ticker):
    b = data_mod.load_benchmark(ticker)["Close"]
    return b.reindex(idx).ffill()


def _fred(idx, name):
    p = os.path.join(data_mod.ROOT, "data", "fred", f"{name}.csv")
    s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
    return s.reindex(idx).ffill()


def _rank(df, elig):
    return df.where(elig).rank(axis=1, pct=True)


def build_features(P):
    """Return (feat dict of (T,N) arrays, eligible mask, candidate mask,
    forward-63d stock return, forward-63d QQQ return)."""
    close, vol = P["close"], P["volume"]
    idx = close.index
    elig_df = (P["member"] & (close.notna().rolling(252).count() >= 252)
               & close.notna())
    elig = elig_df.to_numpy(bool)

    logret = np.log(close).diff()
    spy = _bench_close(idx, "SPY")
    spy_ret = np.log(spy).diff()

    mom121 = close.shift(21) / close.shift(252) - 1
    mom61 = close.shift(21) / close.shift(126) - 1
    mom31 = close.shift(21) / close.shift(63) - 1
    vol63 = logret.rolling(63).std()
    vol126 = logret.rolling(126).std()
    cov = logret.rolling(252).cov(spy_ret)
    var = spy_ret.rolling(252).var()
    beta = cov.div(var, axis=0)
    dist_high = close / close.rolling(252).max() - 1
    dd = close / close.cummax() - 1
    volvol = vol63.rolling(63).std()
    dollar = (close * vol)
    voltrend = dollar.rolling(20).mean() / dollar.rolling(120).mean()

    xs = {"mom121": mom121, "mom61": mom61, "mom31": mom31, "vol63": vol63,
          "vol126": vol126, "beta252": beta, "dist_high": dist_high, "dd": dd,
          "volvol": volvol, "voltrend": voltrend}
    feat = {k: _rank(v, elig_df).to_numpy(np.float32) for k, v in xs.items()}

    # --- regime features (one value per date, broadcast across names) ---
    spy_gap = (spy / spy.rolling(200).mean() - 1)
    above200 = close > close.rolling(200).mean()
    breadth = (above200 & elig_df).sum(axis=1) / elig_df.sum(axis=1).replace(0, np.nan)
    vix = _fred(idx, "VIXCLS")
    vix_chg = vix - vix.shift(21)
    term = _fred(idx, "T10Y2Y")
    disp = (close / close.shift(63) - 1).where(elig_df).std(axis=1)
    reg = {"spy_gap": spy_gap, "breadth": breadth, "vix": vix,
           "vix_chg": vix_chg, "term": term, "disp": disp}
    N = close.shape[1]
    for k, s in reg.items():
        feat[k] = np.repeat(s.to_numpy(np.float32)[:, None], N, axis=1)

    cv = close.to_numpy(float)
    T = len(idx)
    fwd = np.full((T, N), np.nan)
    fwd[:T - HORIZON] = cv[HORIZON:] / cv[:T - HORIZON] - 1
    qqq = _bench_close(idx, "QQQ").to_numpy(float)
    fwd_q = np.full(T, np.nan)
    fwd_q[:T - HORIZON] = qqq[HORIZON:] / qqq[:T - HORIZON] - 1

    # candidate pool: top-M eligible by 12-1 momentum each date
    mrank = mom121.where(elig_df).rank(axis=1, ascending=False)
    cand = (mrank <= POOL_M).to_numpy(bool) & elig
    return feat, elig, cand, fwd, fwd_q


def _matrix(feat, rows, cols):
    return np.column_stack([feat[f][rows, cols] for f in FEATS])


def build_scores(P=None, verbose=True):
    import lightgbm as lgb

    if P is None:
        P = data_mod.build_panel()
    close = P["close"]
    idx, colz = close.index, close.columns
    T, N = close.shape
    feat, elig, cand, fwd, fwd_q = build_features(P)
    label = (fwd > fwd_q[:, None]).astype(np.float32)   # beat QQQ over next 63d

    first_fit = idx.searchsorted(pd.Timestamp(FIRST_FIT))
    train_grid = np.arange(252, T - HORIZON, TRAIN_STRIDE)
    score_grid = np.arange(max(252, first_fit), T, SCORE_STRIDE)
    scores = np.full((T, N), np.nan, np.float32)
    fit_rows = list(range(first_fit, T, REFIT_EVERY))
    imp = np.zeros(len(FEATS))

    for fi, fit in enumerate(fit_rows):
        tr = train_grid[train_grid + HORIZON <= fit]
        Rr, Rc = [], []
        for r in tr:
            c = np.nonzero(cand[r] & np.isfinite(fwd[r]))[0]
            if len(c):
                Rr.append(np.full(len(c), r)); Rc.append(c)
        rows = np.concatenate(Rr); cols = np.concatenate(Rc)
        X = _matrix(feat, rows, cols); y = label[rows, cols]
        ds = lgb.Dataset(X, label=y, feature_name=FEATS)
        params = dict(objective="binary", learning_rate=0.03, num_leaves=31,
                      min_data_in_leaf=200, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                      seed=SEED)
        model = lgb.train(params, ds, num_boost_round=300)
        imp += model.feature_importance(importance_type="gain")

        hi = fit_rows[fi + 1] if fi + 1 < len(fit_rows) else T
        sgrid = score_grid[(score_grid >= fit) & (score_grid < hi)]
        for r in sgrid:
            c = np.nonzero(cand[r])[0]
            if len(c):
                scores[r, c] = model.predict(_matrix(feat, np.full(len(c), r), c))
        if verbose:
            print(f"  refit {idx[fit].date()} train={len(y):>6} "
                  f"pos={y.mean():.2f}", flush=True)

    if verbose:
        order = np.argsort(imp)[::-1]
        print("  gain importance:",
              ", ".join(f"{FEATS[i]}={imp[i]/imp.sum():.2f}" for i in order[:8]))
    df = pd.DataFrame(scores, index=idx, columns=colz)
    return df.ffill(limit=SCORE_STRIDE * 2)


if __name__ == "__main__":
    import protocol

    print("building META (meta-labeled momentum) scores...", flush=True)
    P = data_mod.build_panel()
    S = build_scores(P)
    S.to_parquet(os.path.join(HERE, "research", "meta_scores.parquet"))
    print("scored days:", int((S.notna().sum(axis=1) > 0).sum()), "/", len(S))

    c = P["close"]
    base = c.shift(21) / c.shift(252) - 1.0      # raw 12-1 momentum
    print("\n--- META (meta-labeled momentum) ---")
    for k in (2, 3):
        protocol.evaluate_signal(S, f"meta_mom_k{k}", k=k)
    print("--- raw 12-1 momentum (full universe, for reference) ---")
    for k in (2, 3):
        protocol.evaluate_signal(base, f"raw_mom121_k{k}", k=k)
