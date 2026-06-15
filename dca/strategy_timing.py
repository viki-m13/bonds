"""TIMING — a regime-timing risk-management overlay on the 12-1 momentum book.

Every cross-sectional model we tried subtracts from momentum. This does the
opposite: it KEEPS the working momentum book and only asks the time-series
question — *is now a moment to hold momentum, or to stand down into
Treasuries?* The edge it targets is risk-adjusted (Sortino / drawdown), not
beating cap-weighted QQQ on raw return, which the validation playbook shows is
near-impossible by selection.

Mechanics (one decision per biweekly signal date):
  * risk-ON  -> contribution (and any proceeds) buys the top-k 12-1 momentum
    names at the next open; existing stock lots are held.
  * risk-OFF -> the whole stock sleeve is liquidated into the bond ETF (IEF) at
    the next open, and new contributions buy bonds, until risk turns back on.
All trades pay `cost_bps`. Strictly next-open execution.

The risk-ON probability is a LightGBM classifier on TRAILING regime features
(SPY trend, breadth, VIX, term spreads, SPY vol/drawdown), refit walk-forward;
its binary label (train only) is "did the momentum book out-return IEF over the
next 63d", fully realised before each fit date.

Validation (run by `validate_timing.py`): OOS split, cutoff-date trajectory,
max drawdown, Sortino, and rolling-window beat-rate vs both QQQ-DCA and the
always-in momentum book.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data as data_mod                       # noqa: E402
from engine import schedule_dates             # noqa: E402

HORIZON = 63
REFIT_EVERY = 126
FIRST_FIT = "2009-01-01"
TRAIN_STRIDE = 5
SEED = 7
BOND = "IEF"          # defensive leg (7-10y Treasuries)

REG_FEATS = ["spy_gap200", "spy_gap50", "breadth", "vix", "vix_chg",
             "term2y", "term3m", "spy_vol21", "spy_dd", "spy_ret126"]


def _bench_close(idx, t):
    return data_mod.load_benchmark(t)["Close"].reindex(idx).ffill()


def _fred(idx, name):
    p = os.path.join(data_mod.ROOT, "data", "fred", f"{name}.csv")
    return pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0].reindex(idx).ffill()


def momentum_rank(P):
    c = P["close"]
    elig = (P["member"] & (c.notna().rolling(252).count() >= 252) & c.notna())
    mom = c.shift(21) / c.shift(252) - 1
    rank = mom.where(elig).rank(axis=1, ascending=False)
    return rank, elig


def regime_features(P):
    c = P["close"]
    idx = c.index
    spy = _bench_close(idx, "SPY")
    spy_ret = np.log(spy).diff()
    above200 = c > c.rolling(200).mean()
    elig = (P["member"] & c.notna())
    breadth = (above200 & elig).sum(axis=1) / elig.sum(axis=1).replace(0, np.nan)
    F = pd.DataFrame(index=idx)
    F["spy_gap200"] = spy / spy.rolling(200).mean() - 1
    F["spy_gap50"] = spy / spy.rolling(50).mean() - 1
    F["breadth"] = breadth
    F["vix"] = _fred(idx, "VIXCLS")
    F["vix_chg"] = F["vix"] - F["vix"].shift(21)
    F["term2y"] = _fred(idx, "T10Y2Y")
    F["term3m"] = _fred(idx, "T10Y3M")
    F["spy_vol21"] = spy_ret.rolling(21).std() * np.sqrt(252)
    F["spy_dd"] = spy / spy.cummax() - 1
    F["spy_ret126"] = spy / spy.shift(126) - 1
    return F


def build_riskon(P, k=3, verbose=True):
    """Walk-forward risk-ON probability per date (ffilled between signal days)."""
    import lightgbm as lgb

    c = P["close"]
    idx = c.index
    T = len(idx)
    rank, _ = momentum_rank(P)
    cv = c.to_numpy(float)
    rk = rank.to_numpy(float)
    F = regime_features(P)
    X_all = F[REG_FEATS].to_numpy(np.float32)

    bond = _bench_close(idx, BOND).to_numpy(float)
    # forward 63d return of the equal-weight top-k momentum book vs IEF
    book_fwd = np.full(T, np.nan)
    bond_fwd = np.full(T, np.nan)
    for i in range(T - HORIZON):
        picks = np.where(rk[i] <= k)[0]
        if len(picks) == 0:
            continue
        fr = cv[i + HORIZON, picks] / cv[i, picks] - 1
        fr = fr[np.isfinite(fr)]
        if len(fr):
            book_fwd[i] = fr.mean()
        bond_fwd[i] = bond[i + HORIZON] / bond[i] - 1
    label = (book_fwd > bond_fwd).astype(np.float32)

    first_fit = idx.searchsorted(pd.Timestamp(FIRST_FIT))
    grid = np.arange(252, T - HORIZON, TRAIN_STRIDE)
    prob = np.full(T, np.nan, np.float32)
    fit_rows = list(range(first_fit, T, REFIT_EVERY))
    imp = np.zeros(len(REG_FEATS))
    for fi, fit in enumerate(fit_rows):
        tr = grid[(grid + HORIZON <= fit) & np.isfinite(label[grid])
                  & np.isfinite(X_all[grid]).all(1)]
        X, y = X_all[tr], label[tr]
        ds = lgb.Dataset(X, label=y, feature_name=REG_FEATS)
        params = dict(objective="binary", learning_rate=0.03, num_leaves=15,
                      min_data_in_leaf=80, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=SEED)
        model = lgb.train(params, ds, num_boost_round=200)
        imp += model.feature_importance(importance_type="gain")
        hi = fit_rows[fi + 1] if fi + 1 < len(fit_rows) else T
        rows = np.arange(fit, hi)
        ok = np.isfinite(X_all[rows]).all(1)
        prob[rows[ok]] = model.predict(X_all[rows[ok]])
    if verbose:
        order = np.argsort(imp)[::-1]
        print("  regime gain importance:",
              ", ".join(f"{REG_FEATS[i]}={imp[i]/imp.sum():.2f}"
                        for i in order[:6]))
    return pd.Series(prob, index=idx).ffill()


def backtest(P, riskon, k=3, thresh=0.5, every=10, start=None, end=None,
             contribution=1000.0, cost_bps=5.0, mode="timing"):
    """mode: 'timing' (rotate), 'always' (momentum, never rotate),
    'bond' (always IEF), 'qqq' (always QQQ). Returns dict with value/invested
    daily Series and the realised daily investment-return series."""
    c, o = P["close"], P["open"]
    idx = c.index
    rank, _ = momentum_rank(P)
    rk = rank.to_numpy(float)
    cols = list(c.columns)
    cpos = {t: j for j, t in enumerate(cols)}
    cv = c.to_numpy(float)
    ov = o.to_numpy(float)
    bond_o = _bench_close(idx, BOND).reindex(idx).to_numpy(float)
    bond_c = bond_o
    bo = data_mod.load_benchmark(BOND); bo_o = bo["Open"].reindex(idx).ffill().to_numpy(float)
    bo_c = bo["Close"].reindex(idx).ffill().to_numpy(float)
    qqq = data_mod.load_benchmark("QQQ")
    q_o = qqq["Open"].reindex(idx).ffill().to_numpy(float)
    q_c = qqq["Close"].reindex(idx).ffill().to_numpy(float)
    ron = riskon.reindex(idx).to_numpy(float)

    sig = schedule_dates(idx, every, 0, start, end)
    sig = sig[idx.searchsorted(sig) + 1 < len(idx)]
    pos = idx.searchsorted(sig[0])
    end_pos = (idx.searchsorted(pd.Timestamp(end), side="right")
               if end is not None else len(idx))
    sig_set = set(sig)
    cost = cost_bps / 1e4

    stock_sh = {}          # col -> shares
    bond_sh = 0.0
    qqq_sh = 0.0
    cash = 0.0
    last_close = {}
    total_in = 0.0
    pending = None         # dict(date, regime_on, picks)
    val_rows, inv_rows = [], []

    for i in range(pos, end_pos):
        d = idx[i]
        # --- execute pending at today's open ---
        if pending is not None and pending["date"] == d:
            if mode == "qqq":
                qqq_sh += cash * (1 - cost) / q_o[i]; cash = 0.0
            elif mode == "bond":
                bond_sh += cash * (1 - cost) / bo_o[i]; cash = 0.0
            else:
                on = pending["regime_on"]
                if on:
                    if bond_sh > 0:                      # rotate bonds -> cash
                        cash += bond_sh * bo_o[i] * (1 - cost); bond_sh = 0.0
                    picks = [p for p in pending["picks"] if np.isfinite(ov[i, p])]
                    if picks and cash > 0:
                        per = cash / len(picks)
                        for p in picks:
                            stock_sh[p] = stock_sh.get(p, 0.0) + per * (1 - cost) / ov[i, p]
                        cash = 0.0
                else:
                    for p, sh in list(stock_sh.items()):   # rotate stocks -> cash
                        px = ov[i, p] if np.isfinite(ov[i, p]) else last_close.get(p, np.nan)
                        if np.isfinite(px):
                            cash += sh * px * (1 - cost)
                    stock_sh = {}
                    if cash > 0:
                        bond_sh += cash * (1 - cost) / bo_o[i]; cash = 0.0
            pending = None

        # --- mark to market (+ delisting cleanup) ---
        v = cash + bond_sh * bo_c[i] + qqq_sh * q_c[i]
        for p, sh in list(stock_sh.items()):
            px = cv[i, p]
            if np.isfinite(px):
                last_close[p] = px; v += sh * px
            else:
                fut = cv[i + 1:i + 6, p] if i + 1 < len(idx) else np.array([np.nan])
                if np.all(~np.isfinite(fut)):
                    cash += sh * last_close.get(p, 0.0) * (1 - cost)
                    v += sh * last_close.get(p, 0.0) * (1 - cost)
                    del stock_sh[p]
                else:
                    v += sh * last_close.get(p, 0.0)

        # --- signal date: decide & queue next-open order ---
        if d in sig_set:
            total_in += contribution; cash += contribution
            v += contribution        # value now reflects the new cash (aligns flow)
            r = ron[i]
            regime_on = True if (mode == "always" or not np.isfinite(r)) else (r > thresh)
            if mode in ("bond", "qqq"):
                regime_on = False
            picks = list(np.where(rk[i] <= k)[0])
            pending = {"date": idx[i + 1], "regime_on": regime_on, "picks": picks}

        val_rows.append(v); inv_rows.append(total_in)

    span = idx[pos:end_pos]
    value = pd.Series(val_rows, index=span)
    invested = pd.Series(inv_rows, index=span)
    flow = invested.diff().fillna(invested.iloc[0])
    ret = ((value - value.shift(1) - flow) / value.shift(1).replace(0, np.nan))
    return {"value": value, "invested": invested, "ret": ret.fillna(0.0)}


if __name__ == "__main__":
    print("building risk-ON timing signal (walk-forward)...", flush=True)
    P = data_mod.build_panel()
    ron = build_riskon(P, k=3)
    ron.to_frame("riskon").to_parquet(os.path.join(HERE, "research",
                                                    "timing_riskon.parquet"))
    print("risk-ON share of signal days (2010+):",
          f"{(ron.loc['2010':] > 0.5).mean():.0%}")
