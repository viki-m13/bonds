"""QUANTUM — ML signal ensemble for leveraged-ETF rotation (walk-forward).

Angle
-----
Gradient-boosted regressor (xgboost) predicting each LETF's next-N-day log
return. Features per (date, ticker):

  Per-ticker (using close[t-1]):
    * Momentum lags       : 5, 21, 63, 252 day log returns
    * Realized vol        : 21d, 63d std of daily log ret
    * Return/vol (Sharpe) : 21d, 63d mean/std
    * Relative vs SPY     : 21d, 63d excess return
    * Distance 200dma     : (close-ma)/ma
    * Cross-sectional ranks on each of the above

  Macro (same for all tickers per day):
    * VIX level, VIX 21d change
    * HY OAS (BAMLH0A0HYM2) level, 21d slope
    * T10Y2Y level
    * SPY 21d vs 63d MA spread

Targets: next N-day log return (forward, close-to-close, ranking objective).

Walk-forward protocol (no in-sample history is ever published):
  * Models are refit once per calendar year on an EXPANDING window of all
    data available before that year, minus an N-trading-day embargo before
    the training cutoff (so no training target overlaps the prediction era).
  * Predictions for year Y come exclusively from the model trained on data
    through Dec-31 of Y-1. The first model trains on 2010-03-11..2013-12-31,
    so the published return series starts 2014-01 (WF_START). There is no
    "IS backtest" of the final model on its own training data — the pre-2014
    era is simply not published.
  * Hyperparameter N (rebalance horizon) is selected by expanding-window CV
    INSIDE THE FIRST TRAINING WINDOW ONLY (2010-2013) and frozen for all
    later refits. K (names held) is a declared design constant = 3, not
    CV-selected (an earlier version claimed K was CV-selected; the CV never
    actually used K).
  * Heavy regularization: max_depth=4, min_child_weight=20, subsample=0.7,
    colsample_bytree=0.7, reg_lambda=5.

Portfolio: every N trading days rank LETFs by predicted return, equal-weight
the top K; remainder in cash at 0%. Weights decided from close[t-1] features
fill at open[t]; returns are booked with the unified realization-dated
convention (see alt/sleeve_engine.py): the value at date t is the P&L
realized over open[t-1] -> open[t]. 10 bps/side TC.

Outputs
  data/results/quantum_metrics.json
  data/results/quantum_returns.csv        (starts at WF_START)
  data/results/quantum_model.pkl          (per-year model cache)
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except ImportError as e:
    raise SystemExit("xgboost required: pip install xgboost") from e


ROOT = Path(__file__).resolve().parent.parent
ETF_DIR = ROOT / "data/etfs"
FRED_DIR = ROOT / "data/fred"
RESULTS = ROOT / "data/results"
RESULTS.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    "TQQQ", "UPRO", "QLD", "SSO", "SOXL", "TECL",
    "FAS", "ERX", "DRN", "EDC", "YINN",
    "UCO", "UGL", "NUGT",
    "TMF", "UBT", "TYD",
]
BENCH = "SPY"

DATA_START = "2010-03-11"   # first date of usable feature history
WF_START = "2014-01-02"     # first walk-forward (publishable) date
IS_END = "2018-12-31"       # kept for reporting splits only
OOS_START = "2019-01-02"

TC_BPS = 10.0  # per side
TRADING_DAYS = 252
SEED = 42
K_HELD = 3     # design constant — number of names held (NOT CV-selected)
DEFAULT_N = 21


# ------------------------------------------------------------------ loaders
def load_etf(tkr: str) -> pd.DataFrame:
    df = pd.read_csv(ETF_DIR / f"{tkr}.csv", parse_dates=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"])
    df = df.set_index("Date")[["Open", "Close"]].astype(float)
    return df


def load_all_prices() -> Tuple[pd.DataFrame, pd.DataFrame]:
    opens, closes = {}, {}
    for t in UNIVERSE + [BENCH]:
        d = load_etf(t)
        opens[t] = d["Open"]
        closes[t] = d["Close"]
    opens = pd.concat(opens, axis=1).sort_index()
    closes = pd.concat(closes, axis=1).sort_index()
    return opens, closes


def load_macro(index: pd.DatetimeIndex) -> pd.DataFrame:
    vix = pd.read_csv(FRED_DIR / "VIXCLS.csv", parse_dates=["Date"]).set_index("Date")["VIXCLS"]
    oas = pd.read_csv(FRED_DIR / "BAMLH0A0HYM2.csv", parse_dates=["Date"]).set_index("Date")["BAMLH0A0HYM2"]
    t10y2y = pd.read_csv(FRED_DIR / "T10Y2Y.csv", parse_dates=["Date"]).set_index("Date")["T10Y2Y"]
    macro = pd.concat({"VIX": vix, "HYOAS": oas, "T10Y2Y": t10y2y}, axis=1)
    macro = macro.reindex(index).ffill()
    macro["VIX_chg21"] = macro["VIX"].diff(21)
    macro["HYOAS_slope21"] = macro["HYOAS"].diff(21)
    return macro


# ------------------------------------------------------------------ features
def build_features(opens: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """Return long dataframe indexed by (Date, Ticker) with features.

    All features use data up to and including close[t-1] — i.e., computed on
    the prior day's close. The row's Date = t (the signal date, which will
    drive the open[t] fill).
    """
    bench_c = closes[BENCH]
    spy_ma21 = bench_c.rolling(21).mean()
    spy_ma63 = bench_c.rolling(63).mean()
    spy_ma_spread = (spy_ma21 - spy_ma63) / spy_ma63

    idx = closes.index
    macro = load_macro(idx)

    frames = []
    for t in UNIVERSE:
        c = closes[t]
        r = np.log(c / c.shift(1))

        feat = pd.DataFrame(index=idx)
        feat["mom_5"] = np.log(c / c.shift(5))
        feat["mom_21"] = np.log(c / c.shift(21))
        feat["mom_63"] = np.log(c / c.shift(63))
        feat["mom_252"] = np.log(c / c.shift(252))

        vol21 = r.rolling(21).std()
        vol63 = r.rolling(63).std()
        feat["vol_21"] = vol21
        feat["vol_63"] = vol63

        feat["sharpe_21"] = r.rolling(21).mean() / vol21.replace(0, np.nan)
        feat["sharpe_63"] = r.rolling(63).mean() / vol63.replace(0, np.nan)

        feat["exc_21"] = feat["mom_21"] - np.log(bench_c / bench_c.shift(21))
        feat["exc_63"] = feat["mom_63"] - np.log(bench_c / bench_c.shift(63))

        ma200 = c.rolling(200).mean()
        feat["d_200dma"] = (c - ma200) / ma200

        # Shift by 1: features represent state at close[t-1] used on date t.
        feat = feat.shift(1)

        # macro (also shift by 1 so using prior close's macro)
        for mc in ["VIX", "VIX_chg21", "HYOAS", "HYOAS_slope21", "T10Y2Y"]:
            feat[mc] = macro[mc].shift(1)
        feat["SPY_MA_spread"] = spy_ma_spread.shift(1)

        feat["Ticker"] = t
        feat["Date"] = idx
        frames.append(feat.reset_index(drop=True))

    long = pd.concat(frames, ignore_index=True)

    # cross-sectional ranks per date for ticker-level features
    rank_cols = [
        "mom_5", "mom_21", "mom_63", "mom_252",
        "vol_21", "vol_63",
        "sharpe_21", "sharpe_63",
        "exc_21", "exc_63",
        "d_200dma",
    ]
    for col in rank_cols:
        long[f"rk_{col}"] = long.groupby("Date")[col].rank(pct=True)

    return long.set_index(["Date", "Ticker"]).sort_index()


def build_targets(closes: pd.DataFrame, N: int) -> pd.DataFrame:
    """Forward N-day log return for each ticker, aligned to signal date t."""
    frames = []
    for t in UNIVERSE:
        c = closes[t]
        fwd = np.log(c.shift(-N) / c)
        frames.append(pd.DataFrame({"Date": c.index, "Ticker": t, "fwd_ret": fwd.values}))
    long = pd.concat(frames, ignore_index=True).set_index(["Date", "Ticker"]).sort_index()
    return long


# ------------------------------------------------------------------ model
def make_model(seed=SEED):
    return xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=20,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    df = pd.DataFrame({"y": y_true, "yp": y_pred, "d": dates})
    ics = []
    for _, g in df.groupby("d"):
        if len(g) < 5:
            continue
        c = g["y"].rank().corr(g["yp"].rank())
        if np.isfinite(c):
            ics.append(c)
    return float(np.mean(ics)) if ics else float("nan")


def cv_select_N(df_first: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """Expanding-window CV inside the FIRST training window only to pick N.

    K is a design constant (K_HELD) and is not part of the search — the CV
    objective (per-date rank IC of predictions vs realized N-day returns)
    does not depend on how many names are subsequently held.
    """
    candidates = [5, 10, 21, 42]
    scores = {}
    dates = df_first.index.get_level_values("Date")
    unique_dates = np.array(sorted(set(dates)))
    n_folds = 3
    fold_size = len(unique_dates) // (n_folds + 1)

    for N in candidates:
        target_col = f"fwd_{N}"
        if target_col not in df_first.columns:
            continue
        fold_ics = []
        for f in range(n_folds):
            train_end_idx = fold_size * (f + 1)
            val_start_idx = train_end_idx + N  # embargo = N days
            val_end_idx = min(train_end_idx + fold_size, len(unique_dates) - 1)
            if val_start_idx >= val_end_idx:
                continue
            train_end = unique_dates[train_end_idx - 1]
            val_start = unique_dates[val_start_idx]
            val_end = unique_dates[val_end_idx - 1]

            tr = df_first[(dates <= train_end)].dropna(subset=feature_cols + [target_col])
            va_mask = (dates >= val_start) & (dates <= val_end)
            va = df_first[va_mask].dropna(subset=feature_cols + [target_col])
            if len(tr) < 1000 or len(va) < 200:
                continue

            m = make_model()
            m.fit(tr[feature_cols].values, tr[target_col].values, verbose=False)
            pred = m.predict(va[feature_cols].values)
            ic = rank_ic(va[target_col].values, pred, va.index.get_level_values("Date").values)
            fold_ics.append(ic)
        mean_ic = float(np.nanmean(fold_ics)) if fold_ics else float("nan")
        scores[N] = {"mean_ic": mean_ic, "folds": fold_ics}

    best_N, best_ic = None, -np.inf
    for N, v in scores.items():
        if np.isfinite(v["mean_ic"]) and v["mean_ic"] > best_ic:
            best_ic = v["mean_ic"]
            best_N = N
    if best_N is None:
        print(f"[WARN] CV produced no valid scores; falling back to N={DEFAULT_N}")
        best_N, best_ic = DEFAULT_N, 0.0
    return {"best_N": best_N, "best_ic": best_ic,
            "scores": {str(n): v for n, v in scores.items()}}


# ------------------------------------------------------------------ walk-forward fitting
def fit_walk_forward(data: pd.DataFrame, feature_cols: List[str], N: int,
                     years: List[int]) -> Dict[int, "xgb.XGBRegressor"]:
    """Fit one model per prediction year Y on all data with Date <= Dec-31(Y-1),
    minus an N-trading-day embargo before the cutoff (so no training target
    overlaps year Y)."""
    dates = data.index.get_level_values("Date")
    unique_dates = np.array(sorted(set(dates)))
    tgt_col = f"fwd_{N}"
    models: Dict[int, xgb.XGBRegressor] = {}
    for y in years:
        cutoff = pd.Timestamp(f"{y - 1}-12-31")
        eligible = unique_dates[unique_dates <= np.datetime64(cutoff)]
        if len(eligible) <= N + 250:
            continue
        train_end = eligible[-(N + 1)]  # embargo: last N dates' targets leak past cutoff
        tr = data[dates <= train_end].dropna(subset=feature_cols + [tgt_col])
        if len(tr) < 2000:
            continue
        m = make_model()
        m.fit(tr[feature_cols].values, tr[tgt_col].values, verbose=False)
        models[y] = m
        print(f"  fit year {y}: train <= {pd.Timestamp(train_end).date()}  rows={len(tr)}")
    return models


def predict_walk_forward(data: pd.DataFrame, feature_cols: List[str],
                         models: Dict[int, "xgb.XGBRegressor"]) -> pd.DataFrame:
    """Predictions for every (Date, Ticker) row whose year has a model,
    strictly from that year's model (trained on prior years only)."""
    dates = data.index.get_level_values("Date")
    parts = []
    for y, m in sorted(models.items()):
        mask = (dates.year == y)
        rows = data[mask].dropna(subset=feature_cols)
        if len(rows) == 0:
            continue
        pv = m.predict(rows[feature_cols].values)
        parts.append(pd.DataFrame({"pred": pv}, index=rows.index))
    if not parts:
        return pd.DataFrame(columns=["pred"])
    return pd.concat(parts).sort_index()


# ------------------------------------------------------------------ weights
def build_wf_weights(closes: pd.DataFrame, preds: pd.DataFrame,
                     N: int, K: int) -> pd.DataFrame:
    """Daily decision-dated weight frame: rebalance every N trading days from
    WF_START, top-K equal weight by prediction; cash (0%) otherwise. W[t] is
    the weight to hold from open[t], decided from close[t-1] features."""
    all_dates = closes.index
    bt_dates = all_dates[all_dates >= pd.Timestamp(WF_START)]
    reb_set = set(bt_dates[::N])

    pred_dates = set(preds.index.get_level_values("Date")) if len(preds) else set()
    W = pd.DataFrame(0.0, index=bt_dates, columns=UNIVERSE)
    current_w = pd.Series(0.0, index=UNIVERSE)
    for d in bt_dates:
        if d in reb_set:
            current_w = pd.Series(0.0, index=UNIVERSE)
            if d in pred_dates:
                sl = preds.xs(d, level="Date", drop_level=True)["pred"].dropna()
                if len(sl) >= K:
                    top = sl.nlargest(K).index
                    current_w.loc[top] = 1.0 / K
        W.loc[d] = current_w.values
    return W


# ------------------------------------------------------------------ live weight builder
def build_weights(live_extend: bool = False) -> pd.DataFrame:
    """Compute the canonical QUANTUM daily target-weight DataFrame from the
    cached per-year walk-forward models (RESULTS/quantum_model.pkl).

    live_extend: If True, extend the close index by one BDay forward
        (ffilled) so the last row is W[t+1] using close[t] info.
    """
    cache_path = RESULTS / "quantum_model.pkl"
    if not cache_path.exists():
        raise RuntimeError(
            f"QUANTUM cache missing at {cache_path}. Run quantum_strategy.py first.")
    with open(cache_path, "rb") as f:
        cached = pickle.load(f)
    models = cached["models"]
    N = cached["N"]
    K = cached.get("K", K_HELD)
    cached_cols = cached["feature_cols"]

    opens, closes = load_all_prices()
    if live_extend and len(closes) > 0:
        next_day = closes.index[-1] + pd.tseries.offsets.BDay()
        opens.loc[next_day] = opens.iloc[-1]
        closes.loc[next_day] = closes.iloc[-1]
        opens = opens.sort_index()
        closes = closes.sort_index()
    feats = build_features(opens, closes)
    feature_cols = [c for c in cached_cols if c in feats.columns]
    if len(feature_cols) != len(cached_cols):
        raise RuntimeError("QUANTUM cache feature set no longer matches build_features()")

    # A prediction year with no fitted model (e.g. current year missing from a
    # stale cache) falls back to the latest available model.
    dates = feats.index.get_level_values("Date")
    years_needed = sorted(set(dates[dates >= pd.Timestamp(WF_START)].year))
    latest = max(models.keys())
    models_use = {y: models.get(y, models[latest]) for y in years_needed}

    preds = predict_walk_forward(feats, feature_cols, models_use)
    return build_wf_weights(closes, preds, N=N, K=K)


# ------------------------------------------------------------------ metrics
def compute_metrics(r: pd.Series, label: str) -> Dict:
    r = r.dropna()
    if len(r) == 0:
        return {"label": label}
    ann = float(r.mean() * TRADING_DAYS)
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    sharpe = ann / vol if vol > 0 else float("nan")
    eq = (1.0 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    peak = eq.cummax()
    dd = eq / peak - 1.0
    mdd = float(dd.min())
    hit = float((r > 0).mean())
    return {
        "label": label, "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
        "n_days": len(r), "ann_ret": ann, "ann_vol": vol, "sharpe": sharpe,
        "cagr": cagr, "mdd": mdd, "hit_rate": hit,
    }


# ------------------------------------------------------------------ main
def main():
    from sleeve_engine import backtest_weights

    print("Loading prices...")
    opens, closes = load_all_prices()

    print("Building features...")
    feats = build_features(opens, closes)
    feature_cols = [c for c in feats.columns if c not in ("Ticker", "Date")]

    print("Building targets for all candidate horizons...")
    Ns = [5, 10, 21, 42]
    tgt_frames = []
    for N in Ns:
        t = build_targets(closes, N).rename(columns={"fwd_ret": f"fwd_{N}"})
        tgt_frames.append(t)
    tgts = pd.concat(tgt_frames, axis=1)

    data = feats.join(tgts, how="left")
    dates_idx = data.index.get_level_values("Date")
    data = data[dates_idx >= pd.Timestamp(DATA_START)]
    dates_idx = data.index.get_level_values("Date")

    years_needed = sorted(set(dates_idx[dates_idx >= pd.Timestamp(WF_START)].year))
    last_data_year = int(dates_idx.max().year)

    # CACHE: reuse per-year models when the cache covers every prediction
    # year in the data. Past years' models are frozen by construction
    # (their training windows are closed), so only a new calendar year
    # triggers a refit.
    cache_path = RESULTS / "quantum_model.pkl"
    cache_hit = False
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if (cached.get("feature_cols") == feature_cols
                    and cached.get("wf_start") == WF_START
                    and set(years_needed) <= set(cached.get("models", {}).keys())):
                models = cached["models"]
                N = cached["N"]
                cv = cached["cv"]
                print(f"[CACHE HIT] {len(models)} yearly models from {cache_path} (N={N})")
                cache_hit = True
        except Exception as e:
            print(f"[CACHE MISS] Failed to load cache ({e}); refitting.")

    if not cache_hit:
        first_window_mask = dates_idx <= pd.Timestamp(f"{int(pd.Timestamp(WF_START).year) - 1}-12-31")
        print("Selecting N by CV inside the first training window (2010-2013)...")
        cv = cv_select_N(data[first_window_mask], feature_cols)
        N = cv["best_N"]
        print(f"CV best: N={N} (mean rank-IC {cv['best_ic']:.4f}); K={K_HELD} (design constant)")

        print("Fitting walk-forward models (one per prediction year)...")
        models = fit_walk_forward(data, feature_cols, N, years_needed)
        if not models:
            raise RuntimeError("QUANTUM: walk-forward fitting produced no models")
        with open(cache_path, "wb") as f:
            pickle.dump({"models": models, "N": N, "K": K_HELD, "cv": cv,
                         "wf_start": WF_START, "feature_cols": feature_cols}, f)
        print(f"[CACHE WRITE] Saved {len(models)} yearly models to {cache_path}")

    # Predictions strictly from each year's prior-data model
    preds = predict_walk_forward(data, feature_cols, models)

    print("Building weights + running unified backtest...")
    W = build_wf_weights(closes, preds, N=N, K=K_HELD)
    bt = backtest_weights(W, opens.loc[W.index.min():], cost_bps=TC_BPS)
    port_ret = bt["ret"]

    # Reporting splits: "WF-IS" (2014-2018, walk-forward but overlapping the
    # blend-weight fitting window) and OOS (2019+).
    is_r = port_ret.loc[:IS_END]
    oos_r = port_ret.loc[OOS_START:]
    m_is = compute_metrics(is_r, "WF_2014_2018")
    m_oos = compute_metrics(oos_r, "OOS_2019on")
    m_full = compute_metrics(port_ret, "FULL_WF")

    out_returns = pd.DataFrame({"ret": port_ret})
    out_returns.index.name = "Date"
    out_returns.to_csv(RESULTS / "quantum_returns.csv")

    fi_last = {}
    try:
        last_model = models[max(models.keys())]
        fi = dict(zip(feature_cols, last_model.feature_importances_.astype(float).tolist()))
        fi_last = dict(sorted(fi.items(), key=lambda kv: -kv[1])[:15])
    except Exception:
        pass

    metrics = {
        "strategy": "QUANTUM (walk-forward)",
        "params": {
            "N": int(N), "K": int(K_HELD),
            "rebalance_cadence_days": int(N),
            "tc_bps_per_side": TC_BPS,
            "universe": UNIVERSE,
            "wf_start": WF_START,
            "refit": "annual expanding window, N-day embargo before cutoff",
            "model": "xgboost.XGBRegressor",
            "xgb_params": {
                "n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
                "min_child_weight": 20, "subsample": 0.7, "colsample_bytree": 0.7,
                "reg_lambda": 5.0, "objective": "reg:squarederror",
            },
        },
        "cv_first_window": cv,
        "fitted_years": sorted(int(y) for y in models.keys()),
        "feature_importance_top15_latest": fi_last,
        "WF_2014_2018": m_is, "OOS_2019on": m_oos, "FULL_WF": m_full,
    }
    (RESULTS / "quantum_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    print("\n==== QUANTUM (walk-forward) ====")
    for m in (m_is, m_oos, m_full):
        if "sharpe" in m:
            print(f"{m['label']:>12} | Sharpe {m['sharpe']:.2f}  CAGR {m['cagr']*100:5.1f}%  "
                  f"MDD {m['mdd']*100:6.1f}%  Vol {m['ann_vol']*100:5.1f}%  n={m['n_days']}")


if __name__ == "__main__":
    main()
