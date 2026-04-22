"""QUANTUM — ML signal ensemble for leveraged-ETF rotation.

Angle
-----
Train a gradient-boosted regressor (xgboost) on IS (2010-03-11..2018-12-31)
to predict each LETF's next-N-day log return. Features per (date, ticker):

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

Targets: next N-day log return (forward).

At each rebalance (every N days), rank LETFs by predicted return and
equal-weight the top K; remainder to cash (0% return). Close[t-1] signals
drive open[t] fills with 10 bps/side TC.

Anti-overfit:
  * Train STRICTLY on IS only; model frozen for OOS.
  * Embargo: drop training rows whose target window overlaps within N days
    of the validation fold boundary.
  * Heavy regularization: max_depth=4, min_child_weight=20, subsample=0.7,
    colsample_bytree=0.7, reg_lambda=5.
  * Hyperparam (N, K) selected via K-fold expanding CV inside IS using
    rank-IC (Spearman) of predicted vs realized N-day return — never touch
    OOS during selection.

Outputs
  /home/user/bonds/data/results/quantum_metrics.json
  /home/user/bonds/data/results/quantum_returns.csv
"""
from __future__ import annotations

import json
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


ROOT = Path("/home/user/bonds")
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

IS_START = "2010-03-11"
IS_END = "2018-12-31"
OOS_START = "2019-01-02"
OOS_END = "2026-04-02"

TC_BPS = 10.0  # per side
TRADING_DAYS = 252
SEED = 42


# ------------------------------------------------------------------ loaders
def load_etf(tkr: str) -> pd.DataFrame:
    df = pd.read_csv(ETF_DIR / f"{tkr}.csv", parse_dates=["Date"]).sort_values("Date")
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
    spy_ret = np.log(bench_c / bench_c.shift(1))
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
    """Forward N-day log return for each ticker, aligned to signal date t.

    We use close-to-close next N. Realized strategy PnL uses open[t]→open[t+N]
    (see execution), but for the ML target rank-IC we just use close_t→close_{t+N}
    as a ranking objective; this only affects TRAIN targets, not OOS returns.
    """
    frames = []
    for t in UNIVERSE:
        c = closes[t]
        fwd = np.log(c.shift(-N) / c)
        frames.append(pd.DataFrame({"Date": c.index, "Ticker": t, "fwd_ret": fwd.values}))
    long = pd.concat(frames, ignore_index=True).set_index(["Date", "Ticker"]).sort_index()
    return long


# ------------------------------------------------------------------ model
FEATURE_COLS = None  # set in main


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


def cv_select(df_is: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """Walk-forward CV inside IS to pick (N, K). Returns best config + mean IC."""
    candidates = [(N, K) for N in [5, 10, 21, 42] for K in [3, 4, 5]]
    scores = {}
    dates = df_is.index.get_level_values("Date")
    unique_dates = np.array(sorted(set(dates)))
    n_folds = 4
    fold_size = len(unique_dates) // (n_folds + 1)

    for N, K in candidates:
        target_col = f"fwd_{N}"
        if target_col not in df_is.columns:
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

            tr = df_is[(dates <= train_end)].dropna(subset=feature_cols + [target_col])
            va_mask = (dates >= val_start) & (dates <= val_end)
            va = df_is[va_mask].dropna(subset=feature_cols + [target_col])
            if len(tr) < 2000 or len(va) < 200:
                continue

            m = make_model()
            m.fit(tr[feature_cols].values, tr[target_col].values, verbose=False)
            pred = m.predict(va[feature_cols].values)
            ic = rank_ic(va[target_col].values, pred, va.index.get_level_values("Date").values)
            fold_ics.append(ic)
        mean_ic = float(np.nanmean(fold_ics)) if fold_ics else float("nan")
        scores[(N, K)] = {"mean_ic": mean_ic, "folds": fold_ics}

    # choose best by mean IC; K within (IC, K) tie broken by IC at that N
    best = None
    best_ic = -np.inf
    for (N, K), v in scores.items():
        if not np.isfinite(v["mean_ic"]):
            continue
        if v["mean_ic"] > best_ic:
            best_ic = v["mean_ic"]
            best = (N, K)
    return {"best_N": best[0], "best_K": best[1], "best_ic": best_ic, "scores": {f"{n}_{k}": v for (n, k), v in scores.items()}}


# ------------------------------------------------------------------ backtest
def backtest(opens: pd.DataFrame, closes: pd.DataFrame, preds: pd.DataFrame,
             N: int, K: int) -> pd.Series:
    """Rebalance every N trading days: at signal date t (close[t-1] info), fill
    top-K equal weight at open[t]. Hold until next rebalance.

    Returns daily strategy returns from open-to-open within the holding period,
    with transaction costs applied on rebalance days.

    preds : DataFrame indexed (Date, Ticker) with column 'pred'.
    """
    # Build open-to-open daily returns for each ticker:
    # ret on day t (from open[t] close-through to open[t+1]) computed as
    # open[t+1] / open[t] - 1. We'll use close-to-close within hold but apply
    # the rebalance fill at the open of the rebalance day.
    # Simpler and standard: compute daily close-to-close returns; at rebalance,
    # charge TC and assume entry at open (so the rebalance day's return is
    # open->close of that day, which is close[t]/open[t]-1, for NEW positions).

    all_dates = closes.index
    # rebalance dates: every N trading days starting from first valid date
    start_bt = pd.Timestamp(IS_START)
    end_bt = pd.Timestamp(OOS_END)
    mask_bt = (all_dates >= start_bt) & (all_dates <= end_bt)
    bt_dates = all_dates[mask_bt]

    # pick rebalance dates every N
    reb_dates = bt_dates[::N]

    c2c = closes[UNIVERSE].pct_change()
    o2c = (closes[UNIVERSE] / opens[UNIVERSE] - 1.0)  # intraday return (open->close)

    port_ret = pd.Series(0.0, index=bt_dates)
    current_w = pd.Series(0.0, index=UNIVERSE)

    reb_set = set(reb_dates)
    # precompute reb_date -> target weights
    target_w_map: Dict[pd.Timestamp, pd.Series] = {}
    for d in reb_dates:
        if d not in preds.index.get_level_values("Date"):
            target_w_map[d] = pd.Series(0.0, index=UNIVERSE)
            continue
        sl = preds.xs(d, level="Date", drop_level=True)["pred"]
        sl = sl.dropna()
        if len(sl) < K:
            target_w_map[d] = pd.Series(0.0, index=UNIVERSE)
            continue
        top = sl.nlargest(K).index
        w = pd.Series(0.0, index=UNIVERSE)
        w.loc[top] = 1.0 / K
        target_w_map[d] = w

    prev_w = pd.Series(0.0, index=UNIVERSE)
    for i, d in enumerate(bt_dates):
        if d in reb_set:
            new_w = target_w_map[d].fillna(0.0)
            turnover = (new_w - prev_w).abs().sum()
            tc = turnover * (TC_BPS / 1e4)
            # on rebalance day, new positions earn open->close
            day_ret_vec = o2c.loc[d].fillna(0.0)
            gross = float((new_w * day_ret_vec).sum())
            port_ret.loc[d] = gross - tc
            current_w = new_w * (1.0 + day_ret_vec)
            current_w = current_w / current_w.sum() if current_w.sum() != 0 else current_w
            prev_w = new_w  # bookkeeping: for next rebalance turnover calc
        else:
            day_ret_vec = c2c.loc[d].fillna(0.0)
            gross = float((current_w * day_ret_vec).sum())
            port_ret.loc[d] = gross
            # drift weights
            current_w = current_w * (1.0 + day_ret_vec)
            s = current_w.sum()
            if s > 0:
                current_w = current_w / s
    return port_ret


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
    global FEATURE_COLS
    print("Loading prices...")
    opens, closes = load_all_prices()

    # Align on common index where UNIVERSE members have data
    first_valid = closes[UNIVERSE].dropna(how="any").index.min()
    print(f"First date with all-universe data: {first_valid.date()}")

    print("Building features...")
    feats = build_features(opens, closes)
    feature_cols = [c for c in feats.columns if c not in ("Ticker", "Date")]
    FEATURE_COLS = feature_cols

    print("Building targets for all candidate horizons...")
    Ns = [5, 10, 21, 42]
    tgt_frames = []
    for N in Ns:
        t = build_targets(closes, N).rename(columns={"fwd_ret": f"fwd_{N}"})
        tgt_frames.append(t)
    tgts = pd.concat(tgt_frames, axis=1)

    data = feats.join(tgts, how="left")

    # Slice IS
    dates_idx = data.index.get_level_values("Date")
    is_mask = (dates_idx >= pd.Timestamp(IS_START)) & (dates_idx <= pd.Timestamp(IS_END))
    data_is = data[is_mask]
    print(f"IS rows (pre-clean): {len(data_is)}")

    # CACHE: if a trained model + chosen-N cache exists, skip CV + fit entirely.
    # The training data is FROZEN at 2010-03-11..2018-12-31 (IS_END) so retraining
    # produces identical outputs. Caching drops runtime from ~60s to ~2s per cron.
    import pickle
    cache_path = RESULTS / "quantum_model.pkl"
    cache_hit = False
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            if (cached.get("is_end") == IS_END and
                cached.get("feature_cols") == feature_cols):
                final_model = cached["model"]
                N = cached["N"]
                K = cached["K"]
                cv = cached["cv"]
                print(f"[CACHE HIT] Loaded trained model from {cache_path}")
                print(f"  N={N}, K={K}, IC={cv['best_ic']:.4f}")
                cache_hit = True
        except Exception as e:
            print(f"[CACHE MISS] Failed to load cache ({e}); retraining.")

    if not cache_hit:
        print("Cross-validating (N, K) inside IS...")
        cv = cv_select(data_is, feature_cols)
        N = cv["best_N"]
        K = cv["best_K"]
        print(f"CV best: N={N}, K={K}, mean_rank_IC={cv['best_ic']:.4f}")

        # Train final model on ALL of IS using chosen N
        tgt_col = f"fwd_{N}"
        train = data_is.dropna(subset=feature_cols + [tgt_col])
        print(f"Final train rows: {len(train)}")
        final_model = make_model()
        final_model.fit(train[feature_cols].values, train[tgt_col].values, verbose=False)

        # Persist the trained artifact
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({"model": final_model, "N": N, "K": K, "cv": cv,
                             "is_end": IS_END, "feature_cols": feature_cols}, f)
            print(f"[CACHE WRITE] Saved trained model to {cache_path}")
        except Exception as e:
            print(f"[CACHE WRITE ERR] {e}")

    # Feature importances
    fi = dict(zip(feature_cols, final_model.feature_importances_.astype(float).tolist()))
    fi_sorted = dict(sorted(fi.items(), key=lambda kv: -kv[1])[:15])

    # Predict over full window
    full_mask = (dates_idx >= pd.Timestamp(IS_START)) & (dates_idx <= pd.Timestamp(OOS_END))
    data_full = data[full_mask].dropna(subset=feature_cols)
    preds_vec = final_model.predict(data_full[feature_cols].values)
    preds_df = pd.DataFrame({"pred": preds_vec}, index=data_full.index)

    print("Running backtest...")
    port_ret = backtest(opens, closes, preds_df, N=N, K=K)
    port_ret = port_ret.loc[pd.Timestamp(IS_START):pd.Timestamp(OOS_END)]

    # split IS / OOS / FULL
    is_r = port_ret.loc[IS_START:IS_END]
    oos_r = port_ret.loc[OOS_START:OOS_END]
    full_r = port_ret
    m_is = compute_metrics(is_r, "IS")
    m_oos = compute_metrics(oos_r, "OOS")
    m_full = compute_metrics(full_r, "FULL")

    # save
    out_returns = pd.DataFrame({"ret": port_ret})
    out_returns.to_csv(RESULTS / "quantum_returns.csv")

    metrics = {
        "strategy": "QUANTUM",
        "params": {
            "N": int(N), "K": int(K),
            "rebalance_cadence_days": int(N),
            "tc_bps_per_side": TC_BPS,
            "universe": UNIVERSE,
            "is": [IS_START, IS_END],
            "oos": [OOS_START, OOS_END],
            "model": "xgboost.XGBRegressor",
            "xgb_params": {
                "n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
                "min_child_weight": 20, "subsample": 0.7, "colsample_bytree": 0.7,
                "reg_lambda": 5.0, "objective": "reg:squarederror",
            },
        },
        "cv": {"best_N": int(N), "best_K": int(K), "best_ic": cv["best_ic"],
               "all_scores": cv["scores"]},
        "feature_importance_top15": fi_sorted,
        "IS": m_is, "OOS": m_oos, "FULL": m_full,
    }
    (RESULTS / "quantum_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    print("\n==== QUANTUM ====")
    for m in (m_is, m_oos, m_full):
        print(f"{m['label']:>4} | Sharpe {m['sharpe']:.2f}  CAGR {m['cagr']*100:5.1f}%  "
              f"MDD {m['mdd']*100:6.1f}%  Vol {m['ann_vol']*100:5.1f}%  n={m['n_days']}")
    print("Top features:")
    for k, v in fi_sorted.items():
        print(f"  {k:25s} {v:.4f}")


if __name__ == "__main__":
    main()
