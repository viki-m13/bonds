"""NOVA7 — ML walk-forward regime classifier + concentrated positions.

User green-light: concentrate positions, use leverage, inverse ETFs,
ML, math. But MUST NOT overfit or cherry-pick.

Design principles (to avoid overfit):
  1. One-pass walk-forward: retrain XGBoost monthly on last N years,
     no peeking at future data. Evaluated ONLY on strict post-training
     OOS returns.
  2. No hyperparameter tuning on OOS. Fixed model (100 trees, depth 3,
     lr 0.05). Fixed lookback (5y). Fixed feature list (chosen a priori
     from well-known macro/technical features).
  3. Fixed position buckets chosen before seeing results:
        p(up) > 0.60 → UPRO (3x SPY)
        p(up) ∈ (0.50, 0.60] → SPY
        p(up) ∈ (0.40, 0.50] → cash (BIL)
        p(up) ∈ (0.30, 0.40] → SH (−1x SPY)
        p(up) ≤ 0.30 → SDS (−2x SPY)
  4. Report both in-sample (training-window fit) AND strict OOS (each
     day's position comes from a model trained only on data BEFORE
     that day).
  5. TC 15 bps on regime changes.

Features (31, fixed a priori, NO tuning):
  Price momentum: SPY/QQQ/TLT/GLD/EFA 1m, 3m, 6m, 12m returns
  Trend: SPY vs SMA50, SMA100, SMA200
  Vol/VIX: VIX level, VIX 20d chg, VIX / VIX 63d-MA
  Credit: HY spread level + 20d chg, IG spread level
  Yield curve: T10Y2Y, T10Y3M levels
  Rates: 10y level, 2y level
  FX: DXY 63d return
  Unemployment: 12m chg
  Technical: SPY RSI14, RSI2, 20d vol
"""
from pathlib import Path
import numpy as np
import pandas as pd

from hydra_core import load_etf, load_fred, stats


TC_BPS = 15.0


def build_features(dates):
    """Fixed 31-feature matrix. No tuning, no selection, chosen a priori."""
    feats = {}
    spy = load_etf("SPY").reindex(dates).ffill()
    qqq = load_etf("QQQ").reindex(dates).ffill()
    tlt = load_etf("TLT").reindex(dates).ffill()
    gld = load_etf("GLD").reindex(dates).ffill()
    efa = load_etf("EFA").reindex(dates).ffill()

    for t_, p in [("spy", spy), ("qqq", qqq), ("tlt", tlt), ("gld", gld), ("efa", efa)]:
        feats[f"{t_}_r21"] = p.pct_change(21)
        feats[f"{t_}_r63"] = p.pct_change(63)
        feats[f"{t_}_r126"] = p.pct_change(126)
        feats[f"{t_}_r252"] = p.pct_change(252)

    for lb in [50, 100, 200]:
        feats[f"spy_vs_sma{lb}"] = spy / spy.rolling(lb).mean() - 1

    vix = load_fred("VIXCLS").reindex(dates).ffill()
    feats["vix"] = vix
    feats["vix_chg_20"] = vix.diff(20)
    feats["vix_rel_ma63"] = vix / vix.rolling(63).mean()

    hy = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    feats["hy"] = hy
    feats["hy_chg_20"] = hy.diff(20)
    ig = load_fred("BAMLC0A0CM").reindex(dates).ffill()
    feats["ig"] = ig

    feats["t10y2y"] = load_fred("T10Y2Y").reindex(dates).ffill()
    feats["t10y3m"] = load_fred("T10Y3M").reindex(dates).ffill()
    feats["r10"] = load_fred("DGS10").reindex(dates).ffill()
    feats["r2"] = load_fred("DGS2").reindex(dates).ffill()

    feats["dxy_r63"] = load_fred("DTWEXBGS").reindex(dates).ffill().pct_change(63)
    feats["unemp_chg_252"] = load_fred("UNRATE").reindex(dates).ffill().diff(252)

    # RSI-style technicals
    def rsi(p, n):
        d = p.diff()
        up = d.clip(lower=0).rolling(n).mean()
        dn = (-d.clip(upper=0)).rolling(n).mean()
        rs = up / dn.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).fillna(50)
    feats["spy_rsi14"] = rsi(spy, 14)
    feats["spy_rsi2"] = rsi(spy, 2)
    feats["spy_vol20"] = spy.pct_change().rolling(20).std() * np.sqrt(252)

    return pd.DataFrame(feats).ffill()


def monthly_first_flag(index):
    out = pd.Series(False, index=index)
    out.iloc[0] = True
    for i in range(1, len(index)):
        if index[i].month != index[i - 1].month:
            out.iloc[i] = True
    return out


def walk_forward_predict(X, y, train_years=5, min_train=500):
    """Retrain on first of each month using last `train_years` of data
    (shifted to avoid target leakage). Predict p(up) at month-start.
    Returns a daily series of month-held predictions (ffill)."""
    from xgboost import XGBClassifier

    dates = X.index
    first_idx = np.where(monthly_first_flag(pd.Index(dates)).values)[0]

    pred = pd.Series(np.nan, index=dates)

    for fi in first_idx:
        ms = dates[fi]
        # train: data available STRICTLY before ms
        # target y is forward 21d return. So the last valid training sample
        # must have its 21-day forward window already CLOSED by ms-1.
        # Safe rule: drop last 21 rows of X/y at the edge.
        train_end_idx = fi - 1
        if train_end_idx < min_train:
            continue
        # Drop last 21 from train to avoid forward-leakage
        train_cut = train_end_idx - 21
        if train_cut < min_train:
            continue
        start_cut = max(0, train_cut - 252 * train_years)
        tX = X.iloc[start_cut:train_cut + 1].values
        ty = y.iloc[start_cut:train_cut + 1].values
        if len(ty) < min_train or len(np.unique(ty)) < 2:
            continue
        try:
            model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0,
                eval_metric="logloss",
            )
            model.fit(tX, ty)
            p_up = model.predict_proba(X.iloc[[fi]].values)[0, 1]
            pred.iloc[fi] = p_up
        except Exception as e:
            continue

    pred = pred.ffill()
    return pred


def bucket_position(p_up):
    """Map p(up) → (tic, weight). Buckets fixed before results."""
    if pd.isna(p_up):
        return "BIL", 0.0
    if p_up > 0.60:
        return "UPRO", 1.0
    if p_up > 0.50:
        return "SPY", 1.0
    if p_up > 0.40:
        return "BIL", 1.0
    if p_up > 0.30:
        return "SH", 1.0
    return "SDS", 1.0


def build_nova7(dates):
    print("Building features...", flush=True)
    X = build_features(dates)
    spy = load_etf("SPY").reindex(dates).ffill()
    # Target: forward 21-trading-day return > 0
    fwd_ret = spy.pct_change(21).shift(-21)
    y = (fwd_ret > 0).astype(int)

    keep = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[keep]
    y = y.loc[keep]

    print("Walk-forward training (monthly, 5y rolling, XGBoost)...", flush=True)
    p_up = walk_forward_predict(X, y, train_years=5, min_train=500)
    p_up = p_up.reindex(dates)

    # Position assignment (locked at month-start, 1-bar lag)
    print("Assigning positions...", flush=True)
    first = monthly_first_flag(pd.Index(dates))
    position_tic = pd.Series("", index=dates, dtype=object)
    last_tic = "BIL"
    for i, d in enumerate(dates):
        if first.iloc[i] and not pd.isna(p_up.loc[d]):
            last_tic = bucket_position(p_up.loc[d])[0]
        position_tic.iloc[i] = last_tic

    # Shift by 1 day for execution lag
    position_tic = position_tic.shift(1).fillna("BIL")

    # Load all possible tickers
    asset_rets = {}
    for t in ["UPRO", "SPY", "BIL", "SH", "SDS"]:
        p = load_etf(t)
        asset_rets[t] = (p.reindex(dates).ffill().pct_change().fillna(0) if p is not None
                         else pd.Series(0.0, index=dates))

    # Compute return: for each day, take the return of the asset we're in
    r = pd.Series(0.0, index=dates)
    for t in ["UPRO", "SPY", "BIL", "SH", "SDS"]:
        mask = position_tic == t
        r.loc[mask] = asset_rets[t].loc[mask]

    # TC on position changes
    changes = (position_tic != position_tic.shift(1)).astype(int)
    tc = changes * (TC_BPS / 1e4) * 2  # 2 legs (out + in)
    r = r - tc
    return r, p_up, position_tic


def main():
    spy = load_etf("SPY")
    dates = spy.index
    print(f"Universe: {dates[0].date()} .. {dates[-1].date()} ({len(dates)/252:.1f}y)")
    print("NOVA7 — ML walk-forward + concentrated leveraged/inverse ETFs\n")

    r, p_up, pos = build_nova7(dates)
    # First date where model predicted something
    first_pred = p_up.first_valid_index()
    print(f"\nFirst model prediction: {first_pred.date() if first_pred else 'NEVER'}")
    print(f"Position distribution (ex-BIL pre-start):")
    print(pos[pos != ""].value_counts())

    # Strict OOS (all predictions are walk-forward, so full-series IS OOS)
    # But isolate clearly from first_pred
    r_valid = r.loc[first_pred:] if first_pred is not None else r
    s = stats(r_valid, "NOVA7 (full walk-forward)")
    print(f"\n{s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    # Annual stats
    ann = r_valid.groupby(r_valid.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual:")
    print(ann.to_string())

    # Rolling 5y windows (overlapping not)
    print("\n5y windows:")
    for y0 in range(2011, 2022):
        y1 = y0 + 5
        lo = pd.Timestamp(f"{y0}-01-01")
        hi = pd.Timestamp(f"{y1}-01-01")
        sub = r_valid.loc[lo:hi]
        if len(sub) < 200:
            continue
        s = stats(sub, f"{y0}-{y1-1}")
        print(f"  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  MDD={s['mdd']:>7.2f}%")

    # Save
    out_path = Path("/home/user/bonds/data/results/nova7_returns.csv")
    pd.DataFrame({"NOVA7": r, "p_up": p_up, "position": pos}).to_csv(out_path)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
