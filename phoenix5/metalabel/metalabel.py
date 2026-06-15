"""Meta-labeling risk overlay for PHOENIX (Lopez de Prado style).

The "primary model" is PHOENIX itself (always-on). The META model predicts when
PHOENIX is about to have a BAD stretch, and we size exposure DOWN beforehand
(the overlay can only reduce risk, never add leverage — idle capital earns BIL).

Strict protocol to avoid fooling ourselves:
  * Features at date t use only information through close[t-1].
  * Target = forward 5-day PHOENIX net return < 0 (binary "bad").
  * Model trained ONLY on IS (2010-2018) with purged + embargoed K-fold CV.
    Frozen before touching OOS. No OOS leakage at any step.
  * Overlay hyperparameters (alpha, p0, floor) chosen on IS Sharpe only.
  * Reported on OOS (2019+). The honest question: does OOS Sharpe AND drawdown
    improve, or did the classifier just memorize IS?

Two model classes compared: L2 logistic regression (low variance) and a shallow
gradient-boosted tree. We keep whichever has the better PURGED-CV score on IS,
then evaluate once on OOS.

Outputs under phoenix5/metalabel/results/.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "data/results"
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
OUT = ROOT / "phoenix5/metalabel/results"
OUT.mkdir(parents=True, exist_ok=True)

IS_END = pd.Timestamp("2018-12-31")
OOS = pd.Timestamp("2019-01-02")
W_PROD = {"VANGUARD": 0.236, "ORION": 0.327, "HELIOS": 0.185,
          "QUANTUM": 0.152, "CRYPTO": 0.101}
HORIZON = 5


def px(t):
    s = pd.read_csv(ETF / f"{t}.csv", parse_dates=["Date"], index_col="Date")["Close"]
    return s[~s.index.duplicated()].sort_index()


def fred_series(name):
    s = pd.read_csv(FRED / f"{name}.csv", parse_dates=["Date"], index_col="Date")[name]
    return pd.to_numeric(s, errors="coerce")


def metrics(r):
    r = r.dropna()
    if len(r) < 60:
        return {}
    mu, sd = r.mean() * 252, r.std() * np.sqrt(252)
    c = (1 + r).cumprod()
    mdd = (c / c.cummax() - 1).min()
    yrs = len(r) / 252
    neg = r[r < 0]
    return {"sr": round(float(mu / sd), 3),
            "sortino": round(float(mu / (neg.std() * np.sqrt(252))), 3) if len(neg) else None,
            "cagr": round(float(c.iloc[-1] ** (1 / yrs) - 1), 4),
            "vol": round(float(sd), 4), "mdd": round(float(mdd), 4)}


def load_sleeves():
    van = pd.read_csv(R / "vanguard_returns.csv", parse_dates=[0], index_col=0)["net_ret"]
    ori = pd.read_csv(R / "orion_returns.csv", parse_dates=["Date"]).set_index("Date")["orion"]
    hel = pd.read_csv(R / "helios_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    qua = pd.read_csv(R / "quantum_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    cry = pd.read_csv(R / "crypto_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    return pd.concat({"VANGUARD": van, "ORION": ori, "HELIOS": hel,
                      "QUANTUM": qua, "CRYPTO": cry}, axis=1, sort=True).fillna(0.0).loc["2010-03-11":]


def build_features(sleeves, blend):
    f = pd.DataFrame(index=blend.index)
    # PHOENIX self-state
    for lb in [5, 10, 21]:
        f[f"ret_{lb}"] = blend.rolling(lb).sum()
    f["vol_21"] = blend.rolling(21).std()
    f["vol_63"] = blend.rolling(63).std()
    f["vol_accel"] = blend.rolling(5).std() / blend.rolling(63).std()
    cum = (1 + blend).cumprod()
    f["dd_63"] = cum / cum.rolling(63, min_periods=20).max() - 1
    f["dd_252"] = cum / cum.rolling(252, min_periods=40).max() - 1
    # sleeve dispersion & co-movement (correlation spikes precede bad stretches)
    f["sleeve_disp"] = sleeves.rolling(21).std().mean(axis=1)
    rollcorr = sleeves.rolling(63).corr().groupby(level=0).apply(
        lambda x: x.values[np.triu_indices(5, 1)].mean())
    f["sleeve_corr"] = pd.Series(rollcorr.values, index=sleeves.index[:len(rollcorr)]).reindex(f.index).ffill()
    # macro
    vix = fred_series("VIXCLS").reindex(f.index).ffill()
    f["vix"] = vix
    f["vix_chg5"] = vix.diff(5)
    f["vix_z"] = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
    oas = fred_series("BAMLH0A0HYM2").reindex(f.index).ffill()
    f["hyoas"] = oas
    f["hyoas_chg20"] = oas.diff(20)
    f["term"] = fred_series("T10Y2Y").reindex(f.index).ffill()
    dollar = px("UUP").pct_change().rolling(21).sum().reindex(f.index).ffill()
    f["dollar_21"] = dollar
    return f.shift(1)  # everything known only through t-1


def purged_cv_score(model_fn, X, y, n_splits=5, embargo=HORIZON):
    """Expanding purged K-fold: train on past, validate on a forward block,
    purge `embargo` rows around the boundary. Returns mean AUC."""
    idx = np.arange(len(X))
    fold = len(idx) // (n_splits + 1)
    aucs = []
    for k in range(1, n_splits + 1):
        tr_end = fold * k
        va_start = tr_end + embargo
        va_end = min(tr_end + fold, len(idx))
        if va_start >= va_end - 20:
            continue
        tr = idx[:tr_end - embargo]
        va = idx[va_start:va_end]
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        m = model_fn()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        aucs.append(roc_auc_score(y[va], p))
    return float(np.mean(aucs)) if aucs else 0.5


def main():
    sleeves = load_sleeves()
    blend = sleeves @ pd.Series(W_PROD)
    prod = pd.read_csv(R / "phoenix_production_returns.csv",
                       parse_dates=["Date"]).set_index("Date")["net_ret"]
    bil = px("BIL").pct_change().reindex(prod.index).fillna(0)

    feats = build_features(sleeves, blend)
    fwd = blend.shift(-HORIZON).rolling(HORIZON).sum()   # forward 5d (approx via reversed roll)
    fwd = blend[::-1].rolling(HORIZON).sum()[::-1].shift(-1)  # sum of t+1..t+H
    label = (fwd < 0).astype(int)

    data = feats.join(label.rename("y")).dropna()
    data = data.loc[:prod.index.max()]
    feat_cols = [c for c in feats.columns]
    Xall = data[feat_cols]
    yall = data["y"].values

    is_mask = data.index <= IS_END
    Xis, yis = Xall[is_mask], yall[is_mask]
    print(f"IS rows={is_mask.sum()}  bad-rate={yis.mean():.2f}   total feats={len(feat_cols)}")

    scaler = StandardScaler().fit(Xis.values)
    Xis_s = scaler.transform(Xis.values)

    def logit():
        return LogisticRegression(C=0.3, max_iter=1000, class_weight="balanced")

    def gbm():
        return GradientBoostingClassifier(n_estimators=120, max_depth=2,
                                          learning_rate=0.03, subsample=0.7,
                                          min_samples_leaf=80, random_state=42)

    auc_logit = purged_cv_score(logit, Xis_s, yis)
    # GBM on raw (trees don't need scaling)
    auc_gbm = purged_cv_score(gbm, Xis.values, yis)
    print(f"purged-CV AUC  logit={auc_logit:.3f}  gbm={auc_gbm:.3f}")

    use_gbm = auc_gbm > auc_logit
    print(f"-> using {'GBM' if use_gbm else 'logistic'}")
    if use_gbm:
        model = gbm().fit(Xis.values, yis)
        pall = pd.Series(model.predict_proba(Xall.values)[:, 1], index=data.index)
    else:
        model = logit().fit(Xis_s, yis)
        pall = pd.Series(model.predict_proba(scaler.transform(Xall.values))[:, 1], index=data.index)

    # choose overlay params on IS only
    prod_aln = prod.reindex(data.index).fillna(0)
    bil_aln = bil.reindex(data.index).fillna(0)

    def apply_overlay(p, alpha, p0, floor):
        mult = (1 - alpha * (p - p0)).clip(floor, 1.0)
        mult = mult.shift(1).fillna(1.0)   # act next day
        idle = (1 - mult).clip(lower=0)
        return prod_aln * mult + idle * bil_aln, mult

    best = None
    for alpha in [1.0, 2.0, 3.0, 4.0]:
        for p0 in [0.4, 0.5, 0.6]:
            for floor in [0.0, 0.25, 0.5]:
                r, mult = apply_overlay(pall, alpha, p0, floor)
                m_is = metrics(r.loc[:IS_END])
                if m_is and (best is None or m_is["sr"] > best[0]):
                    best = (m_is["sr"], alpha, p0, floor)
    _, alpha, p0, floor = best
    print(f"chosen overlay (by IS Sharpe): alpha={alpha} p0={p0} floor={floor}")

    r_ml, mult = apply_overlay(pall, alpha, p0, floor)

    print("\n=== results ===")
    base_oos, base_is = metrics(prod_aln.loc[OOS:]), metrics(prod_aln.loc[:IS_END])
    ml_oos, ml_is = metrics(r_ml.loc[OOS:]), metrics(r_ml.loc[:IS_END])
    print(f"  PHOENIX   IS: SR={base_is['sr']} MDD={base_is['mdd']*100:.1f}%  | "
          f"OOS: SR={base_oos['sr']} CAGR={base_oos['cagr']*100:.1f}% MDD={base_oos['mdd']*100:.1f}%")
    print(f"  +metalabel IS: SR={ml_is['sr']} MDD={ml_is['mdd']*100:.1f}%  | "
          f"OOS: SR={ml_oos['sr']} CAGR={ml_oos['cagr']*100:.1f}% MDD={ml_oos['mdd']*100:.1f}%")
    print(f"  avg exposure mult OOS: {mult.loc[OOS:].mean():.3f}")
    dom = (ml_oos["cagr"] > base_oos["cagr"] and ml_oos["mdd"] > base_oos["mdd"]
           and ml_oos["vol"] <= base_oos["vol"])
    print(f"  strict OOS dominance vs production: {dom}")

    # feature importance / coefficients
    if use_gbm:
        imp = dict(sorted(zip(feat_cols, model.feature_importances_.round(3)),
                          key=lambda x: -x[1]))
    else:
        imp = dict(sorted(zip(feat_cols, model.coef_[0].round(3)), key=lambda x: -abs(x[1])))

    out = {
        "model": "gbm" if use_gbm else "logistic",
        "purged_cv_auc": {"logit": round(auc_logit, 4), "gbm": round(auc_gbm, 4)},
        "overlay": {"alpha": alpha, "p0": p0, "floor": floor},
        "production": {"is": base_is, "oos": base_oos},
        "metalabel": {"is": ml_is, "oos": ml_oos},
        "avg_mult_oos": round(float(mult.loc[OOS:].mean()), 3),
        "strict_oos_dominance": bool(dom),
        "feature_importance": imp,
    }
    (OUT / "metalabel_metrics.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"net_ret": r_ml, "mult": mult, "p_bad": pall}).dropna().to_csv(
        OUT / "metalabel_returns.csv")
    print(f"\nSaved metalabel_returns.csv, metalabel_metrics.json in {OUT}")


if __name__ == "__main__":
    main()
