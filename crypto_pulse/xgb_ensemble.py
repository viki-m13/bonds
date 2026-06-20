"""XGBoost ensemble (many models, many objectives) to improve STRATA's OOS Sharpe.

Predicts cross-sectional outperformance (forward-5d return, demeaned per day) on the HL
crypto universe from ~16 causal features. Trains MANY XGBoost models — different
objectives (squared / pseudo-huber / absolute / quantile@0.3/0.5/0.7 / logistic / rank)
x depths x seeds — WALK-FORWARD with a 5-day embargo (target can't leak), ensembles by
averaging cross-sectional ranks. The anti-overfit discipline (expanding-window walk-
forward, embargo, ensemble of weak learners) is the whole point. Then: ML-sleeve Sharpe
IS/OOS, correlation to STRATA, and STRATA + ML combined OOS.

Run from crypto_pulse/:  python xgb_ensemble.py  (-> research/xgb_ensemble.md + png)
"""
import os

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
import max_stack as ms
import grand_stack as gs

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
HORIZON = 5
EMBARGO = 6
RETRAIN = 42


def sh(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANN) if (len(p) > 40 and p.std() > 0) else np.nan


def stats(p):
    p = p.dropna()
    if len(p) < 60:
        return dict(sharpe=np.nan, maxdd=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=sh(p), maxdd=(cum / cum.cummax() - 1).min())


def vt(p, t=0.12):
    return p * (t / (p.rolling(45).std() * np.sqrt(ANN))).shift(1).clip(0, 3)


def csz(df):
    """cross-sectional z-score per day."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0)


def main():
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)
    mkt = R.where(elig).mean(axis=1)

    # ---- features (causal, cross-sectionally z-scored) ----
    feats = {}
    for k in (5, 10, 20, 40, 80):
        feats[f"r{k}"] = csz(C / C.shift(k) - 1)
    feats["accel"] = csz((C / C.shift(20) - 1) - (C.shift(20) / C.shift(40) - 1))
    feats["vol20"] = csz(R.rolling(20).std())
    feats["vol60"] = csz(R.rolling(60).std())
    feats["volr"] = csz(V.rolling(5).mean() / V.rolling(60).mean())
    feats["fund"] = csz(F)
    feats["fund5"] = csz(F.rolling(5).mean())
    feats["fundchg"] = csz(F.rolling(5).mean() - F.rolling(20).mean())
    feats["beta"] = csz(R.rolling(60).cov(mkt).div(mkt.rolling(60).var(), axis=0))
    feats["hi20"] = csz(C / C.rolling(20).max())
    feats["clv"] = csz(((C - L) - (H - C)) / (H - L).replace(0, np.nan))
    fnames = list(feats.keys())

    fwd = (C.shift(-HORIZON) / C - 1)
    target = fwd.sub(fwd.mean(axis=1), axis=0)        # cross-sectional fwd return

    # ---- stack to long (date, coin) — VECTORIZED ----
    idx = C.index
    di_map = pd.Series(np.arange(len(idx)), index=idx)
    cpos = {c: i for i, c in enumerate(C.columns)}
    parts = {f: feats[f].stack(dropna=False) for f in fnames}
    df = pd.DataFrame(parts)
    df["y"] = target.stack(dropna=False)
    df["el"] = elig.stack(dropna=False)
    df = df[df["el"].fillna(False).astype(bool)].drop(columns="el")
    df = df.dropna()
    lvl = df.index.get_level_values
    df["di"] = di_map.reindex(lvl(0)).values
    df["ci"] = [cpos[c] for c in lvl(1)]
    df = df[(df["di"] >= 120) & (df["di"] < len(idx) - HORIZON)]
    X = df[fnames].values.astype(np.float32); y = df["y"].values.astype(np.float32)
    dd = df["di"].values; cc = df["ci"].values

    # ---- model zoo ----
    base = dict(max_depth=3, eta=0.05, subsample=0.7, colsample_bytree=0.7,
                min_child_weight=20, reg_lambda=2.0, verbosity=0, nthread=4)
    zoo = []
    for obj in ["reg:squarederror", "reg:pseudohubererror", "reg:absoluteerror",
                "reg:quantileerror"]:
        for depth in (2, 4):
            for seed in (0,):
                p = dict(base, objective=obj, max_depth=depth, seed=seed)
                if obj == "reg:quantileerror":
                    for q in (0.3, 0.5, 0.7):
                        zoo.append(dict(p, quantile_alpha=q))
                else:
                    zoo.append(p)
    # plus logistic on sign
    for seed in (0,):
        zoo.append(dict(base, objective="binary:logistic", seed=seed, max_depth=3))
    print(f"{len(zoo)} models in the zoo", flush=True)

    # ---- walk-forward predict over HL era ----
    pred = pd.DataFrame(np.nan, index=idx, columns=C.columns)
    hl_di = [di for di in np.unique(dd) if idx[di] >= HL_START]
    retrain_pts = hl_di[::RETRAIN]
    for ri, rdi in enumerate(retrain_pts):
        train_mask = dd < (rdi - EMBARGO)
        if train_mask.sum() < 2000:
            continue
        Xtr, ytr = X[train_mask], y[train_mask]
        dtr = xgb.DMatrix(Xtr, label=ytr)
        dtr_bin = xgb.DMatrix(Xtr, label=(ytr > 0).astype(float))
        boosters = []
        for p in zoo:
            d = dtr_bin if p["objective"] == "binary:logistic" else dtr
            boosters.append(xgb.train(p, d, num_boost_round=60))
        # predict the window [rdi, next)
        nxt = retrain_pts[ri + 1] if ri + 1 < len(retrain_pts) else len(idx)
        win = (dd >= rdi) & (dd < nxt)
        if win.sum() == 0:
            continue
        Xw = X[win]; dw = dd[win]; cw = cc[win]
        dmw = xgb.DMatrix(Xw)
        # ensemble by averaging per-day ranks of each model's prediction
        ens = np.zeros(Xw.shape[0])
        for b in boosters:
            pr = b.predict(dmw)
            # rank within each day
            tmp = pd.DataFrame({"d": dw, "p": pr})
            ens += tmp.groupby("d")["p"].rank(pct=True).values
        ens /= len(boosters)
        for r in range(Xw.shape[0]):
            pred.iat[dw[r], cw[r]] = ens[r]
        if ri % 5 == 0:
            print(f"  walk-forward {ri}/{len(retrain_pts)} ({idx[rdi].date()})", flush=True)

    # ---- ML sleeve: cross-sectional, market-neutral, weekly hold ----
    sd = R.rolling(30).std()
    sig = pred.sub(pred.mean(axis=1), axis=0)         # demean the rank-ensemble
    w = (sig / sd).where(elig); w = w.div(w.abs().sum(axis=1), axis=0)
    w = w.ffill(limit=4)
    wl = w.shift(1)
    ml = vt((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * 4.5 / 1e4
            - (wl * F).sum(axis=1))

    # STRATA 7-sleeve book
    bsl = ms.build_sleeves(C, V, H, L, F)
    sl = {k: bsl[k] for k in ["TREND", "CARRY", "BAB", "SQUEEZE", "ACCEL"]}
    sl["FUNDFADE"] = gs.funding_fade(C, V, H, L, F, R, elig)
    trend = sum(np.sign(C / C.shift(k) - 1) for k in (10, 20, 40, 80)) / 4.0
    vsh = (V.rolling(5).mean() / V.rolling(60).mean())
    rebw = pd.Series(np.arange(len(C)) % 7 == 0, index=C.index)
    def norm(x): return x.div(x.abs().sum(axis=1), axis=0)
    def dm(x): return x.sub(x.mean(axis=1), axis=0)
    wv = norm((dm(vsh.where(elig)) * np.sign(trend)) / sd).where(rebw, axis=0).ffill(limit=6)
    sl["VOLSHOCK"] = ((wv.shift(1) * R).sum(axis=1)
                      - (wv.shift(1) - wv.shift(2)).abs().sum(axis=1) * 4.5 / 1e4
                      - (wv.shift(1) * F).sum(axis=1))
    P = pd.DataFrame({k: vt(p) for k, p in sl.items()}).dropna()
    hl = P.index >= HL_START
    Phl = P[hl]; cut = Phl.index[int(len(Phl) * 0.6)]
    book = Phl.mean(axis=1)
    def io(p):
        q = p[p.index >= HL_START]; return sh(q[q.index < cut]), sh(q[q.index >= cut])

    iM, oM = io(ml)
    rho = pd.concat({"x": ml[hl], "b": book}, axis=1).dropna().corr().iloc[0, 1]
    P2 = P.copy(); P2["XGB"] = ml
    def mv(Pf):
        Pi = Pf[hl][Pf[hl].index < cut]
        mu = Pi.mean().values * ANN; S = Pi.cov().values * ANN
        Ss = 0.6 * np.diag(np.diag(S)) + 0.4 * S
        w = np.clip(np.linalg.solve(Ss + 1e-6 * np.eye(len(mu)), mu), 0, None)
        return pd.Series(w / w.sum(), index=Pf.columns)
    s6 = vt((P[hl] * mv(P)).sum(axis=1)); s7 = vt((P2[hl] * mv(P2)).sum(axis=1))
    o6, o7 = io(s6)[1], io(s7)[1]

    lines = ["# XGBoost ensemble sleeve — does it improve STRATA OOS?\n"]
    lines.append(f"{len(zoo)} XGBoost models (objectives x depth x seed), walk-forward "
                 f"(retrain {RETRAIN}d, embargo {EMBARGO}d), rank-ensembled, predicting "
                 f"cross-sectional fwd-{HORIZON}d return. Net 4.5bps + funding. HL era.\n")
    lines.append("| book | Sharpe | IS | OOS | maxDD | corr to STRATA |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| XGB ensemble sleeve | **{sh(ml[hl]):+.2f}** | {iM:+.2f} | {oM:+.2f} "
                 f"| {stats(ml[hl])['maxdd']:+.0%} | {rho:+.2f} |")
    lines.append(f"| STRATA (7-sleeve) | **{stats(s6)['sharpe']:+.2f}** | {io(s6)[0]:+.2f} | {o6:+.2f} | {stats(s6)['maxdd']:+.0%} | — |")
    lines.append(f"| STRATA + XGB | **{stats(s7)['sharpe']:+.2f}** | {io(s7)[0]:+.2f} | {o7:+.2f} | {stats(s7)['maxdd']:+.0%} | — |")
    lines.append(f"\n## Verdict\n")
    lines.append(f"- XGB ensemble sleeve OOS {oM:+.2f}, corr to STRATA {rho:+.2f}. Adding "
                 f"it takes STRATA OOS {o6:+.2f} -> **{o7:+.2f}** ({o7-o6:+.2f}). " + (
                 "The ensemble genuinely improves STRATA's OOS." if o7 > o6 + 0.05 else
                 "It does NOT robustly improve STRATA OOS — the ML ensemble extracts no "
                 "edge beyond the factor sleeves after honest walk-forward + cost (crypto "
                 "cross-sectional returns are near-unpredictable; the features ARE the "
                 "sleeves)."))
    lines.append("\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + s6[hl].fillna(0)).cumprod().plot(ax=ax, color="#888", lw=1.6, label=f"STRATA (OOS {o6:.2f})")
    (1 + s7[hl].fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=2.2, label=f"+ XGB ensemble (OOS {o7:.2f})")
    (1 + ml[hl].fillna(0)).cumprod().plot(ax=ax, color="#2980b9", lw=1.0, ls="--", label=f"XGB sleeve ({sh(ml[hl]):.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("XGBoost ensemble + STRATA (HL era, net, walk-forward)")
    ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "xgb_ensemble.png"), dpi=110)
    with open(os.path.join(HERE, "xgb_ensemble.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("[written] research/xgb_ensemble.md + png")


if __name__ == "__main__":
    main()
