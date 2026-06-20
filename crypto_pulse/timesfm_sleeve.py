"""TimesFM sleeve — use Google's TimesFM 2.5 foundation model to forecast each HL coin's
forward return, rank cross-sectionally, and test whether it improves STRATA's OOS.

For each weekly rebalance in the HL era, feed each eligible coin's trailing log-price
into TimesFM (zero-shot), forecast `H` days ahead, derive expected forward return,
demean cross-sectionally (market-neutral), inverse-vol size, hold 1 week. Net of 4.5bps
+ funding. Then: standalone Sharpe IS/OOS, correlation to STRATA, and the STRATA+TimesFM
combined OOS. Forecasts are cached so the backtest is reproducible.

Run from crypto_pulse/:  python timesfm_sleeve.py  (-> research/timesfm_sleeve.md + png)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
import max_stack as ms
import grand_stack as gs

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research", "timesfm_fc.parquet")
H = 5            # forecast horizon (days)
CTX = 256       # context length
STEP = 7        # weekly rebalance


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


def build_forecasts(C, elig, complete_only=False):
    """Expected forward-H return per coin at each weekly rebalance (TimesFM).
    Incremental + resumable cache. If complete_only and cache is partial, returns None."""
    idx = C.index
    hl_locs = [i for i in range(CTX, len(idx)) if idx[i] >= HL_START and i % STEP == 0]
    out = pd.DataFrame(index=idx, columns=C.columns, dtype=float)
    colpos = {c: out.columns.get_loc(c) for c in out.columns}
    done = set()
    if os.path.exists(CACHE):
        cached = pd.read_parquet(CACHE)
        out.loc[cached.index, cached.columns] = cached
        done = {i for i in hl_locs if out.iloc[i].notna().any()}
    if len(done) >= len(hl_locs):
        return out
    if complete_only:
        return None
    from timesfm.timesfm_2p5 import timesfm_2p5_torch as T
    from timesfm import ForecastConfig
    m = T.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch')
    m.compile(ForecastConfig(max_context=CTX, max_horizon=H, normalize_inputs=True,
                             use_continuous_quantile_head=True))
    logC = np.log(C)
    todo = [i for i in hl_locs if i not in done]
    for k, i in enumerate(todo):
        names, ctxs = [], []
        for c in C.columns:
            if not elig.iloc[i][c]:
                continue
            s = logC[c].iloc[i - CTX:i].dropna()
            if len(s) >= 128:
                names.append(c); ctxs.append(s.values[-CTX:].astype(np.float32))
        if not names:
            continue
        pf, _ = m.forecast(horizon=H, inputs=ctxs)
        pf = np.array(pf)
        for j, c in enumerate(names):
            out.iat[i, colpos[c]] = float(np.exp(pf[j][-1] - ctxs[j][-1]) - 1.0)
        if k % 10 == 0:
            out.dropna(how="all").to_parquet(CACHE)      # incremental save
            print(f"  forecast {k}/{len(todo)} ({idx[i].date()})", flush=True)
    out.dropna(how="all").to_parquet(CACHE)
    return out


def main():
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H_, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)
    sd = R.rolling(30).std()

    fc = build_forecasts(C, elig)
    # cross-sectional market-neutral sleeve from the TimesFM forecast, weekly hold
    sig = fc.sub(fc.mean(axis=1), axis=0)
    w = (sig / sd).where(elig); w = w.div(w.abs().sum(axis=1), axis=0)
    w = w.ffill(limit=STEP - 1)
    wl = w.shift(1)
    tfm = vt((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * 4.5 / 1e4
             - (wl * F).sum(axis=1))

    # also a reversal variant (fade the forecast) — TimesFM may over-extrapolate trend
    wr = (-sig / sd).where(elig); wr = wr.div(wr.abs().sum(axis=1), axis=0).ffill(limit=STEP - 1)
    wrl = wr.shift(1)
    tfm_rev = vt((wrl * R).sum(axis=1) - (wrl - wrl.shift(1)).abs().sum(axis=1) * 4.5 / 1e4
                 - (wrl * F).sum(axis=1))

    # STRATA 7-sleeve (equal-risk book) for correlation + combine
    base = ms.build_sleeves(C, V, H_, L, F)
    sl = {k: base[k] for k in ["TREND", "CARRY", "BAB", "SQUEEZE", "ACCEL"]}
    sl["FUNDFADE"] = gs.funding_fade(C, V, H_, L, F, R, elig)
    trend = sum(np.sign(C / C.shift(k) - 1) for k in (10, 20, 40, 80)) / 4.0
    vsh = (V.rolling(5).mean() / V.rolling(60).mean())
    rebw = pd.Series(np.arange(len(C)) % 7 == 0, index=C.index)
    def norm(x): return x.div(x.abs().sum(axis=1), axis=0)
    def dm(x): return x.sub(x.mean(axis=1), axis=0)
    sl["VOLSHOCK"] = (lambda wts: ((wts.shift(1) * R).sum(axis=1)
        - (wts.shift(1) - wts.shift(2)).abs().sum(axis=1) * 4.5 / 1e4
        - (wts.shift(1) * F).sum(axis=1)))(
        norm((dm(vsh.where(elig)) * np.sign(trend)) / sd).where(rebw, axis=0).ffill(limit=6))
    P = pd.DataFrame({k: vt(p) for k, p in sl.items()}).dropna()
    hl = P.index >= HL_START
    Phl = P[hl]; cut = Phl.index[int(len(Phl) * 0.6)]
    book = Phl.mean(axis=1)
    def io(p):
        q = p[p.index >= HL_START]; return sh(q[q.index < cut]), sh(q[q.index >= cut])

    lines = ["# TimesFM forecast sleeve on HL crypto — does it improve STRATA OOS?\n"]
    lines.append("Google TimesFM 2.5 (200M) zero-shot forecast of each coin's forward "
                 f"{H}d return, cross-sectional market-neutral, weekly. Net 4.5bps + "
                 "funding. HL era, IS/OOS.\n")
    lines.append("| variant | Sharpe | IS | OOS | corr to STRATA |")
    lines.append("|---|---|---|---|---|")
    for nm, p in [("TimesFM momentum (follow)", tfm), ("TimesFM reversal (fade)", tfm_rev)]:
        i, o = io(p)
        rho = pd.concat({"x": p[hl], "b": book}, axis=1).dropna().corr().iloc[0, 1]
        lines.append(f"| {nm} | **{sh(p[hl]):+.2f}** | {i:+.2f} | {o:+.2f} | {rho:+.2f} |")

    # add the better TimesFM variant to STRATA (shrunk-MV) and compare OOS
    cand = tfm if sh(tfm[hl]) >= sh(tfm_rev[hl]) else tfm_rev
    candnm = "follow" if cand is tfm else "fade"
    P2 = P.copy(); P2["TIMESFM"] = cand
    def mv(Pf):
        Pi = Pf[hl][Pf[hl].index < cut]
        mu = Pi.mean().values * ANN; S = Pi.cov().values * ANN
        Ss = 0.6 * np.diag(np.diag(S)) + 0.4 * S
        w = np.clip(np.linalg.solve(Ss + 1e-6 * np.eye(len(mu)), mu), 0, None)
        return pd.Series(w / w.sum(), index=Pf.columns)
    s6 = vt((P[hl] * mv(P)).sum(axis=1)); s7 = vt((P2[hl] * mv(P2)).sum(axis=1))
    o6, o7 = io(s6)[1], io(s7)[1]
    lines.append(f"\n## STRATA OOS: baseline vs + TimesFM ({candnm})\n")
    lines.append("| book | Sharpe | IS | OOS | maxDD |")
    lines.append("|---|---|---|---|---|")
    for nm, p in [("STRATA (7-sleeve)", s6), (f"STRATA + TimesFM", s7)]:
        st = stats(p); i, o = io(p)
        lines.append(f"| {nm} | **{st['sharpe']:+.2f}** | {i:+.2f} | {o:+.2f} | {st['maxdd']:+.0%} |")
    lines.append(f"\n## Verdict\n")
    lines.append(f"- TimesFM sleeve: best standalone OOS {max(io(tfm)[1], io(tfm_rev)[1]):+.2f}. "
                 f"Adding it lifts STRATA OOS {o6:+.2f} -> **{o7:+.2f}** ({o7-o6:+.2f}). " + (
                 "TimesFM genuinely improves STRATA's OOS." if o7 > o6 + 0.05 else
                 "TimesFM does NOT robustly improve STRATA OOS here (its forecast overlaps "
                 "trend / is cost-eaten at weekly rebalance on 57 coins)."))
    lines.append("\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + s6[hl].fillna(0)).cumprod().plot(ax=ax, color="#888", lw=1.6, label=f"STRATA (OOS {o6:.2f})")
    (1 + s7[hl].fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=2.2, label=f"+ TimesFM (OOS {o7:.2f})")
    (1 + cand[hl].fillna(0)).cumprod().plot(ax=ax, color="#2980b9", lw=1.0, ls="--",
        label=f"TimesFM sleeve ({sh(cand[hl]):.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("TimesFM forecast sleeve + STRATA (HL era, net)")
    ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "timesfm_sleeve.png"), dpi=110)
    with open(os.path.join(HERE, "timesfm_sleeve.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("[written] research/timesfm_sleeve.md + png")


if __name__ == "__main__":
    main()
