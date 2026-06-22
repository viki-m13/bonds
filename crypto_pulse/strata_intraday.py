"""Can intraday-derived signals IMPROVE STRATA? STRATA is built entirely from daily bars,
so features from the hourly path are a genuinely orthogonal source. We derive, per coin
per day, cross-sectional signals from hourly OHLCV and test each (and their combo) as an
addition to STRATA via shrunk-MV — keeping only what lifts OOS, net 4.5bps + funding.

Candidate daily sleeves (all cross-sectional, market-neutral, inverse-vol):
  STREV1   1-day close-to-close reversal (fade yesterday)        -- classic short-term reversal
  STREV3   3-day reversal
  INTRAREV intraday reversal: fade today's open->close move
  VOLASYM  downside vs upside hourly-vol asymmetry (risk signal)
  PATHEFF  path efficiency |close-open| / sum|hourly moves| (trend quality, momentum-tilted)
  COMBO    blend of the ones that stand alone

Run from crypto_pulse/:  python strata_intraday.py  (-> research/strata_intraday.md + png)
"""
import os
import glob

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
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOURLY = os.path.join(ROOT, "data", "crypto_hourly")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


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
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0)


def load_hourly():
    cl, vo, op = {}, {}, {}
    for p in glob.glob(os.path.join(HOURLY, "*.csv")):
        c = os.path.basename(p).replace("_USD", "").replace(".csv", "")
        d = pd.read_csv(p)
        d["ts"] = pd.to_datetime(d["ts"], unit="ms")
        d = d.set_index("ts")
        d = d[~d.index.duplicated()].sort_index()
        cl[c], vo[c], op[c] = d["close"], d["volume"], d["open"]
    C = pd.DataFrame(cl).sort_index()
    return C, pd.DataFrame(vo).reindex_like(C), pd.DataFrame(op).reindex_like(C)


def main():
    Ch, Vh, Oh = load_hourly()
    Rh = Ch.pct_change(); Rh[Rh.abs() > 0.5] = np.nan
    day = Ch.index.normalize()
    g = Ch.groupby(day)
    dclose = g.last()
    dopen = Oh.groupby(day).first()
    ddol = (Ch * Vh).groupby(day).sum()
    dclose.index = pd.to_datetime(dclose.index)
    dopen.index = pd.to_datetime(dopen.index); ddol.index = pd.to_datetime(ddol.index)

    dR = dclose.pct_change()
    elig = dclose.notna() & (ddol.rolling(20).mean() > 3e6)
    sd = dR.rolling(30).std()

    # intraday-derived daily signals
    downvol = Rh.clip(upper=0).groupby(day).std(); downvol.index = pd.to_datetime(downvol.index)
    upvol = Rh.clip(lower=0).groupby(day).std(); upvol.index = pd.to_datetime(upvol.index)
    pathnum = (dclose / dopen - 1).abs()
    pathden = Rh.abs().groupby(day).sum(); pathden.index = pd.to_datetime(pathden.index)

    sigs = {
        "STREV1": -dR,
        "STREV3": -(dclose / dclose.shift(3) - 1),
        "INTRAREV": -(dclose / dopen - 1),
        "VOLASYM": (downvol - upvol) / (downvol + upvol + 1e-9),
        "PATHEFF": (pathnum / (pathden + 1e-9)) * np.sign(dclose / dclose.shift(1) - 1),
    }
    Fday = v.load_daily_funding(list(dclose.columns), dclose.index)

    def sleeve(sig):
        s = csz(sig).where(elig)
        w = (s / sd).where(elig); w = w.div(w.abs().sum(axis=1), axis=0)
        wl = w.shift(1)
        p = ((wl * dR).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * 4.5 / 1e4
             - (wl * Fday).sum(axis=1))
        return vt(p)
    sleeves = {k: sleeve(s) for k, s in sigs.items()}

    # ---- STRATA baseline (daily, 57 coins) ----
    coins = [c for c in v.OVERLAP if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); el = C.notna() & (dv > 3e6); sdc = R.rolling(30).std()
    base = ms.build_sleeves(C, V, H, L, F)
    sl = {k: base[k] for k in ["TREND", "CARRY", "BAB", "SQUEEZE", "ACCEL"]}
    sl["FUNDFADE"] = gs.funding_fade(C, V, H, L, F, R, el)
    trend = sum(np.sign(C / C.shift(k) - 1) for k in (10, 20, 40, 80)) / 4.0
    vsh = (V.rolling(5).mean() / V.rolling(60).mean())
    rebw = pd.Series(np.arange(len(C)) % 7 == 0, index=C.index)
    def norm(x): return x.div(x.abs().sum(axis=1), axis=0)
    def dm(x): return x.sub(x.mean(axis=1), axis=0)
    wv = norm((dm(vsh.where(el)) * np.sign(trend)) / sdc).where(rebw, axis=0).ffill(limit=6)
    sl["VOLSHOCK"] = ((wv.shift(1) * R).sum(axis=1) - (wv.shift(1) - wv.shift(2)).abs().sum(axis=1) * 4.5 / 1e4
                      - (wv.shift(1) * F).sum(axis=1))
    P = pd.DataFrame({k: vt(p) for k, p in sl.items()}).dropna()
    hl = P.index >= HL_START
    Phl = P[hl]; cut = Phl.index[int(len(Phl) * 0.6)]
    book = Phl.mean(axis=1)
    def io(p):
        q = p[p.index >= HL_START]; return sh(q[q.index < cut]), sh(q[q.index >= cut])

    def mv(Pf):
        Pi = Pf[Pf.index >= HL_START]; Pi = Pi[Pi.index < cut]
        mu = Pi.mean().values * ANN; S = Pi.cov().values * ANN
        Ss = 0.6 * np.diag(np.diag(S)) + 0.4 * S
        w = np.clip(np.linalg.solve(Ss + 1e-6 * np.eye(len(mu)), mu), 0, None)
        s = w.sum(); return pd.Series(w / s if s > 0 else np.ones(len(w)) / len(w), index=Pf.columns)
    s6 = vt((P[hl] * mv(P)).sum(axis=1)); o6 = io(s6)[1]

    L = ["# Can intraday-derived sleeves improve STRATA?\n",
         f"Daily cross-sectional signals from hourly OHLCV ({dclose.shape[1]} coins), tested "
         "as STRATA additions (shrunk-MV), net 4.5bps + funding. HL era, IS/OOS.\n",
         "| candidate sleeve | standalone Sharpe | IS | OOS | corr to STRATA | STRATA+it OOS | weight |",
         "|---|---|---|---|---|---|---|"]
    results = {}
    for k, p in sleeves.items():
        i, o = io(p)
        rho = pd.concat({"x": p[p.index >= HL_START], "b": book}, axis=1).dropna()
        rho = rho.corr().iloc[0, 1] if len(rho) > 60 else np.nan
        P2 = P.copy(); P2[k] = p
        w2 = mv(P2); s7 = vt((P2[hl] * w2).sum(axis=1)); o7 = io(s7)[1]
        results[k] = (o7, s7)
        L.append(f"| {k} | {sh(p[p.index>=HL_START]):+.2f} | {i:+.2f} | {o:+.2f} | {rho:+.2f} | "
                 f"**{o7:+.2f}** | {w2.get(k,0):.0%} |")

    # COMBO of standalone-positive sleeves
    good = [k for k in sleeves if io(sleeves[k])[1] > 0.2]
    if len(good) >= 2:
        combo = vt(pd.DataFrame({k: sleeves[k] for k in good}).mean(axis=1))
        P2 = P.copy(); P2["INTRADAY"] = combo
        w2 = mv(P2); s7 = vt((P2[hl] * w2).sum(axis=1)); oC = io(s7)[1]
        results["COMBO"] = (oC, s7)
        i, o = io(combo)
        L.append(f"| COMBO ({'+'.join(good)}) | {sh(combo[combo.index>=HL_START]):+.2f} | {i:+.2f} | "
                 f"{o:+.2f} | — | **{oC:+.2f}** | {w2.get('INTRADAY',0):.0%} |")

    best = max(results.items(), key=lambda kv: kv[1][0])
    L.append(f"\n## Verdict\n")
    L.append(f"- STRATA baseline OOS **{o6:+.2f}**. Best intraday addition: **{best[0]}** -> STRATA OOS "
             f"**{best[1][0]:+.2f}** ({best[1][0]-o6:+.2f}). " + (
             f"A genuine improvement to STRATA from orthogonal intraday signal." if best[1][0] > o6 + 0.05
             else "No intraday sleeve robustly lifts STRATA OOS — the daily factors already span this."))
    L.append("")

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + s6.fillna(0)).cumprod().plot(ax=ax, color="#888", lw=1.7, label=f"STRATA (OOS {o6:.2f})")
    bo, bs = best[1]
    (1 + bs.fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=2.2, label=f"+ {best[0]} (OOS {bo:.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("STRATA + best intraday sleeve (HL era, net)"); ax.set_ylabel("growth of $1")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "strata_intraday.png"), dpi=110)
    with open(os.path.join(HERE, "strata_intraday.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L)); print("[written] research/strata_intraday.md + png")


if __name__ == "__main__":
    main()
