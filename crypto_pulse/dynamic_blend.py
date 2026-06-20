"""Dynamic STRATA/VOL allocator — tilt toward whichever book is leading, vs static 50/50.

STRATA and VOL trade leadership (VOL led 2023-24, STRATA leads 2025-26). A trailing-
performance tilt should add IF leadership persists; strategy-momentum can also whipsaw,
so we test honestly vs static 50/50. STRATA = 7-sleeve shrunk-MV (v2). Both vol-
targeted 12%. Allocators:
  STATIC 50/50
  DYN-RET   weight prop. to trailing-126d return (clip 20-80%), monthly, lagged
  DYN-SHARPE weight prop. to trailing-126d Sharpe (clip 20-80%), monthly, lagged
  DYN-INVVOL inverse trailing-vol (risk parity)
Run from crypto_pulse/:  python dynamic_blend.py  (-> research/dynamic_blend.md + png)
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
import vol_blend as vb

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def sh(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANN) if (len(p) > 40 and p.std() > 0) else np.nan


def stats(p):
    p = p.dropna()
    if len(p) < 60:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=sh(p), cagr=cum.iloc[-1] ** (ANN / len(p)) - 1,
                maxdd=(cum / cum.cummax() - 1).min())


def vt(p, t=0.12):
    return p * (t / (p.rolling(45).std() * np.sqrt(ANN))).shift(1).clip(0, 3)


def strata_v2():
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)
    sd = R.rolling(30).std(); n = len(C)
    rebw = pd.Series(np.arange(n) % 7 == 0, index=C.index)
    def norm(w): return w.div(w.abs().sum(axis=1), axis=0)
    def dm(x): return x.sub(x.mean(axis=1), axis=0)
    def pnl(w, hold=None):
        if hold is not None:
            w = w.where(hold, axis=0).ffill(limit=6)
        wl = w.shift(1)
        return ((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * 4.5 / 1e4
                - (wl * F).sum(axis=1))
    base = ms.build_sleeves(C, V, H, L, F)
    sl = {k: base[k] for k in ["TREND", "CARRY", "BAB", "SQUEEZE", "ACCEL"]}
    sl["FUNDFADE"] = gs.funding_fade(C, V, H, L, F, R, elig)
    trend = sum(np.sign(C / C.shift(k) - 1) for k in (10, 20, 40, 80)) / 4.0
    vsh = (V.rolling(5).mean() / V.rolling(60).mean())
    sl["VOLSHOCK"] = pnl(norm((dm(vsh.where(elig)) * np.sign(trend)) / sd), hold=rebw)
    P = pd.DataFrame({k: vt(p) for k, p in sl.items()})
    hl = P.index >= HL_START
    cut = P.index[hl][int(hl.sum() * 0.6)]
    Pis = P[(P.index >= HL_START) & (P.index < cut)]
    mu = Pis.mean().values * ANN; S = Pis.cov().values * ANN
    Ssh = 0.6 * np.diag(np.diag(S)) + 0.4 * S
    w = np.clip(np.linalg.solve(Ssh + 1e-6 * np.eye(len(mu)), mu), 0, None)
    w = pd.Series(w / w.sum(), index=P.columns)
    return vt((P * w).sum(axis=1).reindex(C.index))


def dyn_weight(a, b, lb, mode, clip=(0.2, 0.8)):
    """monthly, lagged weight on 'a' from trailing-lb metric vs 'b'."""
    if mode == "ret":
        ma, mb = a.rolling(lb).sum(), b.rolling(lb).sum()
    elif mode == "sharpe":
        ma, mb = a.rolling(lb).mean() / a.rolling(lb).std(), b.rolling(lb).mean() / b.rolling(lb).std()
    elif mode == "invvol":
        ma, mb = 1 / a.rolling(lb).std(), 1 / b.rolling(lb).std()
    wa = (ma / (ma.abs() + mb.abs() + 1e-9)).clip(*clip)
    # monthly hold
    wa = wa.where(pd.Series(np.arange(len(wa)) % 21 == 0, index=wa.index)).ffill().shift(1)
    return wa.fillna(0.5)


def main():
    strata = strata_v2()
    vol = vb.vt(vb.load_vol("t5rvt_net_daily_2018_2026.csv"))
    B = pd.concat({"S": strata, "V": vol}, axis=1).dropna()
    B = B[B.index >= HL_START]
    cut = B.index[int(len(B) * 0.6)]

    allocs = {"STATIC 50/50": 0.5 * B["S"] + 0.5 * B["V"]}
    for mode in ("ret", "sharpe", "invvol"):
        wa = dyn_weight(B["S"], B["V"], 126, mode)
        allocs[f"DYN-{mode}"] = wa * B["S"] + (1 - wa) * B["V"]

    def io(p):
        return sh(p[p.index < cut]), sh(p[p.index >= cut])

    lines = ["# Dynamic STRATA/VOL allocator vs static 50/50 (HL era)\n"]
    lines.append("STRATA = 7-sleeve v2. Tilt toward the leading book on trailing 126d "
                 "metric (monthly, lagged, clip 20-80%). IS/OOS.\n")
    lines.append("| allocator | Sharpe | IS | OOS | CAGR | maxDD |")
    lines.append("|---|---|---|---|---|---|")
    best = None
    for nm, p in allocs.items():
        s = stats(p); i, o = io(p)
        lines.append(f"| {nm} | **{s['sharpe']:+.2f}** | {i:+.2f} | {o:+.2f} | "
                     f"{s['cagr']:+.0%} | {s['maxdd']:+.0%} |")
        score = min(i, o)
        if best is None or score > best[0]:
            best = (score, nm, p, s)
    st = stats(allocs["STATIC 50/50"])
    lines.append("\n## Verdict\n")
    lines.append(f"- Best allocator: **{best[1]}** Sharpe {best[3]['sharpe']:+.2f} "
                 f"(min IS,OOS {best[0]:+.2f}) vs static 50/50 {st['sharpe']:+.2f}. "
                 + ("Dynamic tilt ADDS — leadership is persistent enough to time."
                    if best[1] != "STATIC 50/50" and best[3]['sharpe'] > st['sharpe'] + 0.05
                    else "Static 50/50 is as good — leadership doesn't persist reliably "
                    "enough to time without whipsaw; keep it simple.") + "\n")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    (1 + allocs["STATIC 50/50"].fillna(0)).cumprod().plot(ax=ax, color="#888", lw=1.6,
        label=f"static 50/50 ({st['sharpe']:.2f})")
    (1 + best[2].fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=2.2,
        label=f"{best[1]} ({best[3]['sharpe']:.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("Dynamic vs static STRATA/VOL allocation (HL era, net)")
    ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "dynamic_blend.png"), dpi=120)
    with open(os.path.join(HERE, "dynamic_blend.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("[written] research/dynamic_blend.md + png")


if __name__ == "__main__":
    main()
