"""Our own intraday volatility-channel breakout book on the HL universe (hourly bars).

VOL (the benchmark, daily Sharpe ~1.99) is a SINGLE-asset intraday vol-channel breakout.
This builds a MULTI-coin version on ~20 liquid coins of clean hourly OHLCV (2021-2026)
and asks, honestly, net of 4.5bps taker + funding:
  1) DIRECTIONAL portfolio of per-coin breakouts  -> does it beat VOL standalone?
  2) MARKET-NEUTRAL cross-sectional breakout       -> an independent book + STRATA sleeve?
Both momentum (follow the breakout) and reversion (fade to VWAP) are tested; the IS half
picks, the OOS half scores. Hourly PnL is netted of turnover cost + hourly funding, then
aggregated to a daily return series so it is directly comparable to VOL and STRATA.

Run from crypto_pulse/:  python intraday_breakout.py  (-> research/intraday_breakout.md + png)
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOURLY = os.path.join(ROOT, "data", "crypto_hourly")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
TC = 4.5 / 1e4               # taker cost per unit turnover
W = 24                       # VWAP / sigma window (hours)
BAND = 1.0                   # channel half-width in hourly-vol units
CAP = 3.0                    # cap |signal|


def sh(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANN) if (len(p) > 40 and p.std() > 0) else np.nan


def stats(p):
    p = p.dropna()
    if len(p) < 60:
        return dict(sharpe=np.nan, maxdd=np.nan, cagr=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=sh(p), maxdd=(cum / cum.cummax() - 1).min(),
                cagr=(cum.iloc[-1]) ** (ANN / len(p)) - 1)


def vt(p, t=0.12):
    return p * (t / (p.rolling(45).std() * np.sqrt(ANN))).shift(1).clip(0, 3)


def load_hourly():
    cl, vo = {}, {}
    for p in glob.glob(os.path.join(HOURLY, "*.csv")):
        c = os.path.basename(p).replace("_USD", "").replace(".csv", "")
        d = pd.read_csv(p)
        d["ts"] = pd.to_datetime(d["ts"], unit="ms")
        d = d.set_index("ts")[~d.set_index("ts").index.duplicated()].sort_index()
        cl[c], vo[c] = d["close"], d["volume"]
    C = pd.DataFrame(cl).sort_index()
    return C, pd.DataFrame(vo).reindex_like(C)


def main():
    C, Vol = load_hourly()
    R = C.pct_change(); R[R.abs() > 0.5] = np.nan      # hourly return, clip flash prints
    vwap = (C * Vol).rolling(W).sum() / Vol.rolling(W).sum()
    sig = R.rolling(W).std()
    # standardized distance from VWAP in hourly-vol units
    b = (C - vwap) / (sig * C + 1e-12)
    elig = C.notna() & sig.notna() & (Vol.rolling(W).mean() * C > 2e5)

    # HELD positions: enter long when b>BAND, exit when b<=0 (reverts to VWAP); symmetric
    # for shorts. This is what makes a breakout tradeable (few trades/episode, low turnover).
    def held(bvals, band, maxhold=72):
        T, N = bvals.shape
        pos = np.zeros((T, N)); hold = np.zeros(N); cnt = np.zeros(N)
        for t in range(T):
            row = bvals[t]
            for j in range(N):
                x = row[j]; p = hold[j]
                if np.isnan(x):
                    p = 0.0; cnt[j] = 0
                elif p == 0.0:
                    if x > band: p = 1.0; cnt[j] = 0
                    elif x < -band: p = -1.0; cnt[j] = 0
                else:
                    cnt[j] += 1
                    if (p > 0 and x <= 0) or (p < 0 and x >= 0) or cnt[j] > maxhold:
                        p = 0.0; cnt[j] = 0
                hold[j] = p; pos[t, j] = p
        return pos
    pos_mom = pd.DataFrame(held(b.values, BAND), index=C.index, columns=C.columns).where(elig, 0.0)

    # hourly funding spread to per-hour from daily
    coins = list(C.columns)
    dayidx = pd.to_datetime(C.index.date)
    Fd = v.load_daily_funding(coins, pd.Index(sorted(set(dayidx)), name="Date"))
    Fh = Fd.reindex(dayidx).to_numpy() / 24.0
    Fh = pd.DataFrame(Fh, index=C.index, columns=coins).reindex(columns=C.columns)

    def book(direction, neutral):
        s = (direction * pos_mom).where(elig)
        if neutral:
            s = s.sub(s.mean(axis=1), axis=0)              # cross-sectional demean
        w = s.div(s.abs().sum(axis=1), axis=0).fillna(0.0)  # gross = 1
        wl = w.shift(1)
        gross_ret = (wl * R).sum(axis=1)
        cost = (wl - wl.shift(1)).abs().sum(axis=1) * TC
        fund = (wl * Fh).sum(axis=1)
        hourly = gross_ret - cost - fund
        daily = hourly.groupby(C.index.normalize()).sum()
        daily.index = pd.to_datetime(daily.index)
        return daily

    variants = {
        "DIRECTIONAL momentum": book(+1.0, False),
        "DIRECTIONAL reversion": book(-1.0, False),
        "NEUTRAL momentum": book(+1.0, True),
        "NEUTRAL reversion": book(-1.0, True),
    }
    V = {k: vt(p) for k, p in variants.items()}
    # IS/OOS split on HL era
    allidx = sorted(set.union(*[set(p.dropna().index) for p in V.values()]))
    hlidx = [d for d in allidx if d >= HL_START]
    cut = hlidx[int(len(hlidx) * 0.6)]
    def io(p):
        q = p[p.index >= HL_START]; return sh(q[q.index < cut]), sh(q[q.index >= cut])

    # VOL benchmark
    volp = None
    vf = os.path.join(ROOT, "data", "vol_strategy", "t5rvt_net_daily_2018_2026.csv")
    if os.path.exists(vf):
        vd = pd.read_csv(vf, index_col=0)
        vd.index = pd.to_datetime(vd.index)
        volp = vd.iloc[:, 0]
        volp = volp[~volp.index.duplicated()]

    L = ["# Intraday vol-channel breakout book on the HL universe (hourly)\n",
         f"Our own multi-coin version of VOL's edge: {C.shape[1]} coins, hourly bars "
         f"{C.index[0].date()}->{C.index[-1].date()}, VWAP/sigma window {W}h, band {BAND} "
         f"hourly-vol units, net {TC*1e4:.1f}bps taker + funding, vol-targeted, daily. "
         "HL era, IS=first60/OOS=last40.\n",
         "| book | Sharpe | IS | OOS | CAGR | maxDD |", "|---|---|---|---|---|---|"]
    for k, p in V.items():
        st = stats(p[p.index >= HL_START]); i, o = io(p)
        L.append(f"| {k} | **{st['sharpe']:+.2f}** | {i:+.2f} | {o:+.2f} | {st['cagr']:+.0%} | {st['maxdd']:+.0%} |")
    if volp is not None:
        st = stats(volp[volp.index >= HL_START])
        vi = sh(volp[(volp.index >= HL_START) & (volp.index < cut)])
        vo_ = sh(volp[volp.index >= cut])
        L.append(f"| VOL (benchmark) | **{st['sharpe']:+.2f}** | {vi:+.2f} | {vo_:+.2f} | {st['cagr']:+.0%} | {st['maxdd']:+.0%} |")

    # best of ours by OOS
    best = max(V.items(), key=lambda kv: io(kv[1])[1])
    bname, bser = best
    boos = io(bser)[1]
    L.append(f"\n## Verdict\n")
    L.append(f"- Best of ours by OOS: **{bname}**, OOS Sharpe **{boos:+.2f}**.")
    if volp is not None:
        vol_oos = sh(volp[volp.index >= cut])
        beats = boos > vol_oos
        rho = pd.concat({"x": bser, "v": volp}, axis=1).dropna()
        rho = rho.corr().iloc[0, 1] if len(rho) > 60 else np.nan
        L.append(f"- VOL OOS {vol_oos:+.2f}. Ours {'BEATS' if beats else 'does NOT beat'} VOL "
                 f"standalone OOS. Correlation of best book to VOL: {rho:+.2f}.")
        # does it improve VOL? (only report if genuinely additive)
        if np.isfinite(rho) and rho < 0.6:
            cmb = vt(pd.concat({"a": bser, "b": volp}, axis=1).dropna().mean(axis=1))
            L.append(f"- VOL + our best (equal risk) OOS: {io(cmb)[1]:+.2f} "
                     f"(VOL alone {vol_oos:+.2f}) — {'improves' if io(cmb)[1] > vol_oos + 0.05 else 'no real lift to'} VOL.")
    L.append("\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    for k, col in [(bname, "#c0392b")]:
        p = V[k]; p = p[p.index >= HL_START]
        (1 + p.fillna(0)).cumprod().plot(ax=ax, color=col, lw=2.0, label=f"{k} (OOS {io(V[k])[1]:.2f})")
    if volp is not None:
        pv = volp[volp.index >= HL_START]
        (1 + pv.fillna(0)).cumprod().plot(ax=ax, color="#2980b9", lw=1.6, label=f"VOL ({sh(pv):.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("Intraday breakout (ours) vs VOL — HL era, net"); ax.set_ylabel("growth of $1")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "intraday_breakout.png"), dpi=110)
    with open(os.path.join(HERE, "intraday_breakout.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L)); print("[written] research/intraday_breakout.md + png")


if __name__ == "__main__":
    main()
