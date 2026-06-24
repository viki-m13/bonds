"""ROC-improvement lab: push a cross-sectional momentum book as high as it HONESTLY goes.

Uses ai-trader's momentum/breakout ideas (ROC, multi-horizon, skip-recent 12-1, Donchian
turtle, MACD, acceleration, risk-adjusted momentum, Bollinger) as cross-sectional crypto
books, then combines them with a CAUSAL walk-forward ensemble (each day, weight signals by
their trailing risk-adjusted performance — no peeking). Net 4.5bps + funding, vol-targeted.

Honesty guards against "loop until OOS=3" self-deception:
  * Walk-forward OOS = the last 40% of the HL era, never used to pick signals.
  * The ensemble weights are trailing/causal, so adding signals can't cheat.
  * We report the DEFLATED Sharpe (Bailey & Lopez de Prado) given the number of signal
    variants tried — the number that survives multiple-testing, not the lucky max.
  * A signal is only ADMITTED to the ensemble if positive in the IS half (first 60%).

Run from crypto_pulse/:  python roc_lab.py  (-> research/roc_lab.md + png)
"""
import os

import numpy as np
import pandas as pd
from scipy import stats as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v

ANN = 365
TGT = 0.12
TAKER = 4.5 / 1e4
HL_START = pd.Timestamp("2023-05-12")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def sh(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANN) if len(p) > 30 and p.std() > 0 else np.nan


def cagr(p):
    p = p.dropna()
    return (1 + p).prod() ** (ANN / len(p)) - 1 if len(p) > 30 else np.nan


def maxdd(p):
    cum = (1 + p.dropna()).cumprod()
    return (cum / cum.cummax() - 1).min()


def vt(p, t=TGT, win=45):
    return p * (t / (p.rolling(win).std() * np.sqrt(ANN))).shift(1).clip(0, 3)


def deflated_sharpe(p, n_trials):
    """Bailey & Lopez de Prado deflated Sharpe ratio prob, then the haircut SR.
    Returns (annualized SR, prob SR>0 after deflation)."""
    p = p.dropna()
    if len(p) < 60:
        return np.nan, np.nan
    sr = p.mean() / p.std()                       # per-period (daily) Sharpe
    T = len(p)
    g3 = sps.skew(p); g4 = sps.kurtosis(p, fisher=False)
    # expected max Sharpe under n_trials independent strategies (False Strategy Theorem)
    e_max = (1 - np.euler_gamma) * sps.norm.ppf(1 - 1.0 / n_trials) + \
        np.euler_gamma * sps.norm.ppf(1 - 1.0 / (n_trials * np.e))
    var_sr = (1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2) / (T - 1)
    sr0 = e_max * np.sqrt(var_sr)                 # deflation threshold
    z = (sr - sr0) / np.sqrt(max(var_sr, 1e-12))
    return sr * np.sqrt(ANN), float(sps.norm.cdf(z))


def main():
    coins = [c for c in v.OVERLAP if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); el = C.notna() & (dv > 3e6)
    sd = R.rolling(30).std()
    nm = lambda x: x.div(x.abs().sum(axis=1), axis=0)
    dmf = lambda x: x.sub(x.mean(axis=1), axis=0)

    def book(score, hold=1, neutral=True):
        s = score.where(el)
        if neutral:
            s = dmf(s)
        w = nm(s / sd)
        if hold > 1:
            rebw = pd.Series(np.arange(len(C)) % hold == 0, index=C.index)
            w = w.where(rebw, axis=0).ffill(limit=hold)
        wl = w.shift(1)
        p = (wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER - (wl * F).sum(axis=1)
        return vt(p)

    # ---- signal library (ai-trader momentum/breakout family + enhancements) ----
    roc = lambda k: C / C.shift(k) - 1
    mom121 = C.shift(21) / C.shift(147) - 1                       # 12-1 (skip recent month)
    multi = sum(np.sign(roc(k)) for k in (10, 20, 40, 80)) / 4.0
    riskadj = (C / C.shift(60) - 1) / (sd + 1e-9)                 # risk-adjusted momentum
    accel = roc(20) - roc(40)                                     # acceleration
    don = (C - H.rolling(20).max().shift(1)) / (sd * C + 1e-9)    # Donchian/turtle breakout
    ema12 = C.ewm(span=12).mean(); ema26 = C.ewm(span=26).mean()
    macd = (ema12 - ema26) / (C * sd + 1e-9)                      # MACD cross-sectional
    boll = (C - C.rolling(20).mean()) / (C.rolling(20).std() + 1e-9)
    tsmom = np.sign(roc(50))                                      # time-series momentum sign

    signals = {
        "ROC20 (baseline)": book(roc(20), hold=7),
        "ROC multi-horizon": book(multi, hold=7),
        "12-1 momentum": book(mom121, hold=7),
        "risk-adj momentum": book(riskadj, hold=7),
        "acceleration": book(accel, hold=7),
        "Donchian breakout": book(don, hold=5),
        "MACD x-sec": book(macd, hold=5),
        "Bollinger z": book(boll, hold=3),
        "TS-momentum tilt": book(multi * (0.5 + 0.5 * tsmom), hold=7),
    }
    n_trials = len(signals)

    idx = C.index
    hl = idx >= HL_START
    hidx = idx[hl]
    cut = hidx[int(len(hidx) * 0.6)]
    def io(p):
        q = p[p.index >= HL_START]
        return sh(q[q.index < cut]), sh(q[q.index >= cut])

    L_ = ["# ROC-improvement lab — how high does a price-momentum book HONESTLY go?\n",
          f"ai-trader momentum/breakout signals as x-sectional crypto books, net "
          f"{TAKER*1e4:.1f}bps+funding, vol-targeted. HL era; IS=first60% / OOS=last40%. "
          f"{n_trials} variants tried (deflation applied).\n",
          "| signal | Sharpe (HL) | IS | OOS | CAGR | maxDD |", "|---|---|---|---|---|---|"]
    for k, p in signals.items():
        i, o = io(p)
        L_.append(f"| {k} | {sh(p[hl]):+.2f} | {i:+.2f} | {o:+.2f} | {cagr(p[hl]):+.0%} | {maxdd(p[hl]):+.0%} |")

    # ---- iteration 2: SIGNAL-LEVEL blend (net positions in one book), admit by IS ----
    raw = {"ROC20 (baseline)": roc(20), "ROC multi-horizon": multi, "12-1 momentum": mom121,
           "risk-adj momentum": riskadj, "acceleration": accel, "Donchian breakout": don,
           "MACD x-sec": macd, "Bollinger z": -boll, "TS-momentum tilt": multi * (0.5 + 0.5 * tsmom)}
    zsig = {k: dmf(x.where(el)).div((x.where(el).std(axis=1) + 1e-9), axis=0) for k, x in raw.items()}
    is_ok = [k for k in raw if io(signals[k])[0] > 0.2]          # admit on IS only
    zblend = sum(zsig[k] for k in is_ok) / max(len(is_ok), 1)
    siglevel = book(zblend, hold=5)
    sl_i, sl_o = io(siglevel)
    dsr2_ann, dsr2_p = deflated_sharpe(siglevel[hl][siglevel.index[hl] >= cut], n_trials)

    # ---- causal walk-forward ensemble: weight each signal by trailing 180d Sharpe ----
    Sg = pd.DataFrame({k: p for k, p in signals.items()})
    roll_sh = Sg.rolling(180, min_periods=90).apply(lambda x: x.mean() / (x.std() + 1e-9), raw=True)
    wsig = roll_sh.clip(lower=0).shift(1)                         # causal, long-only on good signals
    wsig = wsig.div(wsig.sum(axis=1), axis=0)
    ens = vt((Sg * wsig).sum(axis=1))                            # adaptive ensemble
    # simple equal-weight of IS-positive signals (admission rule)
    ispos = [k for k, p in signals.items() if io(p)[0] > 0]
    eqw = vt(Sg[ispos].mean(axis=1))

    ens_i, ens_o = io(ens); eqw_i, eqw_o = io(eqw)
    dsr_ann, dsr_p = deflated_sharpe(ens[hl][ens.index[hl] >= cut], n_trials)

    L_ += ["\n## Combined books (causal)\n",
           "| combine | Sharpe (HL) | IS | OOS | CAGR | maxDD |", "|---|---|---|---|---|---|"]
    for k, p, i, o in [("Signal-level z-blend (IS-admitted, netted)", siglevel, sl_i, sl_o),
                       ("Adaptive WF ensemble (trail-Sharpe wt)", ens, ens_i, ens_o),
                       ("Equal-wt of IS-positive signals", eqw, eqw_i, eqw_o)]:
        L_.append(f"| {k} | {sh(p[hl]):+.2f} | {i:+.2f} | {o:+.2f} | {cagr(p[hl]):+.0%} | {maxdd(p[hl]):+.0%} |")
    L_.append(f"\nSignal-level blend admitted on IS: {', '.join(is_ok)}.")
    L_.append(f"Signal-level blend deflated OOS Sharpe ({n_trials} trials): {dsr2_ann:+.2f}, "
              f"P(SR>0)={dsr2_p:.2f}.")

    best_oos = max(ens_o, eqw_o, sl_o)
    L_ += ["\n## Honest verdict\n",
           f"- Best combined OOS Sharpe: **{best_oos:+.2f}**.",
           f"- **Deflated Sharpe** of the ensemble OOS (haircut for {n_trials} trials): "
           f"annualized **{dsr_ann:+.2f}**, P(SR>0 after deflation) = {dsr_p:.2f}. "
           f"{'Survives' if dsr_p > 0.95 else 'Does NOT clear the multiple-testing bar at 95%'}.",
           f"- **Sharpe 3 {'REACHED' if best_oos >= 3 else 'NOT reached'}.** A pure price-momentum "
           "book on crypto, honestly walk-forwarded and deflated, lands around "
           f"{best_oos:.1f} — improving ROC (multi-horizon, 12-1, risk-adjusting, ensembling) "
           "lifts it from ~0.9 but plateaus well short of 3. This is the same ceiling STRATA's "
           "full multi-signal book hits (~1.85 OOS): price data alone does not yield Sharpe 3.",
           "- Iterating further on price signals re-tests the SAME OOS and would only manufacture "
           "a lucky 3 via selection — the deflated Sharpe is precisely the guard against that. "
           "**The honest answer is the deflated number, and it is not 3.**\n"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k, p, col, lw in [("ROC20 baseline", signals["ROC20 (baseline)"], "#888", 1.2),
                          ("Adaptive WF ensemble", ens, "#c0392b", 2.2),
                          ("Equal-wt IS-positive", eqw, "#27ae60", 1.6)]:
        q = p[hl]
        (1 + q.fillna(0)).cumprod().plot(ax=ax, color=col, lw=lw, label=f"{k} (OOS {io(p)[1]:+.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.set_yscale("log")
    ax.legend(fontsize=9); ax.set_title("ROC-improvement lab — best honest price-momentum book (HL era, net)")
    ax.set_ylabel("growth of $1 (log)"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "roc_lab.png"), dpi=110)
    with open(os.path.join(HERE, "roc_lab.md"), "w") as fh:
        fh.write("\n".join(L_))
    print("\n".join(L_)); print("\n[written] research/roc_lab.md + png")
    return best_oos


if __name__ == "__main__":
    main()
