"""ROC lab iter-6: regime-conditional exposure (distinct family).

Switch/scale the best price signals by a CAUSAL market-regime state, to see if regime-timing
lifts OOS. Regime = cross-sectional trend agreement (fraction of eligible coins above their
50d MA) and index realized vol — both observable at t-1. Hypotheses:
  - run trend/breakout only when trend-agreement is high (trending regime),
  - cut gross when index vol is extreme (chop / deleveraging).
Compared honestly to the STATIC best book. Net 4.5bps+funding, walk-forward OOS, deflated.

Run from crypto_pulse/:  python roc_lab5.py  (-> research/roc_lab5.md + png)
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
N_TRIALS = 34
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def sh(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANN) if len(p) > 20 and p.std() > 0 else np.nan


def cagr(p):
    p = p.dropna()
    return (1 + p).prod() ** (ANN / len(p)) - 1 if len(p) > 30 else np.nan


def maxdd(p):
    cum = (1 + p.dropna()).cumprod()
    return (cum / cum.cummax() - 1).min()


def vt(p, t=TGT, win=45):
    return p * (t / (p.rolling(win).std() * np.sqrt(ANN))).shift(1).clip(0, 3)


def deflated_sharpe(p, n_trials):
    p = pd.to_numeric(p, errors="coerce").astype(float).dropna()
    if len(p) < 60:
        return np.nan, np.nan
    sr = p.mean() / p.std(); T = len(p)
    g3 = sps.skew(p); g4 = sps.kurtosis(p, fisher=False)
    e_max = (1 - np.euler_gamma) * sps.norm.ppf(1 - 1.0 / n_trials) + \
        np.euler_gamma * sps.norm.ppf(1 - 1.0 / (n_trials * np.e))
    var_sr = (1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2) / (T - 1)
    z = (sr - e_max * np.sqrt(var_sr)) / np.sqrt(max(var_sr, 1e-12))
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

    # best static signal from iter-5: 20d breakout (most robust, OOS 1.38)
    breakout = dmf(((C - C.rolling(20).mean()) / (C.rolling(20).std() + 1e-9)).where(el))
    w_base = nm(breakout / sd)

    def run(wmat, hold=3):
        rebw = pd.Series(np.arange(len(C)) % hold == 0, index=C.index)
        w = wmat.where(rebw, axis=0).ffill(limit=hold); wl = w.shift(1)
        return vt((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER - (wl * F).sum(axis=1))

    static = run(w_base)

    # ---- causal regime states (observable at t-1) ----
    above50 = (C > C.rolling(50).mean()).where(el)
    agree = above50.mean(axis=1)                                  # fraction trending up [0,1]
    idx_ret = R.where(el).mean(axis=1)                            # equal-weight index return
    idx_vol = idx_ret.rolling(20).std() * np.sqrt(ANN)
    trend_strength = (agree - 0.5).abs() * 2                      # 0=balanced/chop, 1=one-sided

    # regime overlays (all scalings shifted -> causal)
    g_trend = (trend_strength > trend_strength.rolling(120, min_periods=40).median()).astype(float)
    g_vol = (idx_vol < idx_vol.rolling(120, min_periods=40).median()).astype(float)  # calm only
    g_cont = trend_strength.clip(0, 1)                            # continuous trend tilt

    books = {
        "STATIC breakout (baseline)": static,
        "regime: trend-on only": run(w_base.mul(g_trend.shift(1), axis=0)),
        "regime: calm-vol only": run(w_base.mul(g_vol.shift(1), axis=0)),
        "regime: continuous trend tilt": run(w_base.mul(g_cont.shift(1), axis=0)),
    }

    idx = C.index; hl = idx >= HL_START; hidx = idx[hl]
    cut = hidx[int(len(hidx) * 0.6)]
    def io(p):
        q = p[p.index >= HL_START]; return sh(q[q.index < cut]), sh(q[q.index >= cut])

    L_ = ["# ROC lab iter-6: regime-conditional exposure (honest)\n",
          "Scale the best static signal (20d breakout) by causal regime states (trend-agreement, "
          "index vol). Does regime-timing beat the static book? HL era, OOS=last40%.\n",
          "| book | Sharpe (HL) | IS | OOS | CAGR | maxDD |", "|---|---|---|---|---|---|"]
    for k, p in books.items():
        i, o = io(p)
        L_.append(f"| {k} | {sh(p[hl]):+.2f} | {i:+.2f} | {o:+.2f} | {cagr(p[hl]):+.0%} | {maxdd(p[hl]):+.0%} |")

    base_o = io(static)[1]
    best = max(books.items(), key=lambda kp: io(kp[1])[1])
    bo = io(best[1])[1]
    dsr_ann, dsr_p = deflated_sharpe(best[1][hl][best[1].index[hl] >= cut], N_TRIALS)
    helped = bo > base_o + 0.1 and best[0] != "STATIC breakout (baseline)"
    L_ += ["\n## Honest verdict (iteration 6)\n",
           f"- Static baseline OOS {base_o:+.2f}. Best regime variant: **{best[0]}** OOS {bo:+.2f}, "
           f"deflated ({N_TRIALS} trials) {dsr_ann:+.2f}, P={dsr_p:.2f}.",
           f"- Regime-timing **{'beat' if helped else 'did NOT beat'}** the static book — "
           f"{'a real lift' if helped else 'consistent with tactical_overlay: timing adds noise, not edge'}.",
           f"- Sharpe 3 {'REACHED' if bo >= 3 and dsr_p > 0.95 else 'NOT reached'}. Iteration 6; "
           "price ceiling holds at ~1.0-1.85 deflated across 6 methods and 3 repos.\n"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    cols = ["#888", "#2980b9", "#27ae60", "#c0392b"]
    for (k, p), col in zip(books.items(), cols):
        (1 + p[hl].fillna(0)).cumprod().plot(ax=ax, color=col, lw=2.0 if "STATIC" in k else 1.4,
            label=f"{k} (OOS {io(p)[1]:+.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.set_yscale("log")
    ax.legend(fontsize=8); ax.set_title("Regime-conditional vs static breakout (HL era, net)")
    ax.set_ylabel("growth of $1 (log)"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "roc_lab5.png"), dpi=110)
    with open(os.path.join(HERE, "roc_lab5.md"), "w") as fh:
        fh.write("\n".join(L_))
    print("\n".join(L_)); print("\n[written] research/roc_lab5.md + png")
    return bo, dsr_ann, dsr_p


if __name__ == "__main__":
    main()
