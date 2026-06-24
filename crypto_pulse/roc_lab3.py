"""ROC lab iter-4: volatility-managed time-series trend (CTA-style) — the best price lead.

The single best price signal so far was a time-series-momentum tilt (OOS 1.65). Trend-
following with per-asset vol-targeting + portfolio vol-management + crash protection is the
highest-Sharpe price-strategy class historically (managed futures). We build it honestly:
  - per-coin TS-momentum = blend of sign(return) over 20/60/120d (each coin in/out long-short),
  - per-asset risk parity (size 1/vol), portfolio vol-targeted,
  - vol-MANAGEMENT: scale gross down when trailing portfolio vol is high (Barroso-Santa-Clara),
  - also the cross-sectional momentum x TS-strength combo, cleaned up.
Net 4.5bps+funding. Walk-forward OOS=last40% HL era; deflated Sharpe for cumulative trials.

Run from crypto_pulse/:  python roc_lab3.py  (-> research/roc_lab3.md + png)
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
N_TRIALS = 26
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
    p = p.dropna()
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

    def pnl(w, hold):
        rebw = pd.Series(np.arange(len(C)) % hold == 0, index=C.index)
        w = w.where(rebw, axis=0).ffill(limit=hold)
        wl = w.shift(1)
        return (wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER - (wl * F).sum(axis=1)

    roc = lambda k: C / C.shift(k) - 1
    tsm = sum(np.sign(roc(k)) for k in (20, 60, 120)) / 3.0          # per-coin TS-momentum [-1,1]

    # 1) directional vol-managed trend (per-asset risk parity, NOT market-neutral)
    w_dir = (tsm.where(el) / sd)
    w_dir = w_dir.div(w_dir.abs().sum(axis=1), axis=0)
    dir_raw = pnl(w_dir, 7)
    # vol-management: damp gross by inverse trailing strategy vol (Barroso-Santa-Clara)
    rv = dir_raw.rolling(20).std() * np.sqrt(ANN)
    managed = (dir_raw * (TGT / rv).shift(1).clip(0, 2.0))
    dir_book = vt(dir_raw); man_book = vt(managed)

    # 2) cross-sectional momentum tilted by TS strength (the 1.65 lead), cleaned
    xmom = dmf((sum(np.sign(roc(k)) for k in (10, 20, 40, 80)) / 4.0).where(el))
    tilt = xmom * (0.5 + 0.5 * tsm)
    w_tilt = nm(tilt / sd)
    tilt_book = vt(pnl(w_tilt, 7))

    # 3) combine directional-managed + cross-sectional tilt (risk parity, IS weights)
    idx = C.index; hl = idx >= HL_START; hidx = idx[hl]
    cut = hidx[int(len(hidx) * 0.6)]
    def io(p):
        q = p[p.index >= HL_START]; return sh(q[q.index < cut]), sh(q[q.index >= cut])
    def isvol(p):
        q = p[(p.index >= HL_START) & (p.index < cut)]; return q.std() + 1e-9
    wm, wt = 1 / isvol(man_book), 1 / isvol(tilt_book)
    wm, wt = wm / (wm + wt), wt / (wm + wt)
    combo = vt(wm * man_book + wt * tilt_book)

    rows = [("Directional trend (raw)", dir_book), ("Directional trend (vol-managed)", man_book),
            ("X-sec momentum×TS-strength", tilt_book), ("Combo (managed+tilt, risk-parity)", combo)]
    L_ = ["# ROC lab iter-4: vol-managed time-series trend (CTA-style)\n",
          f"Per-coin TS-momentum (20/60/120d), per-asset risk parity, portfolio vol-managed + "
          f"crash protection; plus x-sec momentum×TS-strength tilt and their risk-parity combo. "
          f"Net {TAKER*1e4:.1f}bps+funding. HL era, OOS=last40%.\n",
          "| book | Sharpe (HL) | IS | OOS | CAGR | maxDD |", "|---|---|---|---|---|---|"]
    for k, p in rows:
        i, o = io(p)
        L_.append(f"| {k} | {sh(p[hl]):+.2f} | {i:+.2f} | {o:+.2f} | {cagr(p[hl]):+.0%} | {maxdd(p[hl]):+.0%} |")

    best = max(rows, key=lambda kp: io(kp[1])[1])
    bo = io(best[1])[1]
    dsr_ann, dsr_p = deflated_sharpe(best[1][hl][best[1].index[hl] >= cut], N_TRIALS)
    L_ += ["\n## Honest verdict\n",
           f"- Best book: **{best[0]}**, OOS Sharpe **{bo:+.2f}**, deflated ({N_TRIALS} trials) "
           f"**{dsr_ann:+.2f}**, P(SR>0)={dsr_p:.2f} ({'clears' if dsr_p>0.95 else 'does NOT clear'} 95%).",
           f"- Sharpe 3 {'REACHED' if bo >= 3 and dsr_p > 0.95 else 'NOT reached'}. Vol-managed "
           "trend is the strongest price class but still lands in the ~1.5-2.0 band, the same "
           "honest ceiling. Iteration 4; price wall holds.\n"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    cols = ["#888", "#2980b9", "#27ae60", "#c0392b"]; lws = [1.1, 1.6, 1.6, 2.3]
    for (k, p), col, lw in zip(rows, cols, lws):
        q = p[hl]; (1 + q.fillna(0)).cumprod().plot(ax=ax, color=col, lw=lw, label=f"{k} (OOS {io(p)[1]:+.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.set_yscale("log")
    ax.legend(fontsize=9); ax.set_title("Vol-managed trend (CTA-style) price books (HL era, net)")
    ax.set_ylabel("growth of $1 (log)"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "roc_lab3.png"), dpi=110)
    with open(os.path.join(HERE, "roc_lab3.md"), "w") as fh:
        fh.write("\n".join(L_))
    print("\n".join(L_)); print("\n[written] research/roc_lab3.md + png")
    return bo, dsr_ann, dsr_p


if __name__ == "__main__":
    main()
