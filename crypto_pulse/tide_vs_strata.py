"""Equity curves + reconcile TIDE with the existing VOL+STRATA handoff (honest).

The other agent already has a handoff to run VOL+STRATA 50/50. The improved TIDE is a new
standalone book. Before updating their docs, answer the questions that actually matter to them:
  1. Is TIDE redundant with STRATA (STRATA has a TREND sleeve)? -> correlation + combo test.
  2. Is TIDE additive to VOL? -> correlation to live VOL proxy.
  3. What does the best book look like (VOL / STRATA / TIDE / combos)?
Produce a clean equity-curve PNG and the correlation matrix for the handoff.

Run from crypto_pulse/:  python tide_vs_strata.py  (-> research/tide_vs_strata.md + png)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tide import TIDE, sh, cagr, maxdd, vt, HL_START, ANN, TAKER
from strata_beats_vol import build_strata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def vt15(p):  # vol-target to 15% to match the VOL+STRATA handoff convention
    return p * (0.15 / (p.rolling(45).std() * np.sqrt(ANN))).shift(1).clip(0, 3)


def main():
    t = TIDE()
    tide = t.build()
    # phase-2 TIDE + tamed carry (drawdown-floored), risk-parity
    F = t.F.reindex(columns=t.C.columns).fillna(0.0)
    R = t.C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (t.C * t.V).rolling(30).mean(); el = t.C.notna() & (dv > 3e6)
    sd = np.sqrt((np.log(t.H / t.L) ** 2).rolling(30).mean() / (4 * np.log(2))) + 1e-9
    nm = lambda x: x.div(x.abs().sum(axis=1) + 1e-9, axis=0)
    rebw = pd.Series(np.arange(len(t.C)) % 3 == 0, index=t.C.index)
    fsm = F.rolling(14).mean(); car = -(fsm.sub(fsm.mean(axis=1), axis=0)).where(el)
    wc = nm(car / sd).where(rebw, axis=0).ffill(limit=3); wlc = wc.shift(1)
    carry = vt((wlc * R).sum(axis=1) - (wlc - wlc.shift(1)).abs().sum(axis=1) * TAKER - (wlc * F).sum(axis=1))
    cum = (1 + carry.fillna(0)).cumprod(); ddc = cum / cum.cummax() - 1
    carry = carry * (1 + (ddc / 0.10).clip(-1, 0)).shift(1)
    ia = 1.0 / (tide.rolling(45).std() * np.sqrt(ANN)).clip(lower=0.05)
    ib = 1.0 / (carry.rolling(45).std() * np.sqrt(ANN)).clip(lower=0.05)
    wa = (ia / (ia + ib)).shift(1).fillna(0.5)
    tide_carry = vt(tide * wa + carry * (1 - wa))

    strata = build_strata()
    vd = pd.read_csv(os.path.join(ROOT, "data", "vol_strategy", "t5rvt_net_daily_2018_2026.csv"), index_col=0)
    vd.index = pd.to_datetime(vd.index); vol = vd.iloc[:, 0]

    # clean 3-book comparison (each vol-targeted to 15%); phase-2 carry noted separately in text
    books = {"VOL": vt15(vol), "STRATA": vt15(strata), "TIDE": vt15(tide)}
    # common HL-era index
    hl = lambda p: p[p.index >= HL_START]
    df = pd.DataFrame({k: hl(p) for k, p in books.items()}).dropna()

    L = ["# TIDE vs STRATA vs VOL — reconciliation for the VOL operator (honest)\n",
         "All vol-targeted to 15% (the VOL+STRATA handoff convention), net. HL era (common dates).\n",
         "## Standalone books (HL era)\n", "| book | Sharpe | CAGR | maxDD |", "|---|---|---|---|"]
    for k in books:
        p = df[k]; L.append(f"| {k} | {sh(p):+.2f} | {cagr(p):+.0%} | {maxdd(p):+.0%} |")

    cor = df.corr()
    L += ["\n## Correlation (HL era, daily)\n", "| | " + " | ".join(cor.columns) + " |",
          "|" + "---|" * (len(cor.columns) + 1)]
    for r in cor.index:
        L.append(f"| {r} | " + " | ".join(f"{cor.loc[r, c]:+.2f}" for c in cor.columns) + " |")

    # risk-parity combos
    def rp(cols):
        sub = df[cols]
        iv = 1.0 / (sub.rolling(45).std() * np.sqrt(ANN)).clip(lower=0.05)
        w = iv.div(iv.sum(axis=1), axis=0).shift(1).fillna(1.0 / len(cols))
        return vt((sub * w).sum(axis=1))
    combos = {"VOL+STRATA": ["VOL", "STRATA"], "VOL+TIDE": ["VOL", "TIDE"],
              "STRATA+TIDE": ["STRATA", "TIDE"], "VOL+STRATA+TIDE": ["VOL", "STRATA", "TIDE"]}
    L += ["\n## Risk-parity combos (HL era)\n", "| combo | Sharpe | CAGR | maxDD |", "|---|---|---|---|"]
    cres = {}
    for cn, cols in combos.items():
        p = rp(cols); cres[cn] = p
        L.append(f"| {cn} | {sh(p):+.2f} | {cagr(p):+.0%} | {maxdd(p):+.0%} |")

    ts_cor = cor.loc["TIDE", "STRATA"]; tv_cor = cor.loc["TIDE", "VOL"]
    L += ["\n## What this means for the VOL operator\n",
          f"- **TIDE vs STRATA correlation = {ts_cor:+.2f}.** "
          + ("They heavily overlap (STRATA's TREND/CARRY/BAB sleeves include TIDE's momentum core) — "
             "TIDE is best seen as a **cleaner, single-book replacement for STRATA's trend engine**, "
             "not an independent third leg." if ts_cor > 0.5 else
             "They are only partly correlated — TIDE can be a distinct sleeve."),
          f"- **TIDE vs VOL correlation = {tv_cor:+.2f}** — {'low, so TIDE diversifies VOL much like STRATA does' if abs(tv_cor) < 0.35 else 'moderate'}.",
          f"- Best simple book here: **{max(cres, key=lambda k: sh(cres[k]))}** "
          f"(Sharpe {max(sh(p) for p in cres.values()):+.2f}).",
          "- Honest guidance: TIDE and STRATA are the same family (cross-sectional crypto). Run **one**"
          " of them as the market-neutral leg next to VOL — TIDE is the simpler, fully-documented, "
          "higher-capacity choice; STRATA is the 7-sleeve version. Do NOT double-count by running both "
          "at full size. TIDE+carry(phase-2) is the higher-Sharpe but uncertified variant.\n"]

    # equity curve
    fig, ax = plt.subplots(figsize=(11.5, 6))
    colors = {"VOL": "#7f8c8d", "STRATA": "#2980b9", "TIDE": "#c0392b"}
    for k in books:
        (1 + df[k].fillna(0)).cumprod().plot(ax=ax, lw=2.0 if k.startswith("TIDE") else 1.4,
                                             color=colors[k], alpha=0.9, label=f"{k} ({sh(df[k]):.2f})")
    ax.set_yscale("log"); ax.legend(fontsize=10, loc="upper left")
    ax.set_title("HL era, net, vol-targeted to 15% — TIDE vs STRATA vs VOL")
    ax.set_ylabel("growth of $1 (log)"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "tide_vs_strata.png"), dpi=120)

    # full-history TIDE equity (the certified book) — separate clean curve
    fig2, ax2 = plt.subplots(figsize=(11.5, 6))
    tf = vt15(tide).dropna()
    (1 + tf.fillna(0)).cumprod().plot(ax=ax2, color="#c0392b", lw=1.9, label=f"TIDE certified ({sh(tf):.2f})")
    ax2.axvline(HL_START, color="#2980b9", ls="--", lw=1, label="HL era starts")
    ax2.set_yscale("log"); ax2.legend(fontsize=10); ax2.grid(alpha=0.3)
    ax2.set_title("Certified TIDE — full ~12-year equity curve (net, vol-targeted 15%)")
    ax2.set_ylabel("growth of $1 (log)"); fig2.tight_layout()
    fig2.savefig(os.path.join(HERE, "tide_equity_full.png"), dpi=120)

    with open(os.path.join(HERE, "tide_vs_strata.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L)); print("\n[written] research/tide_vs_strata.md + tide_vs_strata.png + tide_equity_full.png")


if __name__ == "__main__":
    main()
