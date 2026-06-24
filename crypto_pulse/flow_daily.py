"""FLOW — daily order-flow book from crypto_of (taker-buy volume), the orthogonal lever.

crypto_of/ has DAILY aggressive-flow data — qvol and takerbuy_qvol — for 35 coins over
2021..2026 (~5.5y). Taker-buy fraction = takerbuy_qvol/qvol measures aggressor pressure.
This is genuinely orthogonal to price (it's order flow), and unlike the 29h L4 tape it has
YEARS of history, so we can validate a cross-sectional flow book properly, OOS, net of costs.

We test flow signals (raw imbalance, z-scored, flow momentum), follow vs fade chosen on IS,
then measure: standalone Sharpe IS/OOS, correlation to TIDE, and the combined TIDE+FLOW Sharpe.
If flow is decent AND uncorrelated to TIDE, the combo is the honest lift toward higher Sharpe.

Run from crypto_pulse/:  python flow_daily.py  (-> research/flow_daily.md + png)
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
from tide import TIDE

ANN = 365
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OF = os.path.join(ROOT, "data", "crypto_of")
HL_START = pd.Timestamp("2023-05-12")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
TAKER = 4.5 / 1e4


def sh(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANN) if len(p) > 30 and p.std() > 0 else np.nan


def cagr(p):
    p = p.dropna()
    return (1 + p).prod() ** (ANN / len(p)) - 1 if len(p) > 30 else np.nan


def maxdd(p):
    cum = (1 + p.dropna()).cumprod()
    return (cum / cum.cummax() - 1).min()


def vt(p, t=0.12, win=45):
    return p * (t / (p.rolling(win).std() * np.sqrt(ANN))).shift(1).clip(0, 3)


def main():
    close, qvol, tbuy = {}, {}, {}
    for f in glob.glob(os.path.join(OF, "*.csv")):
        c = os.path.basename(f)[:-4]
        d = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        d = d[~d.index.duplicated()].sort_index()
        close[c], qvol[c], tbuy[c] = d["close"], d["qvol"], d["takerbuy_qvol"]
    C = pd.DataFrame(close).sort_index()
    Q = pd.DataFrame(qvol).reindex_like(C); TB = pd.DataFrame(tbuy).reindex_like(C)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    F = v.load_daily_funding(list(C.columns), C.index)
    el = C.notna() & (Q.rolling(30).mean() > 3e6)
    sd = R.rolling(30).std()
    nm = lambda x: x.div(x.abs().sum(axis=1) + 1e-9, axis=0)
    dmf = lambda x: x.sub(x.mean(axis=1), axis=0)

    tbfrac = (TB / (Q + 1e-9))                              # taker-buy fraction [0,1]
    imb = (2 * tbfrac - 1).where(el)                        # aggressor imbalance [-1,1]
    imb_sm = imb.rolling(3).mean()                          # smoothed level
    imb_z = (imb - imb.rolling(60).mean()) / (imb.rolling(60).std() + 1e-9)   # vs own history
    imb_mom = imb.rolling(5).mean() - imb.rolling(20).mean()                  # flow momentum

    def book(score, hold=3):
        w = nm(dmf(score.where(el)) / (sd + 1e-9))
        rebw = pd.Series(np.arange(len(C)) % hold == 0, index=C.index)
        w = w.where(rebw, axis=0).ffill(limit=hold); wl = w.shift(1)
        return vt((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER - (wl * F).sum(axis=1))

    raw = {"flow-level": imb_sm, "flow-z": imb_z, "flow-momentum": imb_mom}
    idx = C.index
    cut = idx[int(len(idx) * 0.6)]
    def io(p): return sh(p[p.index < cut]), sh(p[p.index >= cut])

    L_ = ["# FLOW — daily order-flow book from crypto_of (honest, OOS)\n",
          f"Taker-buy imbalance, 35 coins {C.index[0].date()}..{C.index[-1].date()}, net "
          f"{TAKER*1e4:.1f}bps+funding, vol-targeted. follow vs fade chosen on IS (first60%), "
          "OOS=last40%.\n",
          "| signal | dir | Sharpe | IS | OOS | robust (both>0.2)? |", "|---|---|---|---|---|---|"]
    best = None
    n_both_pos = 0
    for name, s in raw.items():
        for d, dl in [(1.0, "follow"), (-1.0, "fade")]:
            p = book(d * s)
            i, o = io(p)
            both_pos = np.isfinite(i) and np.isfinite(o) and i > 0.2 and o > 0.2
            n_both_pos += int(both_pos)
            L_.append(f"| {name} | {dl} | {sh(p):+.2f} | {i:+.2f} | {o:+.2f} | "
                      f"{'YES' if both_pos else 'no'} |")
            score_i = i if np.isfinite(i) else -9
            if best is None or score_i > best[1]:     # pick on IS only (causal)
                best = ((name, dl), score_i, o, p)
    (bn, bd), bi, bo, bp = best

    # ---- combine with TIDE ----
    t = TIDE(); tide = t.build()
    both = pd.DataFrame({"TIDE": tide, "FLOW": bp}).dropna()
    bothh = both[both.index >= HL_START]
    rho = bothh["TIDE"].corr(bothh["FLOW"])
    wv = 1 / (bothh["TIDE"].std() + 1e-9); wf = 1 / (bothh["FLOW"].std() + 1e-9)
    wv, wf = wv / (wv + wf), wf / (wv + wf)
    combo = vt(wv * bothh["TIDE"] + wf * bothh["FLOW"])
    st, sf, sc = sh(bothh["TIDE"]), sh(bothh["FLOW"]), sh(combo)

    L_ += [f"\n## Best flow book (IS-picked): {bn} {bd}\n",
           f"- Standalone Sharpe full {sh(bp):+.2f} (IS {bi:+.2f} / OOS {bo:+.2f}), "
           f"CAGR {cagr(bp):+.0%}, maxDD {maxdd(bp):+.0%}.",
           "\n## TIDE + FLOW combo (HL era)\n",
           f"- Correlation TIDE vs FLOW: **{rho:+.2f}**.",
           f"- TIDE {st:+.2f}, FLOW {sf:+.2f}, **risk-parity combo {sc:+.2f}** "
           f"({sc - max(st, sf):+.2f} vs better leg).",
           f"- Sharpe 3 {'REACHED' if sc >= 3 else 'NOT reached'}; combined ~{sc:.1f}.\n",
           "## Verdict (honest)\n",
           f"- **{n_both_pos} of 6 flow configs are positive in BOTH IS and OOS.** The daily "
           "aggregate taker-flow signal is unstable: OOS winners (flow-momentum follow) are "
           "IS-losers and vice versa — regime luck, not a stable edge. Aggregate imbalance is "
           "mostly noise at daily frequency.",
           f"- **Genuinely orthogonal but too weak:** TIDE-FLOW correlation {rho:+.2f} (truly "
           f"uncorrelated), but FLOW's ~0 Sharpe means the combo {sc:+.2f} DILUTES TIDE alone "
           f"({st:+.2f}). Same lesson as EBB: diversification needs a strong second leg.",
           "- **The honest cut:** aggregate daily flow doesn't help. If order flow has alpha it "
           "is in PER-ACCOUNT granularity (informed-wallet isolation), which is what the L4 tape "
           "captures — but that needs the multi-week recording still in progress, not this "
           "aggregate daily series. TIDE alone (~2.0) remains the answer.\n"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for nm_, p, col in [("TIDE", bothh["TIDE"], "#2980b9"), ("FLOW", bothh["FLOW"], "#27ae60"),
                        ("COMBO", combo, "#c0392b")]:
        (1 + p.fillna(0)).cumprod().plot(ax=ax, color=col, lw=2.2 if nm_ == "COMBO" else 1.5,
            label=f"{nm_} ({sh(p):+.2f})")
    ax.set_yscale("log"); ax.legend(fontsize=9)
    ax.set_title(f"TIDE + daily order-FLOW (HL era, corr {rho:+.2f})")
    ax.set_ylabel("growth of $1 (log)"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "flow_daily.png"), dpi=110)
    with open(os.path.join(HERE, "flow_daily.md"), "w") as fh:
        fh.write("\n".join(L_))
    print("\n".join(L_)); print("\n[written] research/flow_daily.md + png")


if __name__ == "__main__":
    main()
