"""Final comparison: STRATA vs VOL vs 50/50 BLEND, shorter-term and longer-term.

STRATA = the canonical book: full 6-sleeve (TREND+CARRY+BAB+SQUEEZE+ACCEL+FUNDFADE)
from 2023-05 (deployable, ~1.5), spliced with the 4-sleeve price-only proxy before
(funding data absent pre-2023). VOL = the vol repo's leakage-free published series
(t5rvt eq_vt35, Sharpe 1.99, 2018-2026). BLEND = 50/50 equal-risk. All net,
vol-targeted to 12%.

  SHORTER (HL era 2023+): STRATA full book vs VOL vs blend.
  LONGER  (2018-2026):    STRATA canonical splice vs VOL vs blend.

Run from crypto_pulse/:  python strata_vs_vol.py  (-> research/strata_vs_vol.png + md)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
import breadth_leverage as bl
import universe_experiments as ux
import per_coin_cost as pcc
import kelly_cagr as kc
import vol_blend as vb

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def stats(p):
    p = p.dropna()
    if len(p) < 60:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=p.mean() / p.std() * np.sqrt(ANN),
                cagr=cum.iloc[-1] ** (ANN / len(p)) - 1,
                maxdd=(cum / cum.cummax() - 1).min())


def main():
    coins = [c for c in sorted(set(bl.ALL111))
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    member = ux.membership(C, V, 30, 30, 0, 30)
    price, _ = pcc.run(C, V, H, L, member, 1e7, maker=False)
    grand = kc.build_grandstack()
    vol = vb.vt(vb.load_vol("t5rvt_net_daily_2018_2026.csv"))

    warm = C.index[C.index >= C.index[0] + pd.Timedelta(days=220)][0]
    # canonical STRATA: price-only before HL, full book after
    strata_canon = pd.concat([price[(price.index >= warm) & (price.index < HL_START)],
                              grand[grand.index >= HL_START]]).sort_index()
    strata_canon = strata_canon[~strata_canon.index.duplicated()]

    # ---- SHORTER: HL era 2023+ ----
    sS = pd.concat({"STRATA": grand, "VOL": vol}, axis=1).dropna()
    sS = sS[sS.index >= HL_START]
    sBlend = 0.5 * sS["STRATA"] + 0.5 * sS["VOL"]

    # ---- LONGER: 2018-2026 ----
    lS = pd.concat({"STRATA": strata_canon, "VOL": vol}, axis=1).dropna()
    lBlend = 0.5 * lS["STRATA"] + 0.5 * lS["VOL"]

    def row(nm, p):
        s = stats(p)
        return f"| {nm} | **{s['sharpe']:+.2f}** | {s['cagr']:+.0%} | {s['maxdd']:+.0%} |"

    lines = ["# STRATA vs VOL vs BLEND — final comparison\n"]
    lines.append("STRATA = canonical book (full 6-sleeve from 2023, price-only proxy "
                 "before). VOL = leakage-free published series (1.99). 50/50 equal-risk, "
                 "net, vol-targeted 12%.\n")
    lines.append("## SHORTER term — HL era 2023+ (STRATA = full 6-sleeve book)\n")
    lines.append("| book | Sharpe | CAGR | maxDD |")
    lines.append("|---|---|---|---|")
    lines += [row("STRATA", sS["STRATA"]), row("VOL", sS["VOL"]),
              row("50/50 BLEND", sBlend)]
    lines.append(f"\ncorr(STRATA,VOL) = {sS['STRATA'].corr(sS['VOL']):+.2f}\n")
    lines.append("## LONGER term — 2018-2026 (STRATA = canonical splice)\n")
    lines.append("| book | Sharpe | CAGR | maxDD |")
    lines.append("|---|---|---|---|")
    lines += [row("STRATA", lS["STRATA"]), row("VOL", lS["VOL"]),
              row("50/50 BLEND", lBlend)]
    lines.append(f"\ncorr(STRATA,VOL) = {lS['STRATA'].corr(lS['VOL']):+.2f}\n")
    lines.append("## Read\n")
    lines.append(f"- BLEND beats either alone on risk-adjusted return AND drawdown: "
                 f"shorter {stats(sBlend)['sharpe']:+.2f} (DD {stats(sBlend)['maxdd']:+.0%}), "
                 f"longer {stats(lBlend)['sharpe']:+.2f} (DD {stats(lBlend)['maxdd']:+.0%}) "
                 f"— vs VOL's standalone {stats(lS['VOL'])['maxdd']:+.0%} DD. STRATA is "
                 "the calm leg (low DD, leads recently); VOL is the higher-Sharpe leg "
                 "(led 2023-24); together they are smoother than either.\n")

    OUR, VOLC, BL = "#8e44ad", "#16a085", "#c0392b"
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    def panel(a, d, blend, title, logy):
        for col, color, nm in [("STRATA", OUR, "STRATA (ours)"),
                               ("VOL", VOLC, "VOL (theirs)")]:
            s = stats(d[col])
            (1 + d[col].fillna(0)).cumprod().plot(ax=a, color=color, lw=1.6, logy=logy,
                label=f"{nm}: Sharpe {s['sharpe']:.2f}, DD {s['maxdd']:.0%}")
        sb = stats(blend)
        (1 + blend.fillna(0)).cumprod().plot(ax=a, color=BL, lw=2.6, logy=logy,
            label=f"50/50 BLEND: Sharpe {sb['sharpe']:.2f}, DD {sb['maxdd']:.0%}")
        a.set_title(title, fontsize=11); a.legend(fontsize=9, loc="upper left")
        a.grid(alpha=0.3, which="both")
        a.set_ylabel("growth of $1" + (" (log)" if logy else ""))

    panel(ax[0], sS, sBlend, "SHORTER — HL era 2023-2026", False)
    panel(ax[1], lS, lBlend, "LONGER — 2018-2026 (log)", True)
    ax[1].axvline(HL_START, color="gray", ls=":", lw=1)
    fig.suptitle("STRATA vs VOL vs 50/50 BLEND — net, vol-targeted to 12%", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "strata_vs_vol.png"), dpi=120)

    with open(os.path.join(HERE, "strata_vs_vol.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("[written] research/strata_vs_vol.png + md")


if __name__ == "__main__":
    main()
