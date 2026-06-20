"""Final equity curves: STRATA v2 vs VOL vs 50/50 BLEND — recent and long term.

STRATA v2 = 7-sleeve shrunk-MV (VOLSHOCK added), ~1.58. For the long view it is
spliced with the price-only proxy before 2023 (funding sleeves need 2023+ data).
VOL = leakage-free published series (1.99). All net, vol-targeted 12%.
Run from crypto_pulse/:  python strata_v2_curves.py
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
import dynamic_blend as db
import vol_blend as vb

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def stats(p):
    p = p.dropna()
    if len(p) < 60:
        return dict(sharpe=np.nan, maxdd=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=p.mean() / p.std() * np.sqrt(ANN),
                maxdd=(cum / cum.cummax() - 1).min())


def main():
    strata_hl = db.strata_v2()                       # 7-sleeve v2, HL era
    vol = vb.vt(vb.load_vol("t5rvt_net_daily_2018_2026.csv"))
    coins = [c for c in sorted(set(bl.ALL111))
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    member = ux.membership(C, V, 30, 30, 0, 30)
    price, _ = pcc.run(C, V, H, L, member, 1e7, maker=False)
    warm = C.index[C.index >= C.index[0] + pd.Timedelta(days=220)][0]
    strata_canon = pd.concat([price[(price.index >= warm) & (price.index < HL_START)],
                              strata_hl[strata_hl.index >= HL_START]]).sort_index()
    strata_canon = strata_canon[~strata_canon.index.duplicated()]

    sS = pd.concat({"S": strata_hl, "V": vol}, axis=1).dropna()
    sS = sS[sS.index >= HL_START]; sBlend = 0.5 * sS["S"] + 0.5 * sS["V"]
    lS = pd.concat({"S": strata_canon, "V": vol}, axis=1).dropna()
    lBlend = 0.5 * lS["S"] + 0.5 * lS["V"]

    OUR, VOLC, BL = "#8e44ad", "#16a085", "#c0392b"
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    def panel(a, d, blend, title, logy):
        for col, color, nm in [("S", OUR, "STRATA v2"), ("V", VOLC, "VOL")]:
            s = stats(d[col])
            (1 + d[col].fillna(0)).cumprod().plot(ax=a, color=color, lw=1.6, logy=logy,
                label=f"{nm}: Sharpe {s['sharpe']:.2f}, DD {s['maxdd']:.0%}")
        s = stats(blend)
        (1 + blend.fillna(0)).cumprod().plot(ax=a, color=BL, lw=2.6, logy=logy,
            label=f"50/50 BLEND: Sharpe {s['sharpe']:.2f}, DD {s['maxdd']:.0%}")
        a.set_title(title, fontsize=11); a.legend(fontsize=9, loc="upper left")
        a.grid(alpha=0.3, which="both")
        a.set_ylabel("growth of $1" + (" (log)" if logy else ""))

    panel(ax[0], sS, sBlend, "RECENT — HL era 2023-2026 (STRATA v2 full book)", False)
    panel(ax[1], lS, lBlend, "LONG — 2018-2026, log (STRATA v2 spliced w/ price proxy)", True)
    ax[1].axvline(HL_START, color="gray", ls=":", lw=1)
    fig.suptitle("STRATA v2 vs VOL vs 50/50 BLEND — net, vol-targeted to 12%", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "strata_v2_curves.png"), dpi=120)
    print("RECENT  STRATA %.2f  VOL %.2f  BLEND %.2f (DD %.0f%%)" % (
        stats(sS["S"])["sharpe"], stats(sS["V"])["sharpe"], stats(sBlend)["sharpe"],
        stats(sBlend)["maxdd"] * 100))
    print("LONG    STRATA %.2f  VOL %.2f  BLEND %.2f (DD %.0f%%)" % (
        stats(lS["S"])["sharpe"], stats(lS["V"])["sharpe"], stats(lBlend)["sharpe"],
        stats(lBlend)["maxdd"] * 100))
    print("[written] research/strata_v2_curves.png")


if __name__ == "__main__":
    main()
