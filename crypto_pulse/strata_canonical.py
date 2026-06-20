"""STRATA — the one canonical equity curve.

Splices the two honest versions into a single picture:
  * BEFORE 2023-05 (HL funding history starts): STRATA price-only (4 sleeves —
    TREND, BAB, SQUEEZE, ACCEL), per-coin realistic cost. The long-history proxy.
  * FROM 2023-05: STRATA full book (6 sleeves — adds CARRY + FUNDFADE funding
    sleeves), the real deployable strategy (~1.5).
The shaded region marks where the funding sleeves switch on. Both segments are
vol-targeted to 12% and net of realistic cost, so the curve is continuous.

Run from crypto_pulse/:  python strata_canonical.py  (-> research/strata_canonical.png + md)
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

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def sh(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANN) if (len(p) > 60 and p.std() > 0) else np.nan


def mdd(p):
    c = (1 + p.dropna()).cumprod()
    return (c / c.cummax() - 1).min()


def main():
    coins = [c for c in sorted(set(bl.ALL111))
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    member = ux.membership(C, V, 30, 30, 0, 30)
    price, _ = pcc.run(C, V, H, L, member, 1e7, maker=False)   # 4 sleeves, $10M per-coin
    grand = kc.build_grandstack()                               # 6 sleeves, HL era

    warm = C.index[C.index >= C.index[0] + pd.Timedelta(days=220)][0]
    pre = price[(price.index >= warm) & (price.index < HL_START)]
    post = grand[grand.index >= HL_START]
    canon = pd.concat([pre, post]).sort_index()
    canon = canon[~canon.index.duplicated()]

    eq = (1 + canon.fillna(0)).cumprod()
    s_pre, s_post, s_all = sh(pre), sh(post), sh(canon)

    fig, ax = plt.subplots(figsize=(13, 6))
    eq.plot(ax=ax, color="#8e44ad", lw=1.8, logy=True)
    ax.axvspan(HL_START, eq.index[-1], color="#2ecc71", alpha=0.07)
    ax.axvline(HL_START, color="gray", ls="--", lw=1)
    # annotate segments
    ax.text(pre.index[len(pre) // 2], eq.max() * 0.9,
            f"price-only proxy\n(4 sleeves)\nSharpe {s_pre:.2f}",
            ha="center", fontsize=10, color="#6c3483")
    ax.text(post.index[len(post) // 2], eq.loc[post.index].min() * 1.05,
            f"FULL book (6 sleeves,\n+ funding sleeves) — deployable\nSharpe {s_post:.2f}",
            ha="center", fontsize=10, color="#1e8449",
            bbox=dict(boxstyle="round", fc="white", ec="#2ecc71", alpha=0.8))
    ax.set_title(f"STRATA — canonical equity curve (net, vol-targeted 12%, log scale)\n"
                 f"spliced: price-only proxy before 2023-05, full 6-sleeve book after | "
                 f"overall Sharpe {s_all:.2f}, maxDD {mdd(canon):.0%}", fontsize=11)
    ax.set_ylabel("growth of $1 (log)"); ax.set_xlabel("")
    ax.grid(alpha=0.3, which="both")
    ax.annotate("funding sleeves switch ON\n(HL funding data starts 2023-05)",
                xy=(HL_START, eq.loc[HL_START] if HL_START in eq.index else eq.iloc[len(pre)]),
                xytext=(0.5, 0.12), textcoords="axes fraction", fontsize=9,
                ha="center", arrowprops=dict(arrowstyle="->", color="gray"))
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "strata_canonical.png"), dpi=120)

    lines = ["# STRATA — canonical equity curve\n"]
    lines.append("One picture, two honest segments spliced at 2023-05 (where HL funding "
                 "data — and thus the CARRY + FUNDFADE sleeves — begin):\n")
    lines.append(f"- **Pre-2023 (price-only proxy, 4 sleeves):** Sharpe {s_pre:.2f}, "
                 f"{pre.index.min().date()}→{pre.index.max().date()}")
    lines.append(f"- **2023+ (FULL 6-sleeve deployable book):** Sharpe {s_post:.2f}, "
                 f"maxDD {mdd(post):+.0%}")
    lines.append(f"- **Spliced overall:** Sharpe {s_all:.2f}, maxDD {mdd(canon):+.0%}, "
                 f"grew $1 → ${eq.iloc[-1]:.1f} over {len(canon)/365:.0f}y\n")
    lines.append("The pre-2023 segment understates STRATA (it lacks the funding "
                 "sleeves, which can't be computed before HL funding history). The "
                 "deployable STRATA is the 2023+ full book (~1.5); the proxy is shown "
                 "only to give a multi-regime view of the price-action core.\n")
    with open(os.path.join(HERE, "strata_canonical.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("[written] research/strata_canonical.png + md")


if __name__ == "__main__":
    main()
