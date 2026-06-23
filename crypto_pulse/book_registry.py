"""Registry + equity curves of every candidate book toward the Sharpe-3 stack.

Pulls the books we have validated honestly and draws them on one chart (HL era, all
vol-targeted, net of costs) so the state of the search is visible at a glance:
  DEPLOYED/READY : VOL, STRATA, and the VOL+STRATA 50/50 stack (Sharpe ~2.4 full).
  CANDIDATE      : ai-trader ROC cross-sectional momentum (OOS ~1.3, but corr 0.50 to
                   STRATA -> diversifies little; kept as a documented candidate).
  DATA-GATED     : L4 whale-flow (sign-positive, slow, net-negative at fast horizons on
                   29h; revisit at multi-week history). Shown in flow_intraday.png, not here
                   (it is intraday, not on the daily axis).

Run from crypto_pulse/:  python book_registry.py  (-> research/book_registry.md + png)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
import strata_beats_vol as sbv
import aitrader_books as ab

ANN = 365
TGT = 0.12
HL_START = pd.Timestamp("2023-05-12")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def main():
    # ---- core books ----
    vd = pd.read_csv(os.path.join(ROOT, "data", "vol_strategy", "t5rvt_net_daily_2018_2026.csv"),
                     index_col=0)
    vd.index = pd.to_datetime(vd.index)
    vol = vt(vd.iloc[:, 0])
    strata = vt(sbv.build_strata())

    # ---- ai-trader ROC cross-sectional momentum candidate (rebuild standalone) ----
    coins = [c for c in v.OVERLAP if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); el = C.notna() & (dv > 3e6)
    roc = vt(ab.book_from_score(C / C.shift(20) - 1, C, R, F, el, hold=7))

    books = {"VOL": vol, "STRATA": strata, "ai-ROC momentum": roc}
    df = pd.concat(books, axis=1, sort=True).dropna()
    df = df[df.index >= HL_START]
    stack = vt(0.5 * df["VOL"] + 0.5 * df["STRATA"])

    rows = [("VOL", df["VOL"], "deployed"),
            ("STRATA", df["STRATA"], "ready"),
            ("VOL+STRATA 50/50", stack, "**recommended stack**"),
            ("ai-ROC momentum", df["ai-ROC momentum"], "candidate (corr 0.50 to STRATA)")]

    Lm = ["# Book registry — the Sharpe-3 stack, honestly (HL era, net, vol-targeted)\n",
          "| book | status | Sharpe | CAGR | maxDD | corr→stack |", "|---|---|---|---|---|---|"]
    for name, p, status in rows:
        c = pd.concat({"a": p, "s": stack}, axis=1).dropna()
        rho = c["a"].corr(c["s"]) if len(c) > 60 else np.nan
        Lm.append(f"| {name} | {status} | {sh(p):+.2f} | {cagr(p):+.0%} | {maxdd(p):+.0%} | {rho:+.2f} |")

    n_more = 2
    Lm += ["\n## Distance to Sharpe 3\n",
           f"- Current recommended stack (VOL+STRATA 50/50): **Sharpe {sh(stack):.2f}** full HL era.",
           "- To reach 3.0 we need ~2 more books at ~Sharpe 1.9, corr <0.2. The ai-ROC candidate "
           "is positive but corr 0.50 to STRATA (it IS momentum), so it adds little — kept as a "
           "documented fallback, not a stack member.",
           "- The only genuinely-orthogonal source in progress is the **L4 whale-flow book** "
           "(see flow_intraday.md / .png): sign-positive and slow, currently net-negative at "
           "tradeable speed on 29h of tape — revisit at multi-week history.",
           f"- Honest status: **{sh(stack):.2f} now; the path to 3 is {n_more} more orthogonal "
           "books, and the leading candidate is data-gated, not ready.**\n"]

    # ---- equity curves ----
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = {"VOL": "#2980b9", "STRATA": "#27ae60", "VOL+STRATA 50/50": "#c0392b",
               "ai-ROC momentum": "#8e44ad"}
    for name, p, _ in rows:
        (1 + p.fillna(0)).cumprod().plot(ax=ax, lw=2.2 if "50/50" in name else 1.5,
            color=palette[name], label=f"{name} (Sh {sh(p):.2f}, CAGR {cagr(p):+.0%})")
    ax.set_yscale("log"); ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Candidate books toward Sharpe 3 — HL era, net of costs, vol-targeted to 12%")
    ax.set_ylabel("growth of $1 (log)"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "book_registry.png"), dpi=120)
    with open(os.path.join(HERE, "book_registry.md"), "w") as fh:
        fh.write("\n".join(Lm))
    print("\n".join(Lm)); print("\n[written] research/book_registry.md + png")


if __name__ == "__main__":
    main()
