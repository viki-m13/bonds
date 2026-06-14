"""Event study: which OHLCV signals precede parabolic 6-month single-stock runs,
measured honestly with an in-sample / out-of-sample split.

Outcome at day d: fwd6 = close[d+126]/close[d]-1 ; "parabolic" = fwd6 > +50%.
For every feature we report, separately for IS (signal date <= 2015-12-31) and
OOS (>= 2016-01-01):
  * mean cross-sectional rank-IC vs fwd6 (Spearman), and its t-stat across dates
  * top-decile vs bottom-decile P(parabolic) and the top-decile lift over base
  * top-decile mean excess fwd6 (vs same-date cross-sectional mean)
A signal is only trusted if its sign and lift PERSIST out of sample.

Causality: features are trailing (built in features.py); fwd6 looks forward only
for scoring the outcome, never enters a feature. Member mask applied throughout.

Run from parabolic/:  python eventstudy.py        (writes research/eventstudy.md)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "dca"))
import data as dca_data  # noqa: E402
import features as feat  # noqa: E402

H6 = 126
PARAB = 0.50
SPLIT = pd.Timestamp("2016-01-01")
SAMPLE_EVERY = 21          # monthly sampling for decile/lift
IC_EVERY = 10              # biweekly for IC series
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research",
                   "eventstudy.md")

# higher value of feature should mean "more likely parabolic"; flip the ones
# where low = bullish so every IC is comparable in sign.
FLIP = {"vcp_contraction", "bbw_pctile", "vol_dryup", "tightness_20",
        "dist_52w_high", "fip", "corr_120", "age_52w_high"}


def sampled_dates(index, every):
    last_ok = len(index) - H6 - 1
    return index[np.arange(260, last_ok, every)]


def main():
    P = dca_data.build_panel()
    close, member = P["close"], P["member"]
    F = feat.build_features(P)
    for k in FLIP:
        if k in F:
            F[k] = -F[k]
    fwd6 = close.shift(-H6) / close - 1
    valid = member & close.notna()
    idx = close.index

    names = sorted(F)
    samp = sampled_dates(idx, SAMPLE_EVERY)
    samp_ic = sampled_dates(idx, IC_EVERY)

    # ---- base rate, by split
    def base_rate(dates):
        ps, n = [], 0
        for d in dates:
            v = valid.loc[d].values & ~np.isnan(fwd6.loc[d].values)
            if v.sum() < 100:
                continue
            ps.append((fwd6.loc[d].values[v] > PARAB).mean())
            n += v.sum()
        return np.mean(ps), n
    base_is, n_is = base_rate(samp[samp < SPLIT])
    base_oos, n_oos = base_rate(samp[samp >= SPLIT])

    # ---- IC series (per date Spearman), split later
    ic = pd.DataFrame(index=samp_ic, columns=names, dtype=float)
    for d in samp_ic:
        vrow = valid.loc[d].values
        yraw = fwd6.loc[d].values
        ymask0 = vrow & ~np.isnan(yraw)
        if ymask0.sum() < 100:
            continue
        yr_full = pd.Series(yraw)
        for nm in names:
            fv = F[nm].loc[d].values
            m = ymask0 & ~np.isnan(fv)
            if m.sum() < 100:
                continue
            fr = pd.Series(fv[m]).rank().values
            yr = yr_full[m].rank().values
            ic.loc[d, nm] = np.corrcoef(fr, yr)[0, 1]

    # ---- decile lift / excess, by split
    def decile_stats(dates):
        topP = {n: [] for n in names}
        botP = {n: [] for n in names}
        topX = {n: [] for n in names}
        for d in dates:
            v0 = valid.loc[d].values & ~np.isnan(fwd6.loc[d].values)
            if v0.sum() < 100:
                continue
            y = fwd6.loc[d].values
            ymean = y[v0].mean()
            for nm in names:
                fv = F[nm].loc[d].values
                m = v0 & ~np.isnan(fv)
                if m.sum() < 100:
                    continue
                pr = pd.Series(fv[m]).rank(pct=True).values
                yy = y[m]
                top = pr >= 0.9
                bot = pr <= 0.1
                if top.sum():
                    topP[nm].append((yy[top] > PARAB).mean())
                    topX[nm].append((yy[top] - ymean).mean())
                if bot.sum():
                    botP[nm].append((yy[bot] > PARAB).mean())
        return ({n: np.mean(topP[n]) if topP[n] else np.nan for n in names},
                {n: np.mean(botP[n]) if botP[n] else np.nan for n in names},
                {n: np.mean(topX[n]) if topX[n] else np.nan for n in names})

    tP_is, bP_is, tX_is = decile_stats(samp[samp < SPLIT])
    tP_oos, bP_oos, tX_oos = decile_stats(samp[samp >= SPLIT])

    ic_is = ic[ic.index < SPLIT]
    ic_oos = ic[ic.index >= SPLIT]

    rows = []
    for nm in names:
        s_is, s_oos = ic_is[nm].dropna(), ic_oos[nm].dropna()
        t_oos = (s_oos.mean() / s_oos.std() * np.sqrt(len(s_oos))
                 if len(s_oos) > 5 and s_oos.std() > 0 else np.nan)
        rows.append({
            "feature": nm,
            "IC_is": s_is.mean(), "IC_oos": s_oos.mean(), "t_oos": t_oos,
            "topP_is": tP_is[nm], "topP_oos": tP_oos[nm],
            "lift_oos": (tP_oos[nm] / base_oos) if base_oos else np.nan,
            "topX_is": tX_is[nm], "topX_oos": tX_oos[nm],
        })
    tab = pd.DataFrame(rows).set_index("feature")
    tab = tab.sort_values("IC_oos", ascending=False)

    # ---- write markdown
    lines = []
    lines.append("# Event study: OHLCV precursors of parabolic 6m runs (IS/OOS)\n")
    lines.append(f"Outcome: fwd6 = close[d+126]/close[d]-1; parabolic = fwd6 > "
                 f"+{PARAB:.0%}. Member-masked PIT S&P 500 panel "
                 f"{close.shape[0]}d x {close.shape[1]} tickers. "
                 f"IS = signal date < {SPLIT.date()}, OOS >= {SPLIT.date()}. "
                 "Features sign-aligned so higher = more bullish (flipped: "
                 + ", ".join(sorted(FLIP)) + ").\n")
    lines.append(f"Base rate P(parabolic): IS {base_is:.2%} "
                 f"(n={n_is:,}), OOS {base_oos:.2%} (n={n_oos:,}).\n")
    lines.append("`IC_*` = mean per-date Spearman rank-IC of the feature vs "
                 "fwd6. `topP_*` = P(parabolic) in the feature's top cross-"
                 "sectional decile. `lift_oos` = topP_oos / base_oos. `topX_*` "
                 "= top-decile mean fwd6 minus same-date cross-sectional mean "
                 "(market-neutral excess). `t_oos` = t-stat of the OOS IC "
                 "series.\n")
    hdr = ("| feature | IC_is | IC_oos | t_oos | topP_is | topP_oos | lift_oos "
           "| topX_is | topX_oos |")
    lines.append(hdr)
    lines.append("|" + "---|" * 9)
    for nm, r in tab.iterrows():
        lines.append(
            f"| {nm} | {r.IC_is:+.3f} | {r.IC_oos:+.3f} | {r.t_oos:+.1f} | "
            f"{r.topP_is:.1%} | {r.topP_oos:.1%} | {r.lift_oos:.1f}x | "
            f"{r.topX_is:+.2%} | {r.topX_oos:+.2%} |")
    lines.append("")

    # robust shortlist: OOS IC same sign as IS, OOS IC>0, OOS lift>=1.3, |t|>=1.5
    robust = tab[(tab.IC_oos > 0) & (np.sign(tab.IC_is) == np.sign(tab.IC_oos))
                 & (tab.lift_oos >= 1.3) & (tab.t_oos.abs() >= 1.5)]
    lines.append("## Robust shortlist (OOS IC>0, sign-stable IS->OOS, "
                 "OOS lift>=1.3x, |t_oos|>=1.5)\n")
    lines.append(", ".join(robust.index) if len(robust) else "(none)")
    lines.append("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[written] {OUT}")
    ic.to_csv(OUT.replace(".md", "_ic.csv"))


if __name__ == "__main__":
    main()
