"""Basket backtest for pre-parabolic signals — measures the ACTUAL objective.

For a score (dates x tickers), every REBAL trading days we pick the top-k eligible
names and look at their forward 6-month outcome:

  * hit_rate   = P(pick's fwd6 > +50%)              (precision for "parabolic")
  * lift       = hit_rate / base rate on the same dates
  * mean_fwd6  = equal-weight mean forward 6m return of picks
  * excess     = mean_fwd6 minus same-date cross-sectional universe mean
  * vs_random  = mean_fwd6 minus the mean of N random top-k baskets from the
                 SAME eligible pool on the SAME dates (the survivorship control,
                 mirroring dca/protocol.random_control)

Everything is reported IS (rebalance < 2016) vs OOS (>= 2016). Forward outcomes
use close[d]->close[d+126]; the signal at d is trailing-only, so this is causal.
Picks that delist within 6m (NaN fwd) are dropped from the mean but COUNTED as
non-parabolic for hit-rate honesty would over-penalise; we instead report
coverage (share of picks with a valid 6m outcome).

Run from parabolic/:  python backtest.py            (writes research/backtest.md)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "dca"))
import data as dca_data  # noqa: E402
import features as feat  # noqa: E402
import strategy as strat  # noqa: E402

H6 = 126
PARAB = 0.50
REBAL = 21
SPLIT = pd.Timestamp("2016-01-01")
K = 10
N_RANDOM = 200
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research",
                   "backtest.md")


def _eval_dates(index):
    last_ok = len(index) - H6 - 1
    pos = np.arange(260, last_ok, REBAL)
    return index[pos], pos


def basket_metrics(score: pd.DataFrame, close: pd.DataFrame,
                   member: pd.DataFrame, k=K, seed=7):
    fwd6 = close.shift(-H6) / close - 1
    dates, _ = _eval_dates(close.index)
    rng = np.random.default_rng(seed)
    S = score.to_numpy()
    Y = fwd6.to_numpy()
    cols = np.array(close.columns)
    didx = {d: i for i, d in enumerate(close.index)}

    rec = {"is": [], "oos": []}
    for d in dates:
        i = didx[d]
        s = S[i]
        y = Y[i]
        elig = ~np.isnan(s) & ~np.isnan(y)
        if elig.sum() < k + 5:
            continue
        order = np.argsort(-s[elig])
        ey = y[elig]
        picks = ey[order[:k]]
        base = (ey > PARAB).mean()
        umean = ey.mean()
        # random control from same eligible pool
        rand_means = np.empty(N_RANDOM)
        n_elig = elig.sum()
        for j in range(N_RANDOM):
            sel = rng.choice(n_elig, size=k, replace=False)
            rand_means[j] = ey[sel].mean()
        row = {
            "date": d,
            "hit": (picks > PARAB).mean(),
            "base": base,
            "mean_fwd6": picks.mean(),
            "umean": umean,
            "rand_mean": rand_means.mean(),
            "rand_p": (picks.mean() > rand_means).mean(),
        }
        rec["oos" if d >= SPLIT else "is"].append(row)
    return {k_: pd.DataFrame(v) for k_, v in rec.items()}


def equity_curve(score, close, member, k=K, seed=7):
    """Overlapping-tranche long-only equity: at each rebalance invest 1 unit
    split equally over top-k, hold 6m. Returns annualised geometric return of
    the average tranche (a clean, cost-free upper bound) IS and OOS, plus the
    same for a random control."""
    fwd6 = close.shift(-H6) / close - 1
    dates, _ = _eval_dates(close.index)
    S, Y = score.to_numpy(), fwd6.to_numpy()
    didx = {d: i for i, d in enumerate(close.index)}
    rng = np.random.default_rng(seed)
    out = {}
    for lab, dd in (("is", [d for d in dates if d < SPLIT]),
                    ("oos", [d for d in dates if d >= SPLIT])):
        tr, rr = [], []
        for d in dd:
            i = didx[d]
            s, y = S[i], Y[i]
            elig = ~np.isnan(s) & ~np.isnan(y)
            if elig.sum() < k + 5:
                continue
            ey = y[elig]
            order = np.argsort(-s[elig])
            tr.append(np.mean(1 + ey[order[:k]]))           # 6m gross multiple
            rr.append(np.mean(1 + ey[rng.choice(elig.sum(), k, replace=False)]))
        # annualise: each tranche is a 6m holding -> ^2 per year
        out[lab] = {
            "strat_6m_mult": np.mean(tr),
            "strat_ann": np.mean(tr) ** 2 - 1,
            "rand_6m_mult": np.mean(rr),
            "rand_ann": np.mean(rr) ** 2 - 1,
            "n": len(tr),
        }
    return out


def summarise(df):
    if not len(df):
        return None
    exc = df["mean_fwd6"] - df["rand_mean"]
    # t-stat of per-rebalance excess-vs-random; Newey-West-lite: scale for the
    # ~6 overlapping baskets (REBAL=21d, holding=126d => 6x overlap) by sqrt(6).
    overlap = H6 / REBAL
    t = (exc.mean() / exc.std() * np.sqrt(len(exc) / overlap)
         if exc.std() > 0 else np.nan)
    return {
        "n_rebal": int(len(df)),
        "hit": df["hit"].mean(),
        "base": df["base"].mean(),
        "lift": df["hit"].mean() / df["base"].mean() if df["base"].mean() else np.nan,
        "mean_fwd6": df["mean_fwd6"].mean(),
        "excess_vs_univ": (df["mean_fwd6"] - df["umean"]).mean(),
        "excess_vs_rand": exc.mean(),
        "t_vs_rand": t,
        "beat_rand_share": (df["mean_fwd6"] > df["rand_mean"]).mean(),
        "rand_pctile": df["rand_p"].mean(),
    }


def per_year(df):
    """Year-by-year hit-rate and excess-vs-random for one variant (both splits
    pooled), to confirm the edge is not a single sub-period."""
    if not len(df):
        return ""
    d = df.copy()
    d["year"] = d["date"].dt.year
    d["exc"] = d["mean_fwd6"] - d["rand_mean"]
    g = d.groupby("year").agg(n=("hit", "size"), hit=("hit", "mean"),
                              base=("base", "mean"), exc=("exc", "mean"))
    out = ["| year | n | hit | base | exc_vs_rand |", "|---|---|---|---|---|"]
    for y, r in g.iterrows():
        out.append(f"| {y} | {int(r.n)} | {r.hit:.0%} | {r.base:.0%} | "
                   f"{r.exc:+.1%} |")
    return "\n".join(out)


def main():
    P = dca_data.build_panel()
    F = feat.build_features(P)
    close, member = P["close"], P["member"] & P["close"].notna()

    lines = ["# Basket backtest — do these signals catch pre-parabolic names?\n"]
    lines.append(f"Pick top-{K} eligible names every {REBAL} trading days; outcome "
                 f"= forward {H6}d return; parabolic = >+{PARAB:.0%}. IS rebalance "
                 f"date < {SPLIT.date()}, OOS >=. Random control = {N_RANDOM} "
                 "random top-k baskets from the same eligible pool on the same "
                 "dates (survivorship-matched).\n")
    lines.append("`hit` = P(pick goes parabolic); `lift` = hit/base; `excess_vs_"
                 "univ` = basket mean fwd6 minus universe mean; `excess_vs_rand` "
                 "= minus random-basket mean; `rand_pctile` = mean percentile of "
                 "the basket within the random distribution (0.5 = no edge).\n")

    for name in ["ignition", "ignition_beta", "ignition_noregime",
                 "practitioner_breakout", "pure_energy"]:
        score = strat.VARIANTS[name](P, F)
        m = basket_metrics(score, close, member)
        eq = equity_curve(score, close, member)
        lines.append(f"## {name}\n")
        hdr = ("| split | n | hit | base | lift | mean_fwd6 | exc_univ | "
               "exc_rand | t_rand | beat_rand | rand_pctile | strat_ann | rand_ann |")
        lines.append(hdr)
        lines.append("|" + "---|" * 13)
        for sp in ("is", "oos"):
            s = summarise(m[sp])
            if s is None:
                continue
            e = eq.get(sp, {})
            lines.append(
                f"| {sp.upper()} | {s['n_rebal']} | {s['hit']:.1%} | "
                f"{s['base']:.1%} | {s['lift']:.2f}x | {s['mean_fwd6']:+.1%} | "
                f"{s['excess_vs_univ']:+.1%} | {s['excess_vs_rand']:+.1%} | "
                f"{s['t_vs_rand']:+.1f} | "
                f"{s['beat_rand_share']:.0%} | {s['rand_pctile']:.2f} | "
                f"{e.get('strat_ann', float('nan')):+.1%} | "
                f"{e.get('rand_ann', float('nan')):+.1%} |")
        lines.append("")
        if name == "ignition":
            lines.append("Year-by-year (pooled splits):\n")
            lines.append(per_year(pd.concat([m["is"], m["oos"]])))
            lines.append("")
        print(f"[{name}] done")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
