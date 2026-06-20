"""IGNITION — pre-parabolic stock-selection signals (PIT S&P 500, OHLCV only).

Design follows the evidence (research/literature.md, research/eventstudy.md), not
the folklore. The honest event study found that the canonical FinTwit "buy the
breakout near 52-week highs" archetype (Minervini Trend Template, nearness to
high, RS-line new high) does NOT precede parabolic 6-month moves in S&P 500
large caps — those signals select already-extended names and sit BELOW the base
rate for P(parabolic). What actually precedes parabolic runs here:

  * high "energy"  (beta_120 / vol_20d / adr_cc / max_dret_21) — the fat-tail
    selectors the literature (Bali MAX, Ang IVOL) flags as NEGATIVE mean but
    fat right tail; they give 3x P(parabolic) lift,
  * "already turned off the 52-week low" (dist_52w_low) — the single signal with
    a positive, sign-stable IS->OOS rank-IC and positive excess,
  * an episodic-pivot gap catalyst (ep_gap_20) — positive excess in both splits,
  * low correlation to the market (idiosyncratic names go parabolic).

The composite blends the mean-positive conditioners (dist_52w_low, ep_gap, fip
smoothness, low-corr) WITH the tail selectors (energy), exactly the architecture
the academic review prescribed for a convex right-tail objective. Two falsifier
variants (literal practitioner breakout, pure energy) are included so the
write-up can show what was tested and rejected.

Every score at row d uses information through close of d only; cross-sectional
ranks are within-row. Members-only (NaN elsewhere). The DCA engine executes at
the next open.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "dca"))
import data as dca_data  # noqa: E402
import features as feat  # noqa: E402


def _xs_rank(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Row-wise cross-sectional percentile rank in [0,1], NaN outside mask."""
    x = df.where(mask)
    return x.rank(axis=1, pct=True)


def _prep(P, F):
    member = P["member"] & P["close"].notna()
    risk_on = feat.regime_risk_on(P["close"].index)
    return member, risk_on


def ignition_score(P: dict | None = None, F: dict | None = None,
                   energy: str = "adr_cc", regime_aware: bool = True
                   ) -> pd.DataFrame:
    """The IGNITION composite. Higher = more pre-parabolic.

    Blend (equal-weight rank) of: early-turn (dist_52w_low), catalyst
    (ep_gap_20), energy tail (adr_cc/beta_120), idiosyncrasy (-corr_120),
    smoothness (fip). Restricted to the high-energy half of the universe (the
    parabolic-prone pond). If regime_aware, in risk-off the weight tilts toward
    the deep-discount rebound axis (dist_52w_high), which the event study shows
    lights up below the 200dma.
    """
    if P is None:
        P = dca_data.build_panel()
    if F is None:
        F = feat.build_features(P)
    member, risk_on = _prep(P, F)

    adr_rank = _xs_rank(F["adr_cc"], member)
    # energy gate: keep the top ~60% by ADR (parabolic moves need range)
    pond = member & (adr_rank >= 0.4)

    turn = _xs_rank(F["dist_52w_low"], pond)
    gap = _xs_rank(F["ep_gap_20"].fillna(-1), pond)
    en = _xs_rank(F[energy], pond)
    idio = _xs_rank(-F["corr_120"], pond)
    smooth = _xs_rank(-F["fip"], pond)        # low ID = smooth advance = good
    disc = _xs_rank(-F["dist_52w_high"], pond)  # deep discount (rebound axis)

    base = turn.add(gap).add(en).add(idio).add(0.5 * smooth)
    if regime_aware:
        ro = risk_on.reindex(base.index).fillna(True).to_numpy()[:, None]
        # below 200dma, add the deep-discount rebound axis (EDA: P(parab) ~17%)
        off_boost = np.where(ro, 0.0, 1.0) * disc.to_numpy()
        score = base.to_numpy() + off_boost
        score = pd.DataFrame(score, index=base.index, columns=base.columns)
    else:
        score = base
    return score.where(pond)


def practitioner_breakout_score(P=None, F=None) -> pd.DataFrame:
    """FALSIFIER: the literal FinTwit breakout archetype — Trend-Template gate +
    nearness to 52wh + IBD relative strength + volume shock. The event study
    predicts this UNDER-performs the base rate for parabolic capture; we keep it
    to demonstrate that honestly."""
    if P is None:
        P = dca_data.build_panel()
    if F is None:
        F = feat.build_features(P)
    member, _ = _prep(P, F)
    gate = member & (F["trend_template"] > 0.5)
    near = _xs_rank(F["nearness_52wh"], gate)
    rs = _xs_rank(F["rs_ibd_raw"], gate)
    vsh = _xs_rank(F["vol_shock"], gate)
    score = near.add(rs).add(vsh)
    return score.where(gate)


def pure_energy_score(P=None, F=None) -> pd.DataFrame:
    """FALSIFIER: buy the lottery wholesale (top ADR/beta/MAX). Big right tail,
    but the literature says negative mean — the basket backtest will show the
    drag."""
    if P is None:
        P = dca_data.build_panel()
    if F is None:
        F = feat.build_features(P)
    member, _ = _prep(P, F)
    a = _xs_rank(F["adr_cc"], member)
    b = _xs_rank(F["beta_120"], member)
    m = _xs_rank(F["max_dret_21"], member)
    return a.add(b).add(m).where(member)


VARIANTS = {
    "ignition": ignition_score,
    "ignition_beta": lambda P=None, F=None: ignition_score(P, F, energy="beta_120"),
    "ignition_noregime": lambda P=None, F=None: ignition_score(P, F, regime_aware=False),
    "practitioner_breakout": practitioner_breakout_score,
    "pure_energy": pure_energy_score,
}


if __name__ == "__main__":
    P = dca_data.build_panel()
    F = feat.build_features(P)
    last = P["close"].index[-1]
    for name, fn in VARIANTS.items():
        s = fn(P, F)
        top = s.loc[last].dropna().sort_values(ascending=False).head(10)
        print(f"\n[{name}] top-10 picks on {last.date()} "
              f"({s.loc[last].notna().sum()} eligible):")
        print("  " + ", ".join(top.index))
