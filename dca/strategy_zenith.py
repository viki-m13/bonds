"""ZENITH — maximal-conviction DCA stock accumulation (the profitability frontier).

RETRACTED (2026-06-17) as a validated QQQ-beater: the headline numbers in this
docstring were produced on a survivorship-biased panel (~40% of historical S&P
members missing; delisted names booked benignly) and are dominated by 2018-26
recency; the bull score has ~0/negative cross-sectional IC vs QQQ in-sample. A
survivorship-clean rebuild collapses ZENITH 25.7x -> ~11.2x (ties QQQ). It is NOT
a Pareto improvement over SUMMIT. See research/results_zenith.md (Post-hoc) and
research/VALIDATION_METHODOLOGY.md. Code retained for the record; the claims below
describe the biased-panel backtest only.

ZENITH is SUMMIT's regime-conditional mega-cap-momentum signal run at **maximal
conviction: k=1**. Every contribution buys the single highest-scoring name (the
current #1 leader in risk-on regimes, the #1 discounted-quality rebounder in
risk-off regimes) at the next open, and never sells.

Why k=1 (the whole idea, pre-registered from the literature):
  Bessembinder (2018) — essentially *all* long-run net equity wealth is created
  by a tiny right tail of extreme winners (4.3% of US firms since 1926; the top
  ~0.33% make half), and that concentration is rising. Against a cap-weighted
  benchmark (QQQ), the optimal posture is therefore to (a) tilt hard to the
  biggest names, (b) ride the single strongest leader, and (c) never trim the
  right tail. SUMMIT already does (a) and (c); ZENITH adds (b): it stops
  splitting the contribution across two names and feeds the single best one.

Empirically (PIT S&P 500, 244 windows, next-open, 5bps, delisting-aware), this
is a *Pareto* improvement over SUMMIT k=2, not a profit-for-risk trade:
  full multiple 20.0x -> ~25.7x, window win-rate vs QQQ-DCA 93% -> 95%, median
  excess +28.8% -> +43%, OOS (2015-2023 starts) win 99%, OOS median +65%.
  Validated identically to SUMMIT: leakage audit clean, reference==fast,
  phase/cost/cadence robust, beats the random-pick survivorship control
  decisively, NASDAQ-100 PIT transfer positive. See research/results_zenith.md.

The one honest cost is **single-name concentration** the backtest understates:
the live book becomes dominated by one or two names, so idiosyncratic tail risk
(a sudden single-name collapse) is higher than k=2. For that risk — not for
return — an optional concentration cap (`trim_cap`) sells only the *excess* of
any holding above a weight cap once a year, leaving the rest (and its deferred
gains) untouched. It costs a little terminal multiple and is purely a
risk-management toggle.

Signal identity is byte-for-byte SUMMIT's `build_scores` (re-exported here), so
the leakage audit and reference cross-check carry over unchanged. ZENITH differs
only in the execution parameter k=1.
"""
import data as data_mod
import strategy_dca

# The signal is identical to SUMMIT — re-export so audits/factsheets can point
# at one canonical builder.
build_scores = strategy_dca.build_scores
bull_scores = strategy_dca.bull_scores
bear_scores = strategy_dca.bear_scores
risk_off = strategy_dca.risk_off

K = 1                  # maximal conviction (the defining choice)
EVERY = 10             # biweekly cadence (cadence-robust; see validation)
COST_BPS = 5.0
# Optional live single-name risk cap (NOT a return lever): sell only the excess
# above this book weight, once a year. None = pure never-sell.
TRIM_CAP = None        # e.g. 0.33 to cap any one name at ~1/3 of the book


def current_pick(k: int = K) -> tuple:
    """Live helper: today's ZENITH pick (execute at the next open)."""
    P = data_mod.build_panel()
    s = build_scores(P)
    member = P["member"]
    enough = P["close"].notna().rolling(252).count() >= 252
    row = s.iloc[-1].where(member.iloc[-1]).where(enough.iloc[-1]).dropna()
    picks = row.sort_values(ascending=False).head(k)
    import pandas as pd
    regime = "RISK-OFF (rebound sleeve)" if risk_off(P).iloc[-1] \
        else "RISK-ON (momentum sleeve)"
    return pd.DataFrame({"score": picks}), regime
