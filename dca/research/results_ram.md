# Risk-Adjusted Momentum / Volatility-Structure family (ram_*)

Builders: `research/signals_ram.py`. Scorecards: `research/scorecards/ram_*.json`.
All signals trailing-only (rolling windows ending at row date; rolling beta via
rolling cov/var vs SPY; cross-sectional ranks/quantiles within row d, members only).
Default eval: biweekly (every=10), 5 bps, k=3 unless noted.

Reference points: naive 6m momentum k=3 → win_qqq 73%, win_spy 88%, med +14.0%,
worst -39.7%. Random control (30 draws, k=3): mean win_qqq 13%, **max 21%**,
mean med_vs_qqq -14.3%. Every candidate below with win_qqq > 50% clears the
survivorship check by a wide margin.

## Results (sorted by win_qqq)

| name | k | every | win_qqq | win_spy | med_vs_qqq | worst_vs_qqq | regime notes |
|---|---|---|---|---|---|---|---|
| ram_resid126 (β252, Σresid 126d) | 2 | 10 | **82%** | **93%** | **+33.3%** | -59.4% | loses recovery_09-12 (-61%), GFC (-24%), bear22 (-17%); wins bull13-17 +22%, sideways +13%, covid +4%, ai_bull +32% |
| ram_resid126 | 1 | 10 | 80% | 89% | +33.7% | **-92.8%** | single-name tail risk unacceptable |
| ram_resid126_skip10 (meas. thru d-10) | 2 | 10 | 79% | 94% | +29.3% | -56.8% | same shape, slightly tamer tails, best win_spy |
| ram_resid126 monthly | 2 | 21 | 79% | 92% | +29.1% | -65.6% | cadence-robust; biweekly slightly better |
| ram_resid_b252_m252 | 2 | 10 | 78% | 86% | +32.5% | -62.7% | worse recovery_09-12 (-66%) |
| ram_resid_b252_m252 | 3 | 10 | 77% | 86% | +27.9% | -54.7% | best med at k=3 |
| ram_resid_b126_m252 | 3 | 10 | 77% | 86% | +27.3% | -56.3% | β window barely matters |
| ram_resid126 | 3 | 10 | 76% | 91% | +22.5% | -51.2% | p10_vs_qqq -4.3% at k=2: losses concentrated in 07-10-start windows |
| ram_resid126 monthly | 3 | 21 | 75% | 91% | +21.9% | -53.3% | |
| ram_resid_b126_m126 | 3 | 10 | 75% | 90% | +23.3% | -52.2% | |
| ram_resid_mom_blend (rank sum) | 2 | 10 | 74% | 88% | +17.2% | -46.9% | dilutes resid edge toward baseline |
| ram_resid126_ewmkt (EW-member factor) | 2 | 10 | 67% | 86% | +8.6% | -45.3% | factor choice matters: equal-weight market factor loses most of the edge |
| ram_mom126_base (control) | 3 | 10 | 73% | 88% | +14.0% | -39.7% | = protocol baseline, reproduced |
| ram_resid_b252_m63 | 3 | 10 | 71% | 88% | +17.4% | -45.3% | 63d resid window too short |
| ram_resid_mom_blend | 3 | 10 | 69% | 88% | +11.7% | -43.7% | |
| ram_resid126 | 5 | 10 | 68% | 90% | +12.7% | -39.0% | tamest tails, but median halves |
| ram_resid_skewpos25 | 3 | 10 | 67% | 75% | +19.7% | -36.6% | skew tilt costs win-rate |
| ram_skewmom_pos | 2 | 10 | 66% | 77% | +14.1% | -36.6% | bad in covid (-15%) |
| ram_skewmom_pos (mom rank + 0.5·skew rank) | 3 | 10 | 64% | 76% | +8.1% | -32.9% | **positive** skew helps, opposite of literature |
| ram_skewmom_pos_w25 | 3 | 10 | 64% | 80% | +9.6% | -35.4% | weight not sensitive |
| ram_sortino252 | 2 | 10 | 61% | 79% | +7.8% | -48.0% | |
| ram_sortino252 | 3 | 10 | 59% | 82% | +4.5% | -37.2% | weak everywhere ex-bull |
| ram_sortino126 | 3 | 10 | 56% | 81% | +4.9% | -29.5% | |
| ram_antilottery_mom126 (MAX filter) | 3 | 10 | 55% | 82% | +2.2% | -31.2% | filter HURTS: 73%→55% vs unfiltered |
| ram_sharpe252 | 3 | 10 | 53% | 85% | +1.2% | -30.7% | |
| ram_resid_antilottery | 3 | 10 | 52% | 81% | +1.0% | -37.8% | MAX filter hurts resid mom too |
| ram_pathq_126 (ret/Σ|ret|) | 3 | 10 | 44% | 80% | -2.3% | -27.9% | mildest tails of family, no edge |
| ram_resid126_scaled (idio Sharpe) | 3 | 10 | 43% | 86% | -2.2% | -36.6% | scaling by resid vol destroys the edge |
| ram_vsmom_12m_6mv | 3 | 10 | 33% | 77% | -5.2% | -29.2% | |
| ram_vsmom_6m_6mv | 3 | 10 | 29% | 77% | -6.5% | -30.0% | |
| ram_sharpe126 | 3 | 10 | 28% | 75% | -6.5% | -40.6% | |
| ram_skewmom_neg (literature dir.) | 3 | 10 | 19% | 59% | -9.9% | -37.4% | refuted in this setup |
| ram_trendr2_126 (R² × sign) | 3 | 10 | 17% | 72% | -11.3% | -40.0% | smoothness selects boring compounders, lags QQQ |
| ram_antilottery_sharpe126 | 3 | 10 | 16% | 71% | -10.6% | -42.2% | |
| ram_lowvolgate_sharpe126 | 3 | 10 | 10% | 37% | -15.7% | -50.1% | |
| ram_momvolrank_p50 | 3 | 10 | 10% | 36% | -19.6% | -52.8% | |
| ram_lowvolgate_mom126 | 3 | 10 | 7% | 39% | -14.4% | -49.4% | low-vol kills any chance vs QQQ |

## Honest findings

* **Vol-penalizing momentum is consistently harmful here.** Sharpe-scaling,
  vol-scaling, low-vol gates and vol-rank penalties all underperform raw 6m
  momentum (and most lose to SPY). To beat a QQQ DCA you need the high-vol
  high-beta winners; deleting them deletes the edge.
* **The MAX anti-lottery filter hurts** (73% → 55% win_qqq on raw momentum,
  76% → 52% on residual momentum). The "lottery" decile contains exactly the
  names driving outperformance in this universe.
* **Skew result is the opposite of the literature**: positive-skew momentum
  beats negative-skew momentum decisively (64% vs 19%), but neither beats raw
  momentum.
* **Residual (idiosyncratic) momentum is the only genuine improvement**:
  +22.5% median at k=3 vs +14% baseline, rising to +33% at k=2. Robust to
  β window (126/252), residual window (126/252, not 63), a 10-day skip, and
  monthly cadence — all configs land in 75–82% win_qqq.
* **Per-regime weakness, stated plainly**: every resid-mom config loses badly
  in recovery_2009_2012 (-49% to -66% vs QQQ — the classic post-crash momentum
  crash, where 12m-beta is stale and junk rallies), moderately in GFC (~-22%)
  and bear_2022 (~-17%). Worst grid windows are all 2007–2010 starts. Outside
  those, p10_vs_qqq is only -4.3% (k=2): the tail is concentrated, not chronic.
* **Factor-choice sensitivity**: regressing on an equal-weight member-mean
  market factor instead of SPY drops k=2 from 82%/+33% to 67%/+8.6%. The edge
  is specifically residual-vs-cap-weighted-market; part of it is an implicit
  anti-(SPY-beta) tilt. Treat this as a real fragility.
* **Data caveat**: the SPY benchmark file starts 2005-01, so SPY-based resid
  scores are NaN until ~mid-2006 (β 252d + Σ 126d). Windows starting
  2006-01–2006-04 hold cash until the first valid score; that biases those
  windows *against* the resid signals (cash drag in a bull half-year), so the
  headline numbers are conservative there, not flattered.
* k=1 is disqualified by a -93% worst window; k=5 dilutes the median to
  baseline levels. k=2–3 is the usable range.
* Nothing in this family reaches the 85% win_qqq bar on its own; the gap is
  entirely the momentum-crash windows, which suggests pairing with a
  regime/crash gate (e.g. drawdown- or VIX-conditioned fallback) from another
  family rather than further tweaks here.

## Top-3 recommendations

1. **ram_resid126, k=2** (β vs SPY over 252d via rolling cov/var, cumulative
   residual over 126d): win_qqq 82%, win_spy 93%, med +33.3%, worst -59.4%.
   Best of family; candidate for combination with a crash-regime gate.
2. **ram_resid126_skip10, k=2** (same, measured through d-10): 79% / 94%,
   med +29.3%, worst -56.8%. Best win_spy, slightly tamer tails; preferred if
   robustness is weighted over median.
3. **ram_resid126, k=3**: 76% / 91%, med +22.5%, worst -51.2%. The diversified
   choice — still clearly above the naive-momentum baseline with the mildest
   tail among the high-median configs.
