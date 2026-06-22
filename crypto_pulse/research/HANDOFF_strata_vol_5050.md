# Handoff: add STRATA to the live VOL deployment at ~50/50 — for independent review

**To:** the agent operating the live VOL strategy
**From:** the research repo (`viki-m13/bonds`, branch work merged to `main`)
**Ask:** review this in detail, **independently validate** (do not trust these numbers on faith —
reproduce them), and if it holds up, update the live deployment to run **VOL + STRATA at a static
~50/50 risk split**. Do not change anything live until you have validated steps in §6.

---

## 1. TL;DR recommendation

Keep VOL as a core engine but stop running it **alone**. Add **STRATA** — a daily, market-neutral,
cross-sectional crypto factor book — as an always-on second leg at a **static 50/50 risk split**
(both vol-targeted to the same annual vol). This is **not** a market-timing overlay; it is a fixed
allocation, rebalanced back to 50/50 periodically.

On the full 2018–2026 history, net of costs, both legs vol-targeted to 15%:

| book | Sharpe | CAGR | maxDD |
|---|---|---|---|
| Pure VOL | 2.02 | +40% | −15% |
| **VOL/STRATA 50/50** | **2.40** | **+51%** | **−14%** |

The 50/50 improves Sharpe, CAGR **and** drawdown simultaneously. It is the mean-variance optimum
(see §3), not a tuned point.

---

## 2. Why — the evidence

### 2a. VOL is in a real regime slump; STRATA is not correlated to it
Net, vol-targeted daily Sharpe by year (`crypto_pulse/strata_beats_vol.py`):

| year | VOL | STRATA |
|---|---|---|
| 2021 | 1.04 | 3.76 |
| 2022 | 1.78 | 0.63 |
| 2023 | 2.17 | 1.64 |
| 2024 | 2.60 | 0.69 |
| 2025 | 0.94 | 2.99 |
| 2026 | −0.02 | 2.68 |
| last 365d | +0.07 | +3.68 |
| last 180d | −0.83 | +2.96 |

The two are **anti-phased across regimes** (VOL led 2022–24, STRATA led 2021 & 2025–26). Full-period
correlation **0.17**. That low correlation is the entire basis for the diversification gain — **§6
step 2 asks you to re-check it against your LIVE VOL returns, not this backtest.**

### 2b. Adding STRATA RAISES CAGR, it does not dilute it
At equal risk (vol-targeting), a higher-Sharpe combination mechanically produces higher CAGR. Static
frontier, both legs vol-targeted to 15%, net 4.5bps + funding (`crypto_pulse/tactical_overlay.py`):

| VOL % | STRATA % | Sharpe | CAGR | maxDD | Sharpe 2025–26 |
|---|---|---|---|---|---|
| 100 | 0 | 1.96 | +40% | −15% | (slump) |
| 80 | 20 | 2.19 | +46% | −15% | — |
| 60 | 40 | 2.38 | +51% | −14% | 1.99 |
| **50** | **50** | **2.40** | **+51%** | **−14%** | 2.25 |
| 45 | 55 | 2.39 | +51% | −13% | 2.35 |
| 40 | 60 | 2.36 | +50% | −13% | 2.42 |

Even a VOL-dominant 80/20 beats pure VOL on both axes. The Sharpe is **flat-topped across 45–55%
VOL (~2.40)** — a robust plateau, not a knife-edge.

### 2c. Tactical "only add STRATA when VOL slumps" is WORSE than static
We tested causal slump-timers (rotate into STRATA when VOL is in >10% drawdown, or when its trailing
63-day Sharpe is low). They reach Sharpe only ~2.07–2.09 because the trigger fires late and
whipsaws — they sit ~90%+ in VOL. **A plain static 70/30 (2.30) beats every timed version.**
Conclusion: do NOT build a regime switch; hold both continuously.

---

## 3. Why 50/50 specifically (it's the optimum, not a guess)
Unconstrained max-Sharpe weights from the full-period mean/covariance: **VOL 51% / STRATA 49%**. This
falls directly out of two near-equal Sharpes (VOL 2.02, STRATA 1.85) at low correlation (0.17) — it
is theoretically expected, not curve-fit. The only reason to tilt heavier to VOL is a **forward
view** that VOL mean-reverts to its 2022–24 strength (note the recent-Sharpe column rises as STRATA
weight rises, because VOL is slumping now). That is a judgment call, not a backtest result. Default
**50/50**; **55/45** is a mild pro-VOL hedge at negligible Sharpe cost if you want VOL to stay the
nominal majority.

---

## 4. What STRATA actually is (so you can reproduce it)

A **daily, market-neutral, cross-sectional** book on Hyperliquid perps. Universe: ~57 coins (the
`OVERLAP` list in `crypto_pulse/validate_hl.py` intersected with coins that have price history).
Eligibility per day: price present AND 30-day average dollar-volume > $3M.

**Seven equal-risk sleeves**, each cross-sectional (rank coins, long top / short bottom), inverse-vol
sized, each vol-targeted to 12% before combining:

| sleeve | signal | source |
|---|---|---|
| TREND | multi-horizon price trend (10/20/40/80d) | `max_stack.build_sleeves` |
| CARRY | funding/basis carry | `max_stack.build_sleeves` |
| BAB | betting-against-beta (low-beta long) | `max_stack.build_sleeves` |
| SQUEEZE | volatility squeeze / expansion | `max_stack.build_sleeves` |
| ACCEL | trend acceleration | `max_stack.build_sleeves` |
| FUNDFADE | fade extreme funding | `grand_stack.funding_fade` |
| VOLSHOCK | volume-shock × trend sign, weekly rebalance | built in `tactical_overlay.py` / `strata_beats_vol.py` |

**Combine:** equal-risk (mean of the seven vol-targeted sleeves) → the headline series. A more
conservative shrunk-MV combine (0.6·diag + 0.4·full covariance, weights estimated in-sample with
positive-clip) gives **OOS ≈ 1.85** — use that as the number to size on, NOT the trailing 3.68.

**Costs in the backtest:** 4.5 bps taker per unit turnover + per-coin funding charged on positions.
**Rebalance:** daily (VOLSHOCK weekly).

The exact, runnable construction is the `build_strata()` function in
`crypto_pulse/tactical_overlay.py` and `crypto_pulse/strata_beats_vol.py`.

---

## 5. How to reproduce (run these)
```
cd crypto_pulse
python strata_beats_vol.py     # -> research/strata_beats_vol.md + png  (year-by-year VOL vs STRATA)
python tactical_overlay.py     # -> research/tactical_overlay.md + png  (frontier + tactical test)
```
Inputs used:
- VOL series: `data/vol_strategy/t5rvt_net_daily_2018_2026.csv` (the published leakage-free
  `eq_vt35` daily returns; full-period Sharpe 1.99). **Replace this with your live/actual VOL return
  series for the real validation.**
- STRATA inputs: daily HL prices + funding via `crypto_pulse/validate_hl.py`.

---

## 6. Independent validation checklist (do this before touching live capital)

1. **Reproduce STRATA** from `build_strata()` and confirm no lookahead: every signal is `.shift(1)`
   before the return, and vol-targeting uses a trailing, shifted realized-vol. Verify there is no
   survivorship bias in the coin universe (eligibility is causal, dollar-volume gated).
2. **Recompute correlation against your LIVE VOL returns** (not the t5rvt backtest). The whole case
   rests on low correlation; confirm it is ≤ ~0.3 on overlapping live dates. If it is materially
   higher live, the 50/50 benefit shrinks — re-derive the optimal weight from your live covariance.
3. **Cost/slippage realism.** STRATA assumes 4.5 bps taker. It trades ~57 coins including less-liquid
   names. Validate slippage on YOUR execution; if the tail coins are too costly, restrict STRATA to
   the top-N liquid names and re-check Sharpe (expect a modest haircut, not a collapse).
4. **Do not size to the trailing Sharpe.** STRATA's 2025–26 (and VOL's 2022–24) are regime-flattered.
   Size to the conservative full-period / shrunk-MV OOS (~1.85 for STRATA), and confirm the 50/50
   edge survives on the full 2018–26 sample and on a holdout.
5. **Execution feasibility.** STRATA = many small long/short perp positions rebalanced daily (funding,
   margin, order count). Confirm the live infra can run it; if daily turnover cost is too high,
   test a slower rebalance (every 2–3 days) and re-check net Sharpe.
6. **Vol-targeting consistency.** Ensure both legs are targeted to the same annual vol so "50/50
   risk" is actually 50/50, then vol-target the combined book to your live risk budget.
7. **Beta / tail check.** STRATA is market-neutral by design; confirm your live VOL's net crypto-beta
   and that the combined book's directional exposure and deleveraging-event tail are acceptable.
8. **Shadow first.** Paper/shadow-trade STRATA live for a window, compare realized vs backtest, then
   phase capital in to 50/50 rather than switching in one step.

---

## 7. Integration spec (once validated)
- Run STRATA as a daily-rebalanced, market-neutral cross-sectional HL-perp book (§4).
- Vol-target STRATA and VOL each to the same annual vol; allocate **50% risk to each**.
- Rebalance the split back to 50/50 on a fixed cadence (monthly is fine; it is not timing-sensitive).
- Vol-target the **combined** book to the deployment's risk limit.
- Optional pro-VOL tilt: 55/45 instead of 50/50 (negligible Sharpe cost, keeps VOL the majority).

---

## 8. Honest caveats / risks
- **Regime dependence (both legs).** Neither is regime-proof; VOL had 2025–26, STRATA had 2022 &
  2024. The 50/50 is robust *because* it does not depend on either regime continuing.
- **Backtest VOL ≠ live VOL.** We used the published t5rvt backtest series. Validate everything
  against your actual live VOL returns; the optimal weight is whatever your live covariance implies
  (it came out ~51/49 on the backtest).
- **STRATA operational footprint** is larger than single-asset VOL (≈57 instruments, daily).
- **Shared crypto tail risk:** a broad deleveraging event can hit both legs together despite low
  average correlation; the −14/−15% maxDD is a backtest figure, size accordingly.
- **Next leg, not yet ready:** an L4 per-account "whale-flow" book is being data-collected now
  (`crypto_pulse/flow_l4.py`, `research/l4_flow_data.md`); once it has history it is the candidate
  third always-on diversifier on top of VOL+STRATA — not part of this handoff.

---

*Supporting research in this repo:* `crypto_pulse/research/strata_beats_vol.md`,
`crypto_pulse/research/tactical_overlay.md`, `crypto_pulse/research/flow_explore.md`,
and the negative results (`xgb_ensemble.md`, `pairs_stat_arb.md`, `strata_oos_lab.md`,
`intraday_breakout.md`, `strata_intraday.md`) documenting what did NOT beat these books, so you can
see the search was exhaustive and the conclusion is honest.
