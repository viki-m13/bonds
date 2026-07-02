# v2 — PIT-honest rebuild: execution timing, full costs, and the real frontier

**Status:** research complete. READ-ONLY analysis; nothing traded live.
**Date:** 2026-07 session. **Repo:** `viki-m13/bonds`, branch `claude/strategy-cagr-sharpe-optimization-76uujo`.
**Question asked:** can the 4-sleeve system (FINDINGS.md §7) reach **honest OOS CAGR > 20% AND Sharpe > 2** with point-in-time universes, full cost/execution modeling? And: is it better to run the model after the close and execute at the next open, or to execute at the close?

---

## 0. TL;DR — the honest verdict

1. **No. CAGR > 20% with Sharpe > 2 is not reachable with this strategy family on honest data.** The prior "Sharpe 2.04" decomposes almost entirely into survivorship bias, monthly-Sharpe accounting, light costs, and understated cross-sleeve correlations. The honest, fully-costed, PIT frontier is:
   - **max-Sharpe config:** ~**0.94–1.00 daily Sharpe** (≈1.14–1.19 monthly), CAGR ~6–7%, maxDD ~−10/−14% (6-sleeve, unlevered);
   - **max-CAGR config:** ~**19.7% CAGR** at Sharpe 0.73 and −58% maxDD (5-sleeve, ~2.8× gross, financed at FF+150bp) — i.e. QQQ-like return with QQQ-like pain.
2. **The ceiling is structural, not a tuning failure.** Average honest sleeve Sharpe = 0.73, average pairwise correlation = 0.30 ⇒ even with *infinitely many* such sleeves the ensemble Sharpe ceiling is **s̄/√ρ̄ = 1.33 daily**. Sharpe 2 requires either genuinely uncorrelated (ρ≈0) sleeve families (intraday, options-vol, futures carry — data we don't have) or accepting the biased accounting that produced 2.04.
3. **Execution timing (the second question) has a clean, strategy-class-dependent answer:**
   - **Fast mean-reversion wants immediacy or passivity, never a lag.** For the Connors NDX sleeve, "run after the close, work resting **limit orders** during the next day" is BOTH the most honest (no 3:55pm signal approximation) AND by far the best: Sharpe 0.94–1.07 vs 0.50 (same-close MOC), 0.16 (next-open MOO), 0.10 (next-close). The limit discount (0.9×ATR below the signal close) *is* most of the alpha.
   - **The dip sleeve needs same-close MOC** (Sharpe 0.52–0.68 at close t vs 0.32 at close t+1). Running the signal at ~3:55pm is feasible: measured on committed 5-min data, the last-10-minute drift is ~6bp median (p90 ~20bp) and only **5–11% of signal-day signals flip** — and those are the marginal, lowest-edge ones.
   - **Slow sleeves don't care** (momentum, TSMOM, bond/gold trend: ΔSharpe ≤ 0.05 between same-close and next-close). Execute them at whichever auction is operationally convenient.
   - **Next-open (MOO) execution is never the best option** for any sleeve tested: it gives up the overnight mean-reversion continuation without gaining the limit discount.
4. Costs matter exactly where turnover lives: the dip sleeve loses **~3.5%/yr** to costs (9k trades), momentum ~0.3–1%/yr, MR-limit almost nothing (passive fills, 300 trades/decade), TSMOM/crisis negligible.
5. Two upgrades from the prior arc DID replicate on PIT data (dev-picked, holdout-confirmed): **MR capital efficiency** (pf30: dev 1.02/hold 1.09, beats pf20 on all axes) and **risk-parity diversification** (ensemble Sharpe ≥ best sleeve in both halves of both windows). The rest of the prior claim did not survive honest data.

---

## 1. What changed vs FINDINGS.md (v1)

| dimension | v1 (prior arc) | v2 (this work) |
|---|---|---|
| NDX universe | ever-member (survivorship-*reduced*) | **true PIT membership mask, 2015–2026** (`data/pit/n100_panel_*`) |
| S&P sleeves universe | current S&P 500 (biased) | **PIT top-500 dollar-ADV** from the delisting-inclusive Tiingo panel (24k names incl. 8.9k delisted); no index-membership knowledge at all |
| Sharpe basis | monthly (inflated ~+0.2–0.3) | **daily**, plus excess-over-T-bill variant |
| costs | MR only; sleeves light; rebalancing free | **every fill costed** (commission $0.005/sh; auction +1bp+impact; marketable +half-spread by ADV tier; passive commission-only), inter-sleeve rebalance 5bp, √-law impact vs per-name dollar-ADV, $1M book |
| leverage | assumed free | financed at **FEDFUNDS+150bp**, idle cash earns 3M T-bill, on *actual* gross exposure (exposure-aware netting) |
| execution | signal-day close fills | 4 modes modeled per sleeve: same-close MOC / next-open MOO / next-close / resting-limit; 3:55pm signal error measured on 5-min data |
| data source | ephemeral /tmp pickles + API key | **100% committed repo data** — reproducible with no API key |

Data hygiene fixes that mattered: NASDAQ test symbols (ZXZZT/ZTEST/…) and lowercase-dupe tickers dropped; bad ticks (>4× or <0.25× of 11-day rolling median) removed — without this, "momentum" happily buys a 200,000% data error. AAPL/ADP/BKNG were missing from the Tiingo pull (downloader gap) and are grafted from the committed Yahoo-adjusted csvs.

---

## 2. Execution-timing study (the core new result)

### 2.1 NDX-100 PIT mean-reversion (sleeve A), 2015-01→2026-06, all costs, pf20 cash-capped

| execution mode | CAGR | Sh(d) | maxDD | win | #tr | avg/trade |
|---|--:|--:|--:|--:|--:|--:|
| **limit next day (original spec)** | **9.9%** | **0.94** | **−13%** | 70% | 325 | **+1.76%** |
| same-close MOC (3:55 signal) | 6.7% | 0.50 | −28% | 74% | 1,572 | +0.37% |
| next-open MOO | 1.2% | 0.16 | −46% | 67% | 1,552 | +0.13% |
| next-close | 0.1% | 0.10 | −44% | 58% | 1,538 | +0.03% |

Reading: the strategy without the limit-order discount barely exists. The resting limit at close−0.9×ATR only fills when the stock falls *further* — conditional selection that quadruples the per-trade edge. It also removes the same-close signal problem entirely: you compute signals **after** the close and place orders for tomorrow. **Robustness:** requiring the price to trade 5/10/20bp *through* the limit before counting the fill still gives Sharpe 1.04/1.00/0.92 (pf30) — the result does not live on optimistic fills. Costs are near-zero here because fills are passive and turnover is tiny.

### 2.2 Same-close (MOC) feasibility — measured, not assumed

From committed 5-min bars (2016+): drift from 3:55pm→close is median **~6bp** (p90 ~20bp) on SPY/QQQ/IWM, less on TLT/GLD. Recomputing RSI(2)<10 / RSI(5)<20 signals with the 3:55 price flips **5–11% of signals on signal days** (~0.3–1% of all days) — and flipped signals are by construction the marginal ones. MOC orders (NYSE 3:50 / Nasdaq 3:55 cutoffs) execute at the official auction price with no spread. Same-close execution is *implementable* for close-based signals, at a small, quantified fidelity cost.

### 2.3 Verdict per sleeve

| sleeve | best execution | penalty for next-close lag |
|---|---|---|
| A NDX MR | run after close → **resting limits next day** | −0.84 Sharpe (catastrophic) |
| C stock dip RSI(5) | **MOC same close** (3:55 signal) | −0.20…−0.36 Sharpe |
| E ETF RSI(2) | MOC ≈ MOO (0.62/0.60); limit variant *worse* (too few fills) | small |
| B momentum | indifferent | −0.03 |
| D bond/gold trend | indifferent | −0.04 |
| F 15-ETF TSMOM | indifferent | −0.03 |

**Answer to "run after close and execute at open, or run at close?":** neither, as a blanket rule. Run everything after the close; execute the *slow* books at the next convenient auction (open or close — measurably irrelevant), execute the *fast* mean-reversion book via resting limit orders placed for the next session (strictly better than any market execution), and if you trade close-based dip signals, use MOC with a ~3:55 signal calc — the approximation error is ~6bp/10%-flips, far cheaper than the 1-day lag (which costs 0.2–0.8 Sharpe). Next-open market execution is dominated in every test.

---

## 3. Honest per-sleeve results (full costs, own windows)

| sleeve | window | CAGR | Sh(d) | Sh(m) | maxDD | expo | v1 claimed |
|---|---|--:|--:|--:|--:|--:|---|
| A NDX MR limit pf30 (dev-picked) | 2015–26 | 13.1% | **1.04** | 1.22 | −14% | 12% | 18.8%/1.54(m), ever-member |
| B momentum 12-2 top50 σ-rank+gate+vs (dev-picked) | 2001–26 | ~12% | 0.7–0.9 | — | −33% | ~74% | 31.3%/1.22(m), current-S&P |
| C dip RSI5<20 3d hold 20×5% (dev-picked) | 2000–26 | 10.2% | 0.71 | 0.84 | −26% | 43% | 17.4%/1.25(m), current-S&P |
| D bond/gold trend | 2005–26 | 4.7% | 0.61 | 0.61 | −22% | 66% | ~same |
| E SPY+QQQ RSI(2) MR (new) | 2005–26 | 3.9% | 0.62 | 0.81 | −14% | 11% | — |
| F 15-ETF TSMOM long-flat (new) | 2008–26 | 4.4% | 0.60 | 0.66 | −16% | 95% | — |

Key decompositions:
- **Momentum:** survivorship-clean + costs takes 12-1 top-20 from the claimed 31%/1.22(m) to **11%/0.46(d) with −74% maxDD**. The delisting-inclusive universe is full of momentum names that subsequently died; current-membership backtests never met them. This was the single biggest inflation in v1.
- **Dip:** clean universe + costs: 17.4%/1.25 → **8.9%/0.52** (5d hold, MOC). Costs alone are −3.5%/yr. The dev-picked 3-day-hold variant improves to 0.71 overall — but its **holdout (2016+) Sharpe is only ~0.3**: the liquid-universe RSI(5) dip edge has substantially decayed in the last decade. Recorded, not hidden.
- **MR:** the one sleeve that got *better* under honesty-plus-discipline (true PIT, higher pf, passive fills): 1.04 daily Sharpe, dev 1.02 / holdout 1.09.
- Correlations measured daily are **0.3–0.55 among the equity sleeves** (v1's monthly, differently-universed streams suggested 0.07–0.42). The "free lunch" was half mirage.

---

## 4. The ensemble frontier (exposure-aware: idle cash at T-bill, debit at FF+150bp)

**FULL 6-sleeve, 2015+, inverse-vol weights, monthly, all costs** (dev=2015–20, hold=2021–26):

| k | gross | CAGR | Sh(d) | Sh(m) | maxDD | dev/hold Sh |
|--:|--:|--:|--:|--:|--:|---|
| 1.0 | 0.45 | 5.9% | **0.94** | 1.14 | −14% | 0.95 / 0.93 |
| 2.0 | 0.90 | 9.0% | 0.75 | 0.89 | −28% | 0.84 / 0.66 |
| 3.0 | 1.35 | 11.4% | 0.67 | 0.78 | −40% | 0.79 / 0.56 |
| 4.0 | 1.80 | 13.3% | 0.62 | 0.71 | −51% | 0.75 / 0.50 |

**LONG 5-sleeve (no NDX-MR), 2008+**: unlevered 6.1%/1.00(d)/−10%; the most aggressive tested config (equal-vol ×4, ~2.8 gross) reaches **19.7% CAGR at 0.73 Sharpe, −58% maxDD**. QQQ over the same window: 17.7%/0.84/−47%.

Why leverage can't rescue it: the unlevered book earns only ~2–4% over cash; levering multiplies that thin excess while financing eats FF+150 on the margin debit, and vol-drag compounds. Sharpe *falls* monotonically with k in every window/scheme (see `charts/v2_leverage_frontier.png`).

**The ceiling math** (the most useful single number in this study): with average sleeve Sharpe s̄=0.73 and average pairwise correlation ρ̄=0.30, an equal-risk ensemble of N such sleeves has Sharpe s̄·√(N/(1+(N−1)ρ̄)) → **s̄/√ρ̄ = 1.33** as N→∞. Adding more sleeves *of this class* asymptotes there; it never approaches 2. To reach Sharpe 2 you need sleeve families with ρ≈0 to these (intraday vol breakout, options IV-carry, futures term-structure — exactly the author's paywalled, data-gated strategies) or sleeve Sharpes ≥1.2 each, which public daily EOD equity data does not honestly offer.

---

## 5. What survived, what didn't (scoreboard vs v1's claims)

| v1 claim | verdict on honest data |
|---|---|
| RSI(2) dip-quality filter improves MR | ✅ survives (PIT: Sharpe 0.94→1.04 with pf30, holdout-confirmed) |
| capital efficiency (scale the better book) | ✅ survives (pf20→pf30 better on all axes, dev+hold) |
| risk-parity ensemble ≥ best sleeve | ✅ survives (both windows, both halves) |
| bond/gold crisis alpha ~0 corr | ✅ survives (corr 0.02/−0.01 to equity MR/dip) |
| momentum sleeve 31%/1.22 | ❌ survivorship artifact (honest: ~11–13%/0.5) |
| dip sleeve 17%/1.25 | ❌ survivorship + costs (honest: 9–10%/0.5–0.7, decaying) |
| near-zero sleeve correlations | ❌ daily honest: 0.3–0.55 among equity sleeves |
| Sharpe 2.04 system | ❌ honest daily: ~0.94–1.00 unlevered; ceiling 1.33 |
| "~22–23% CAGR deployed" | ❌ honest: 20% CAGR costs Sharpe 0.73 and −58% DD |

---

## 6. Honest caveats on THIS study (do not strip)

1. **NDX PIT window is short** (membership mask starts 2015; 92–96% price coverage 2015–18 — a few delisted early names missing ⇒ MR numbers mildly optimistic).
2. Tiingo panel lacks ~50 OTC-catastrophe delistings (LEHMQ-class) ⇒ clean-universe results still *mildly* optimistic, mostly pre-2010.
3. The broad panel is **close-only**: stock-level MOO could only be tested on the NDX PIT panel (where it lost badly); ETF-level MOO tested directly.
4. Dev/holdout splits guard selection but were used repeatedly across variants; treat every "best" as an optimistic point estimate. Rejected levers are listed in the scripts' outputs (momentum SPY-gate helps dev, hurts holdout; 7d dip hold; ETF-MR limit variant; vol-targeting at ensemble level).
5. Cost model is an estimate (tiered half-spread + √-impact at $1M book). At $100k book, costs are smaller; at $10M+, impact grows.
6. Momentum sleeve exposure is granular (gate on/off) — its financing/cash netting is approximate at the ensemble level.

## 7. Reproduction

```bash
cd dca/research/strategies/crackingmarkets_repro/v2_pit/scripts
python3 00_audit_data.py          # data sanity (no API key needed — all committed)
python3 20_universe_clean.py      # PIT top-500 liquidity universe (~2 min)
python3 10_mr_pit_execution.py    # sleeve A + the 4-mode execution study
python3 21_momentum_clean.py      # sleeve B baseline
python3 22_buythedip_clean.py     # sleeve C
python3 23_crisis_alpha.py        # sleeve D
python3 30_improve_momentum.py    # sleeve B dev/holdout variants
python3 31_etf_meanrev.py         # sleeve E
python3 32_tsmom_multiasset.py    # sleeve F
python3 33_moc_feasibility.py     # 3:55pm signal-error measurement
python3 40_ensemble.py            # v1-style ensemble (naive financing)
python3 41_ensemble_v2.py         # exposure-aware ensemble
python3 42_final_system.py        # FINAL: dev-picked configs, frontier, charts
```
Charts: `charts/v2_final_system.png` (equity/DD/rolling-Sharpe vs QQQ), `charts/v2_leverage_frontier.png` (the CAGR-vs-Sharpe trade-off), `charts/v2_ensemble_final.png`.

## 8. If you still want 20%+/2.0 — the honest menu

1. **Different sleeve families, not more of these:** intraday volatility breakout (needs 1-min futures/ETF data), options IV mean-reversion / vol carry (needs options data), futures trend+carry across 40+ markets. These are the ρ≈0 families that could lift the ceiling. Budget for paid data (CRSP/Sharadar for equities PIT, CME/OPRA for the rest).
2. **Accept ~1.0 honest Sharpe and compound:** the unlevered 6-sleeve at ~6%/−14% or MR-heavy variants at ~13%/−14% are deployable and real. At 1.3–1.5× (gross <1) they beat SPY risk-adjusted with a third of the drawdown.
3. **Re-examine the goal:** QQQ 2015–26 did 19.8% CAGR at 0.92 Sharpe. Any honest "20%+" from long-equity sleeves in this era is roughly "hold QQQ with extra steps." The value the sleeves genuinely add is the **drawdown profile** (−14% vs −35%), not the headline CAGR.
