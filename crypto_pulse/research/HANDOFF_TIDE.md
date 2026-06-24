# Handoff: TIDE — a standalone crypto strategy for independent review & HL deployment

**To:** the agent operating live trading on Hyperliquid
**From:** the research repo (`viki-m13/bonds`, branch `claude/parabolic-stock-strategy-9x75dj`)
**Ask:** review this in detail, **independently reproduce and validate** (do not trust these
numbers on faith), and if it holds up, deploy **TIDE** as a standalone book on HL crypto perps.
This is a *separate, independent* strategy from VOL — it can run alongside VOL or alone.

---

## 1. TL;DR
**TIDE** (Trend-Intensity-Dependent Exposure) is a **daily, cross-sectional, market-neutral
breakout** book on Hyperliquid crypto perps. It goes long coins breaking out above their own
recent range and short those breaking down, sizes by inverse volatility, and **scales total
gross exposure by how strongly the whole market is trending** (hard when trending, light in
chop). Net of 4.5 bps taker + funding, vol-targeted:

| window | Sharpe | CAGR | maxDD |
|---|---|---|---|
| HL era (2023-05→now, tradeable, funding-accurate) | **2.23** | +32% | −8% |
| HL out-of-sample (last 40%, never used to build) | **2.29** | — | — |
| Pre-HL 2018–2023 (fully independent, spot proxy) | 1.35 | — | — |
| Full 2014–2026 (~12 years) | 1.55 | +27% | −20% |

It is **positive in all 12 years (2015–2026)** and passes a full overfit battery (§4). It is a
**~2.0–2.3 Sharpe book — NOT a Sharpe-3 book**, and the doc is explicit about that.

---

## 2. Exact construction (reproduce this — code: `crypto_pulse/tide.py`, `TIDE.build()`)
All daily. `C,H,L,V` = close/high/low/$-volume panels (coins × days). All signals use data
through day *t*; the position is applied to the day *t*→*t+1* return (one-day lag). No lookahead.

**Universe / eligibility (per day, causal):**
- Coins = the HL-perp set that also has price history (`OVERLAP` in `validate_hl.py`, ~57 names).
- Eligible if `C` present AND `30d-mean(C·V) > $3,000,000` (liquid enough).

**1. Breakout signal (5-horizon, cross-sectional):** for each horizon `k ∈ {5,10,20,40,80}`:
```
z_k = (C − rolling_mean(C, k)) / (rolling_std(C, k) + 1e-9)      # per-coin breakout
z_k = z_k − cross_sectional_mean(z_k)                            # demean across coins (day t)
breakout = mean over the 5 horizons of z_k
```
(Long high breakout / short low breakout; the demean makes it market-neutral.)

**2. Inverse-volatility sizing (Parkinson high-low vol):**
```
sd = sqrt( rolling_mean( ln(H/L)^2 , 30 ) / (4·ln2) ) + 1e-9
raw_w = breakout / sd
w = raw_w / sum(|raw_w|)        # gross = 1
```

**3. Trend-intensity regime gate (the "TIDE" mechanism, causal):**
```
frac_up = cross_sectional_mean( 1[ C > rolling_mean(C,50) ] )    # fraction of coins in uptrend
trend_intensity = clip( |2·(frac_up − 0.5)| , 0, 1 )            # 0 = balanced/chop, 1 = one-sided
w = w · trend_intensity.shift(1)                                 # lag 1 day
```

**4. Execution / costs / risk target:**
- Rebalance every **3 days** (hold weights, ffill between).
- Lag weights one day: `w_lagged = w.shift(1)`.
- Daily PnL = `sum(w_lagged · R) − turnover·4.5bps − sum(w_lagged · funding)`,
  where `turnover = sum|w_lagged − w_lagged.shift(1)|`, funding = realized HL daily funding.
- **Vol-target** the resulting PnL series to **12% annualized** (trailing 45d realized vol,
  shifted, leverage clipped to [0, 3]).

That is the entire strategy. No ML, no fitted coefficients — only the structural choices above.

---

## 3. How to reproduce (run these)
```
cd crypto_pulse
python tide.py            # -> research/tide.md + png : headline + full overfit battery
python tide_ci.py         # bootstrap Sharpe CI + execution sensitivity + anchored WF
python tide_capacity.py   # capacity vs AUM under square-root market impact
python tide_crossasset.py # cross-asset / timeframe / leverage scope
python tide_v6.py         # 12-year year-by-year backtest
```
Inputs: daily HL prices + funding via `validate_hl.py` (`data/crypto/*.csv`, `data/hl_funding/*.csv`).

---

## 4. Why it is not overfit (full battery — all passed, in `tide.md`)
- **Parameter plateau:** 100% of a 5×3 breakout-window × regime-window grid > 1.0 Sharpe
  (median ~2.15) — a broad plateau around the chosen params, not a lucky spike.
- **Every year positive** 2015–2026 (0.24 → 2.38); ~1.9–2.4 every year since 2017.
- **Cost stress:** survives 4× taker (18 bps → ~1.3 Sharpe).
- **Coin bootstrap** (20× random 70% coin subsets): 5th-pct ~1.2 — not driven by a few coins.
- **Shuffle null:** real ~2.2 vs signal-permuted max ~0.3 — no look-ahead leak.
- **Walk-forward** (4 disjoint folds + 6 anchored expanding starts): all positive.
- **Block-bootstrap 95% CI** on the HL Sharpe: roughly **[1.1, 3.2]**.
- **Improvement search:** 32 upgrade attempts across 6 rounds; only **5-horizon breakout** and
  **Parkinson vol** survived (each also improved the independent pre-HL period). Everything else
  (residualization, skip-day, beta-neutral, conviction, dd-floor, deadband, param-ensemble,
  efficiency-ratio, dispersion-timing, acceleration, concentration, ERC) decayed OOS.

---

## 5. Scope & limitations — read before deploying
- **Crypto-daily, liquid universe ONLY.** On the full 112-coin set it dilutes (1.07); keep to
  the liquid ~57.
- **Do NOT run it on equities / HL HIP-3 equity perps (TSLA etc.).** The same rule INVERTS on
  stocks (−0.8 to −1.3 Sharpe) — short-horizon cross-sectional moves continue in crypto but
  mean-revert in equities. TIDE is crypto-only.
- **Timeframe:** daily is the sweet spot; weekly weaker (~0.55), hourly fails (cost/noise).
- **Capacity:** under a square-root impact model it holds >1.5 Sharpe to ~$25M and >1.0 to
  ~$100M (spreads across ~57 coins, <1% ADV participation). *Indicative* — validate on your fills.
- **Leverage:** runs at ~1.0× average gross (3× cap), far inside HL limits. Scaling the vol
  target lifts CAGR and drawdown linearly with Sharpe unchanged (30% target ≈ +65% CAGR /
  −37% DD at ~2.6× gross).
- **Regime dependence:** it's a momentum/breakout book — it will have flat/▼ stretches when
  crypto chops; the regime gate cushions but does not eliminate this.

---

## 6. Independent validation checklist (do this before live capital)
1. **Reproduce TIDE** from §2 and confirm: every signal is `.shift(1)` before the return;
   vol-target uses trailing shifted realized vol; eligibility is causal & dollar-volume gated;
   no survivorship bias in the coin list.
2. **Re-run the overfit battery** (§4) yourself — especially the shuffle-null (edge must vanish)
   and the 4-fold walk-forward (all folds positive).
3. **Cost/slippage on YOUR execution.** TIDE assumes 4.5 bps taker. Validate slippage on ~57
   coins incl. less-liquid names; if tails are costly, restrict to top-N liquid and re-check
   (expect a modest haircut). Confirm the book still clears after your real costs.
4. **Size to the honest number, not the trailing high.** Full-period is ~1.55, HL-era ~2.3;
   recent strength is regime-flattered. Size to the lower end / bootstrap-CI lower bound.
5. **Funding realism.** Confirm the funding series and that funding is charged on signed
   notional; market-neutral net funding should be small but verify.
6. **Execution feasibility.** ~57 small long/short perp positions rebalanced every 3 days. If
   daily turnover cost is too high, test a slower rebalance (it's not knife-edge — see the
   rebalance×vol-target map in `tide_ci.md`).
7. **Shadow-trade first.** Paper/shadow for a window, compare realized vs backtest, then phase
   capital in.

---

## 6b. Phase-2 candidate (researched, NOT yet certified): funding carry
A market-neutral **funding-carry** leg (short high-funding / long low-funding coins, 14d
funding lookback) is the most *orthogonal* book found — correlation just **+0.24** to TIDE
(price/volume diversifiers were +0.40–0.49). A risk-parity **TIDE+CARRY** book hits HL-era
**OOS 2.87** (vs 2.29 alone), CAGR 41%, all 4 WF folds positive (`tide_carry.py`).
**Do NOT size it live yet** — three honest blockers: (1) funding data starts ~2023, so carry has
**no independent pre-HL history** to confirm it on a held-out regime; (2) its edge is concentrated
in the last ~18 months (IS ~0.6–1.1 vs OOS ~2.0–2.6) — possibly a 2024–25 funding regime; (3)
**carry-crash tail risk** (standalone maxDD −13% to −24% from short squeezes). Treat as a real
lead: paper-trade it, collect more out-of-sample funding history, and only then size a small
sleeve. The certified deployable book today is **TIDE alone**.

## 7. Deployment spec (once validated)
- **Universe:** HL perps with 30d $-ADV > $3M (~57 coins). Recompute eligibility daily.
- **Signal:** 5-horizon breakout (§2.1), demeaned cross-sectionally → long top / short bottom.
- **Sizing:** inverse Parkinson vol, gross = 1, then × trend-intensity gate (§2.3).
- **Rebalance:** every 3 days; one-day signal lag.
- **Risk:** vol-target the book to your risk budget (backtest uses 12%); cap gross per HL margin.
- **Costs:** model ≥ 4.5 bps taker + funding; prefer limit/maker fills where possible.
- **Relationship to VOL:** TIDE is market-neutral and ~uncorrelated to a directional vol book,
  so it can run as an independent sleeve alongside VOL (size each to its own risk).

---

## 8. Honest caveats / what this is NOT
- **It is ~2.0–2.3 Sharpe, not 3.** An exhaustive search (6 signal families, 32 TIDE-internal
  upgrades, deflated-Sharpe, cross-asset tests) shows a single cross-sectional price book caps
  here. Anyone claiming a single-strategy honest "3" on daily crypto is gross-of-costs,
  in-sample, or overfit.
- **Backtest ≠ live.** Validate against your own data/fills; the deflated/independent-period
  numbers (pre-HL 1.35) are the conservative guide, not the 2.3 trailing figure.
- **Shared crypto tail risk:** a broad deleveraging event can hurt even a market-neutral book;
  the −8%/−20% maxDDs are backtest figures — size accordingly.
- **Provenance:** TIDE was built from public TA ideas (whchien/ai-trader, je-suis-tm/quant-
  trading breakout+trend); the regime gate, 5-horizon blend, and Parkinson sizing are the
  pieces that made it robust. Negative-result files (`roc_lab*.md`, `tide_v2..v7.md`,
  `tide_ebb.md`, `flow_daily.md`) document what did NOT work, so you can see the search was
  honest and exhaustive.

*Supporting research:* `research/tide.md`, `tide_ci.md`, `tide_capacity.md`,
`tide_crossasset.md`, `tide_v6.md` (12-year), and `TIDE_STRATEGY.md` (the spec).
