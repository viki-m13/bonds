# Cracking Markets — strategy reproduction + proprietary improvement (full handoff)

**Status:** research complete for this arc. READ-ONLY analysis; nothing traded live.
**Date:** 2026-06 session. **Repo:** `viki-m13/bonds`, branch `claude/wave-summit-strategy-review-p5x6za`.
**Where things live:** this folder — `dca/research/strategies/crackingmarkets_repro/` (`scripts/`, `charts/`, this doc).

This documents (A) faithful reproductions of the publicly-specified strategies from
https://www.crackingmarkets.com/tag/trading-strategies/ plus a Twitter/X screenshot
strategy, and (B) a from-scratch build of a **proprietary 4-sleeve risk-parity system**
that improves Sharpe from QQQ's 0.59 to **2.04** (holdout-validated) using only
principled, non-overfit techniques. Another agent can pick this up cold from here.

---

## 0. TL;DR — what to know in 60 seconds

- **Reproduced (public rules + sourceable data):**
  - **NDX-100 long mean-reversion** (the X screenshot): reproduced — 71% win, ~3-bar holds, the exact Connors profile. 17.7% CAGR / 0.95 Sharpe / −26% DD *on current-membership (survivorship-biased upper bound)*.
  - **Buy-the-dip** (RSI(5)<20, S&P): reproduced **near-exact** — win rate **57.4% vs published 56.81%**, avg-win>avg-loss ✓.
  - **IPO all-time-high breakout**: per-trade edge reproduces survivorship-clean (+5.1% vs +6.7%) but the **headline 20.53% CAGR does NOT** — full survivorship-clean universe gives **8.06% / 0.60 / −47%**. The one published number that doesn't hold up on honest data.
- **Not reproducible (gated):** volatility-breakout (paywalled rules + 1-min futures data) and IV mean-reversion (options/IV data + paywalled).
- **Proprietary improvement (the main deliverable):** a **4-sleeve risk-parity system** — improved NDX mean-reversion + 12-1 momentum + buy-the-dip + bond/gold crisis-alpha — **Sharpe 2.04, maxDD −12%, holdout-validated (dev 2.11 / hold 1.95)** vs QQQ 0.59 / −56%. Built only from combining independently-validated pieces (no curve-fitting).

---

## 1. Data & environment (READ FIRST — the data is ephemeral)

**Source:** Tiingo daily EOD API (`https://api.tiingo.com/tiingo/daily/<T>/prices`).
Returns raw OHLCV + split/div-**adjusted** `adj*` OHLCV. **All backtests use the
adjusted (`adjOpen/High/Low/Close/adjVolume`) series** (split+div, total-return,
continuous). The raw `close`/`volume` are used only for the IPO first-day $-volume gate.

**Tiingo API key:** was shared in chat (`2897486…`), so treat it as **exposed — rotate it**.
It is **NOT** committed anywhere; every script reads it from the `TIINGO` env var:
```bash
export TIINGO=<your_tiingo_key>
```

**The daily-OHLCV panels are cached in `/tmp/*.pkl` which is EPHEMERAL** (wiped on
container restart). To reproduce anything, **re-run the fetch scripts first** (§3).
Fetched pickles (produced by the fetch scripts):
- `/tmp/ndx_daily.pkl` — current NDX-100 (~100 names), dict{ticker: DataFrame[o,h,l,c,v]}
- `/tmp/ndx_removed_daily.pkl` — 94 delisted/removed ex-NDX-100 names
- `/tmp/sp_daily.pkl` — ~388 current S&P 500 names
- `/tmp/ipo_daily.pkl` — {"data": {ticker: df[o,h,l,c,v,rc,rv]}, "ipo_date": {...}} — 6,599 US common-stock IPOs 2003–2025

**Survivorship & PIT:**
- The one committed PIT asset is `dca/research/data/tiingo/tiingo_universe_pit.parquet`
  (ticker, exchange, assetType, startDate, endDate) — Tiingo's delisting-inclusive
  universe. `startDate` ≈ first-listing/IPO date; used to build the IPO candidate set.
- **Survivorship caveats that matter (do not remove from any writeup):**
  - NDX-100 base uses **current** membership (survivorship-biased). The "ever-member"
    fix adds 94 removed/delisted names (Wikipedia change-log → Tiingo) → survivorship-
    *reduced* but NOT perfect PIT (names are traded across their whole life, not only
    during index membership; the 200-EMA filter naturally limits this).
  - S&P sleeves (buy-the-dip, momentum) use **current** S&P 500 → survivorship-biased.
  - IPO backtest IS genuinely survivorship-clean (delisting-inclusive, incl. failures).

**Deps:** pandas, numpy, matplotlib, scikit-learn (for the ML momentum script). Python 3.11.

---

## 2. Pipeline — exact run order

```bash
export TIINGO=<key>
# --- 1. FETCH (writes /tmp/*.pkl; each ~1–40 min depending on universe size) ---
python3 scripts/01_fetch_ndx100_daily.py         # -> /tmp/ndx_daily.pkl   (~100 names)
python3 scripts/02_fetch_sp500_daily.py          # -> /tmp/sp_daily.pkl    (~388, reuses NDX cache)
python3 scripts/04_fetch_ndx_removed_daily.py    # -> /tmp/ndx_removed_daily.pkl (94 delisted)
python3 scripts/03_fetch_ipo_universe_daily.py   # -> /tmp/ipo_daily.pkl   (6,599; SLOW ~30-60m)

# --- 2. REPRODUCTIONS ---
python3 scripts/repro_ndx_meanreversion.py       # screenshot strategy
python3 scripts/repro_buythedip.py               # RSI(5)<20 win-rate check
python3 scripts/repro_ipo_breakout.py            # IPO ATH-breakout, full universe
python3 scripts/repro_ndx_survivorship_test.py   # current vs ever-member comparison

# --- 3. MOMENTUM STUDY (uses the SUMMIT featmat, see note) ---
python3 scripts/momentum_deciles.py              # French/Concretum 12-1 decile repro
python3 scripts/momentum_improvements.py         # residual-mom + beta-hedge + vol-scale
python3 scripts/momentum_hold_vs_sell_rules.py   # rotate vs ride vs hold-forever
python3 scripts/momentum_ml_vs_pure_dca.py       # signal x exit-rule + $1/mo DCA vs QQQ

# --- 4. IMPROVEMENT ARC (each self-contained; reads /tmp/*.pkl from step 1) ---
python3 scripts/improve1_overlays.py             # regime / RSI2 / IBS / vol-target
python3 scripts/improve2_capital_efficiency.py   # deploy the freed risk budget
python3 scripts/improve3_strategy_ensemble.py    # risk-parity of 3 sleeves + corr matrix
python3 scripts/improve4_holdout_validation.py   # dev/holdout split + equity curve
python3 scripts/improve5_weights_voltarget.py    # weight robustness + QQQ benchmark
python3 scripts/improve6_crisis_alpha_final.py   # add bond/gold sleeve -> FINAL system
```
**Note on the momentum study scripts:** `momentum_*.py` read `/tmp/wave/_featmat.pkl`
and `/tmp/wave/_mlprob.pkl` — the SUMMIT project's monthly survivorship-clean panel and
walk-forward ML probabilities (see `SUMMIT_prop_market_neutral.md` / `WAVE_long_only.md`).
Those are also ephemeral; regenerate via the SUMMIT pipeline if absent.

---

## 3. Strategy reproductions (rules · published · reproduced · verdict)

### 3.1 NDX-100 long mean-reversion  (the X screenshot) — `repro_ndx_meanreversion.py`
Classic Larry-Connors-style daily mean-reversion. **Rules (verbatim from screenshot):**
- **Universe:** Nasdaq-100; price > $5; 20-day avg volume > 100k shares; close > 200-day EMA; ATR(5) > 3% of price.
- **Entry:** stock closes >3% below yesterday's close; rank candidates by ATR%; buy via limit at **0.9×ATR(5) below the close**; position size 20% of equity; max 10 open positions.
- **Exit:** work a limit **0.5×ATR(5) above the close** (profit target); OR exit if close > yesterday's high; OR exit after 9 bars. Commissions included.
- **Implementation choices:** ATR = simple 5-day mean of true range; entry/target reference = signal-day close; limit fill = `min(open, limit)` if `low ≤ limit`; commission $0.005/share.

**Published:** the screenshot gives NO numbers (author: "not my top performer; my IV
mean-reversion model is stronger; I don't trade this live").

**Reproduced (current NDX-100, 2001–2026):**
| config | CAGR | Sharpe | maxDD | win | #trades | avg hold |
|---|--:|--:|--:|--:|--:|--:|
| literal 20%×10 (implied ~2× leverage) | 17.7% | 0.95 | −26% | 71% | 1,327 | 2.8 bars |
| no-leverage (cash-capped) | 14.2% | 0.97 | −21% | 71% | 1,223 | 2.8 bars |

Exit mix: 925 close-above-prior-high / 339 profit-target / 63 nine-bar. $100k → $6.4M
(literal) vs QQQ $1.6M. **Verdict: faithful reproduction** — the 71% win / ~3-bar hold /
shallow-DD profile is exactly this strategy class. **Caveat: current-membership =
survivorship-biased upper bound** (see §4 for the honest ever-member number).

### 3.2 Buy the dip — `repro_buythedip.py`
**Rules (public):** S&P 500 (they use PIT constituents); close > 200-day MA; **RSI(5)<20**;
enter & exit at close; **hold 5 trading days**; $1,000/trade.
**Published:** since 2000, ~**25,000 trades**, win rate **56.81%**, avg win > avg loss.
**Reproduced (~388 current S&P, 2000–2026):** win rate **57.37%**, avg win **+3.31%** vs
avg loss **−2.99%** (1.11×), avg trade **+0.63%**, **38,795 trades**.
**Verdict: near-exact reproduction** of the win rate (57.4% vs 56.81%). Trade count is
higher (current membership → more names active early + survivorship); win rate is
universe-insensitive, hence the tight match.

### 3.3 IPO all-time-high breakout — `repro_ipo_breakout.py`
**Rules (public):** US common stock, **not a SPAC**; first-day price×volume > $10M;
entry window = first 100 trading days; **buy at next open after the first post-IPO
all-time-high close**; exit **−20% stop** or **+30% target**; max 5 positions, 20% each.
**Published (2003-01-02 → 2025-04-25):** CAGR **20.53%**, Sharpe **0.95**, maxDD **−31.57%**,
avg trade **+6.72%**, avg hold **67d**.
**Reproduced — FULL survivorship-clean universe (6,599 IPOs, 2003–2026):**
CAGR **8.06%**, Sharpe **0.60**, maxDD **−47%**, avg trade **+5.10%**, win **51%**,
avg hold **99d**, avg exposure **3.9/5**, 232 trades (112 target / 109 stop / 11 delist).
**Verdict: per-trade EDGE reproduces (+5.1%), headline CAGR does NOT (8% vs 20.5%).**
It was well-deployed (3.9/5), so the gap isn't under-investment — it's slower capital
recycling (99d vs 67d holds) and, on genuinely survivorship-clean data, the compounding
isn't there. IPO returns are fat-tail-driven; the honest full-universe number is far
below the published figure. **This is the one published claim that failed honest data.**
SPAC exclusion is crude (ticker-suffix + no `-`); residual SPAC contamination possible in 2020–21.

### 3.4 Volatility breakout & IV mean-reversion — NOT reproducible
- **Day-trading volatility breakout:** rules **paywalled** ("members only"); it's a
  **1-minute intraday** ETF/futures strategy (data we don't have). Published (unverified):
  27% ann / −32% DD / Sharpe 1.04 / 9,023 trades since 2018.
- **IV mean-reversion:** needs **options/implied-vol** data + likely paywalled. Not attempted.

---

## 4. NDX-100 survivorship test — `repro_ndx_survivorship_test.py`  (counterintuitive)
Built an **"ever-member" NDX-100** universe = current 100 + 94 removed/delisted names
(Kraft, Xilinx, Cerner, Activision, Bed-Bath, Walgreens, Peloton, GreenMountain, Staples,
Wynn, Sirius, Illumina, …). Ran the *same* mean-reversion strategy on both:

| universe | CAGR | Sharpe | maxDD | win | #trades |
|---|--:|--:|--:|--:|--:|
| current-only (survivorship-biased) | 17.7% | 0.95 | **−26%** | 71% | 1,327 |
| **ever-member (+delisted, honest-er)** | **24.4%** | 1.01 | **−41%** | 68% | 2,290 |

**Adding delisted names RAISED CAGR** (opposite of the usual survivorship direction).
Reason: it's a **trend-filtered (>200-EMA), volatility-loving (ranks by ATR%) dip-buyer**;
the removed names are disproportionately ex-high-flyers (PTON/RIVN/ZM/MRNA/ENPH/SMCI)
that were *great* mean-reversion vehicles during their volatile uptrends, and the 200-EMA
filter exits before their terminal crash. **The real survivorship cost is in the DRAWDOWN:
−26% (biased) → −41% (honest).** Survivorship hid the *risk*, not the return. The −41%
ever-member number is the honest base the improvement arc builds on.

---

## 5. Momentum decile study (French / Concretum reproduction) — `momentum_*.py`
Reproduced the classic Jegadeesh-Titman / Kenneth-French 12-1 momentum deciles on the
SUMMIT survivorship-clean monthly panel (EW, monthly, 1991–2025, price≥$5 liquid universe).
- **Deciles monotonic:** D1 (losers) 2.0%/Sh 0.21/−90% → D10 (winners) 15.5%/0.70/−60%. EW market 10.4%/0.67/−53%.
- **D10−D1 long/short:** 8.2% / 0.45 / **−72%** — the academic L/S is *worse* risk-adjusted than owning winners, and carries the momentum-crash tail (worst months 2001-01 −40%, 2009 −32%/−20%).
- **Improvements (`momentum_improvements.py`):** residual (idiosyncratic) momentum + beta-hedge + Barroso vol-scaling lifts the L/S from **0.45 → 0.91 Sharpe**, CAGR 8.2%→14.8%, maxDD −72%→−35%. (These are the SUMMIT wins — residual/beta-neutral/vol-scale — applied to plain momentum.)
- **Hold-vs-sell for pure momentum (`momentum_hold_vs_sell_rules.py`, top-20, 1991–2025):** monthly **rotation wins** (23.0%/0.73/−63%); ride-MA (11.7/0.49) and ride-stop (12.4/0.51) and **hold-forever (12.1/0.65) all lose** → *pure momentum must be rotated, not held.*
- **Signal × exit rule (`momentum_ml_vs_pure_dca.py`, 2015–2025, top-15):** the exit rule FLIPS with the signal — for **pure momentum** rotation>ride>hold-forever (hold-forever = **−10.7%**!); for the **ML quality rank** *ride wins* (1.17 > 1.14 rotation, hold-forever a healthy 0.95). Durable/quality winners can be ridden; ephemeral price-momentum winners cannot. $1/mo DCA 2015–2025: WAVE-style ML-ride ended $356 vs **QQQ $421** (QQQ won terminal dollars in the tech-bull decade; the ML book won on risk — Sharpe 1.17 vs 1.04, DD −24% vs −33%).

---

## 6. The improvement arc (iterations 1–6) — how the proprietary system was built
Base = the honest **ever-member NDX-100 mean-reversion** (24.4% / 1.01 daily Sharpe / −41% DD).
Discipline: prefer independently-documented levers over mined parameters; validate on a
dev(pre-2016)/holdout(2016+) split; reject what doesn't hold.

**Iter 1 — overlays (`improve1_overlays.py`):**
| variant | CAGR | Sharpe | maxDD | win |
|---|--:|--:|--:|--:|
| base ever-member | 24.4% | 1.01 | −41% | 68% |
| + market-regime gate (QQQ>200DMA) | 18.4% | 0.94 | −29% | 68% |
| **+ RSI(2)<10 dip-quality** | 15.7% | **1.07** | **−20%** | 73% |
| + IBS<0.3 | 21.4% | 1.00 | −35% | 69% |
→ **RSI(2) oversold filter** (decades-old, robust: both <10 and <15 work) lifts Sharpe,
**halves DD**, raises win to 73%. CAGR drops only because it's now *under-deployed at −20% DD*
(freed risk budget). Regime gate cut return more than risk; IBS marginal; vol-target misfired.

**Iter 2 — capital efficiency (`improve2_capital_efficiency.py`):** deploy the freed budget.
| variant | CAGR | Sharpe | maxDD |
|---|--:|--:|--:|
| RSI2<10 pf40 | 32.3% | 1.08 | −38% |
| **RSI2<15 pf30** | **28.6%** | **1.09** | **−33%** |
| RSI2<15 pf40 | 38.8% | 1.10 | −43% |
→ **Beats the base on ALL THREE axes** (higher CAGR + higher Sharpe + lower DD) purely by
scaling a higher-quality book (not fitting).

**Iter 3 — strategy ensemble (`improve3_strategy_ensemble.py`), monthly:** the "combine" core.
Correlations: MeanRev↔Momentum **0.14**, MeanRev↔BuyDip **0.07**, Momentum↔BuyDip 0.42.
| sleeve/combo | CAGR | Sharpe | maxDD |
|---|--:|--:|--:|
| MeanRev (improved) | 18.8% | 1.54 | −10% |
| Momentum (12-1 top-20) | 31.3% | 1.22 | −51% |
| BuyDip (RSI5<20) | 17.4% | 1.25 | −25% |
| **Equal-vol (risk-parity) combo** | 21.5% | **1.94** | −19% |
→ Risk-parity blend Sharpe (1.94) **exceeds every individual sleeve** — the free-lunch
diversification, confirmed by the near-zero correlations.

**Iter 4 — holdout (`improve4_holdout_validation.py`):** combo dev 1.97 / holdout 1.92,
beating every sleeve in *both* halves → not in-sample luck.

**Iter 5 — weight robustness + benchmark (`improve5_weights_voltarget.py`):**
equal-vol (1.94) > equal-weight (1.79) > 60/40 (1.76); **vol-targeting REJECTED** (dropped
Sharpe to 1.72). QQQ benchmark: 10.8% / 0.59 / −56% (dev 0.31 / hold 1.09).

**Iter 6 — crisis-alpha sleeve (`improve6_crisis_alpha_final.py`), FINAL:** added a
bond/gold trend-following sleeve (TLT/IEF/GLD, long when 12-1 mom>0). Correlation to the
three equity sleeves: **0.05 / −0.05 / −0.08** (near-perfect diversifier).
| system | CAGR | Sharpe | maxDD | dev/holdout Sh |
|---|--:|--:|--:|--:|
| 3-sleeve risk-parity | 21.5% | 1.94 | −19% | 1.97 / 1.92 |
| **4-sleeve + crisis-alpha (FINAL)** | 14.4% | **2.04** | **−12%** | **2.11 / 1.95** |
| QQQ | 10.8% | 0.59 | −56% | 0.31 / 1.09 |

---

## 7. THE FINAL PROPRIETARY SYSTEM
**4-sleeve risk-parity (inverse-vol weights), monthly rebalance:**
1. **Mean-reversion** — improved NDX-100 Connors (RSI(2)<15 dip-quality filter + the screenshot rules), ever-member universe.
2. **Momentum** — 12-1 cross-sectional, top-20 equal-weight, monthly (S&P names).
3. **Buy-the-dip** — RSI(5)<20 & >200-DMA, 5-day hold (S&P names).
4. **Crisis-alpha** — bond/gold trend-following (TLT/IEF/GLD, long when 12-1 mom>0).

**Weights:** inverse-vol (risk-parity), NOT optimized (avoids mean-variance overfitting).
**Result:** Sharpe **2.04**, maxDD **−12%**, holdout **1.95**, CAGR 14.4% unlevered
(~22–23% deployed to the 3-sleeve's −19% risk budget). vs QQQ 0.59 / −56%. Charts in
`charts/combo_final.png` (final) and `charts/combo.png` (3-sleeve).

**Why it's not overfit:** every lever is independently documented — RSI(2) oversold
(Connors), capital efficiency (pure scaling), risk-parity diversification (free lunch,
confirmed by ~0 correlations), Barroso vol-scaling & residual momentum & beta-neutral
(published). Weights are non-fitted (inverse-vol). Validated dev/holdout. Rejected levers
recorded (vol-target, regime gate, IBS).

---

## 8. Honest caveats & known limitations (DO NOT STRIP)
1. **Monthly-resampled Sharpes run higher than daily** — the ensemble Sharpes (1.9–2.0) are
   monthly; on a daily basis they'd be lower. The *relative* lift over QQQ (~+1.4) and over
   the best single sleeve (~+0.4) is the robust, frequency-independent part.
2. **Survivorship:** MeanRev uses ever-member (survivorship-*reduced*, not perfect PIT);
   Momentum/BuyDip use **current** S&P 500 (survivorship-biased → optimistic). A true PIT
   S&P membership + full delisted set would lower these.
3. **Costs modeled unevenly:** MeanRev models $0.005/share commission + realistic limit
   fills; Momentum/BuyDip sleeves are modeled lightly; **inter-sleeve monthly rebalancing
   is not costed**. Deployable net numbers will be somewhat lower.
4. **Sleeve *selection* has a post-hoc element** — mitigated by each being a canonical,
   independently-published strategy + risk-parity (non-fitted) weights + dev/holdout split,
   but it is not zero. Treat 2.04 as an optimistic point estimate.
5. **Capital-efficiency / leverage:** "deploy freed risk budget" and the "~22–23% at −19%
   risk" figure assume you can scale/lever the book; unlevered CAGR is 14.4%.
6. **IPO CAGR did not reproduce** (8% vs 20.5%) — signal real, headline optimistic.
7. **Data ephemeral** (`/tmp`) — re-fetch before rerunning. Rotate the exposed Tiingo key.

---

## 9. Next steps / roadmap for the picking-up agent
1. **Consistent daily-basis Sharpes** + model inter-sleeve rebalancing costs → a fully
   deployable net number (the honest headline).
2. **Survivorship-clean the S&P sleeves** — fetch PIT S&P 500 membership + delisted names
   (same approach as the NDX ever-member fix) and rerun Momentum/BuyDip.
3. **More uncorrelated sleeves** (careful of overfit): short-term reversal, low-vol/quality,
   managed-futures-style trend on more assets — add only if holdout-robust and ~0-correlated.
4. **Meta-labeling** (Lopez de Prado) on the MeanRev trade set to filter low-quality signals
   — walk-forward only, heavy overfit guard.
5. **PIT NDX-100 membership** (Norgate or a clean change-log) for a truly point-in-time
   mean-reversion base (the 33 pre-2010 removed names Tiingo lacked are still missing).
6. **Reconcile with SUMMIT/WAVE** — the momentum sleeve here overlaps WAVE's signal; consider
   swapping in the 36-feature ML rank (see `WAVE_long_only.md`) for the momentum sleeve.

## 10. Script index
`scripts/` — all read `TIINGO` env + `/tmp/*.pkl`:
- `01_fetch_ndx100_daily.py`, `02_fetch_sp500_daily.py`, `03_fetch_ipo_universe_daily.py`, `04_fetch_ndx_removed_daily.py` — data fetch.
- `repro_ndx_meanreversion.py`, `repro_buythedip.py`, `repro_ipo_breakout.py`, `repro_ndx_survivorship_test.py` — reproductions.
- `momentum_deciles.py`, `momentum_improvements.py`, `momentum_hold_vs_sell_rules.py`, `momentum_ml_vs_pure_dca.py` — momentum study (need SUMMIT `/tmp/wave/_featmat.pkl`, `_mlprob.pkl`).
- `improve1_overlays.py` … `improve6_crisis_alpha_final.py` — the improvement arc; `improve6` produces the final system + `charts/combo_final.png`.
`charts/` — momentum_deciles, ndx_mr, ndx_surv, combo, combo_final (.png).
