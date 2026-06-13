# SUMMIT — Biweekly DCA Stock-Selection Strategy

*Final methodology, validation results, and live execution spec.*
*Research session 2026-06-12. Code: `dca/strategy_dca.py`; engines and
protocol in `dca/`; full research trail in `dca/research/`.*

---

## 1. What it is

Every two weeks, a fixed contribution buys **2 stocks** from the S&P 500,
selected by a regime-switched rule. **Nothing is ever sold.** Lots
accumulate for as long as the account lives.

* **RISK-ON** (SPY ≥ 200-day MA and >40% of index members above their own
  200-day MA): buy the top 2 by `rank(multi-horizon momentum) + 5 ×
  rank(dollar volume)` — i.e. *the largest stocks that are also leading*.
  Momentum = sum of return ranks over 63/126/189/252 trading days, each
  skipping the most recent 21 days.
* **RISK-OFF** (SPY < 200dma OR breadth < 40%): keep buying — but buy
  `rank(discount depth) + 1 × rank(dollar volume)` among **quality
  rebounders**: names whose long-term uptrend is intact (above 400-day MA
  or positive 24-month return) trading **30–60% below their all-time
  high**. *Buy the index's giants while they're on sale.*
* Execution: all signals computed at a close; all buys at the **next
  session's open**. No sells, no stops, no recovery triggers (each was
  tested and made things worse — §5).

### Why this works (rationale)

1. **The benchmark is cap-weighted; naive stock-picking is not.** Most of
   QQQ/SPY's return comes from a handful of giant winners. An equal-weight
   picker that ignores size systematically lags cap-weighted indexes in
   mega-cap-led eras (2010-13, 2017, 2020-21, 2023-26). The size tilt
   closes that structural gap; momentum then adds selection alpha on top.
   Both ingredients were independently necessary in testing: pure momentum
   plateaus at ~60% window win-rate vs QQQ, pure size at ~77-81%; the
   product reaches 93%.
2. **Momentum is regime-conditional** (own-EDA finding: 6m-momentum rank-IC
   is +0.024 above the 200dma, −0.054 below it; deep-drawdown names hit
   17% parabolic rate below the 200dma vs 3% base). So the rule switches
   *what it buys*, not *whether it buys*: DCA's bear-market contributions
   are its cheapest, highest-payoff lots — pausing them (trend-gate) and
   panic-selling both tested worse than buying discounted quality.
3. **Never selling** turns the portfolio into an accumulation of
   era-winners bought either while leading or while cheap, with zero
   turnover cost and no crystallized momentum-crash losses. What the
   strategy actually bought, mechanically and causally: AAPL throughout
   2006-2017, GS/MA in the GFC trough, NFLX 2010-13, NVDA from 2014, AMD/
   TSLA 2018-21, META/NVDA/PLTR 2022-26.

---

## 2. Validation summary (all on clean PIT panel)

Setup: point-in-time S&P 500 membership (fja05680), 730 historical
constituents with prices, delisting-aware engine, 5 bps/trade, biweekly
(10 trading days), k=2, signals at close → execution next open.
Benchmarks: identical-cadence DCA into QQQ and SPY.

**Window grid: 244 windows = quarterly start dates 2006→2023 × horizons
{3y, 5y, 10y, full}.**

| horizon | n | win vs QQQ | win vs SPY | median vs QQQ | worst vs QQQ | worst vs SPY |
|---|---|---|---|---|---|---|
| 3y   | 70 | 84%  | 91%  | +10.0% | -10.6% | -11.7% |
| 5y   | 62 | 92%  | 100% | +19.1% | -7.4%  | +1.4%  |
| 10y  | 42 | 100% | 100% | +31.6% | +8.4%  | +31.7% |
| full | 70 | 100% | 100% | +120%  | +45.9% | +63.1% |
| **all** | **244** | **93%** | **98%** | **+28.8%** | **-10.6%** | **-11.7%** |

**Regime windows — all positive vs both benchmarks:**

| regime | money multiple | vs QQQ | vs SPY |
|---|---|---|---|
| GFC 2007-2009 | 1.28 | +9.0% | +21.0% |
| recovery 2009-2012 | 1.43 | +6.8% | +12.1% |
| bull 2013-2017 | 2.03 | +26.9% | +43.2% |
| sideways 2015-2016 | 1.14 | +3.8% | +3.2% |
| vol 2018 - mid-2019 | 1.11 | +1.4% | +2.8% |
| COVID 2020-2021 | 1.58 | +13.8% | +19.0% |
| bear 2022 - mid-2023 | 1.41 | +17.6% | +27.9% |
| AI bull 2023-2026 | 2.40 | +49.3% | +69.3% |

**Full period (2006-01 → 2026-06, $1,000 biweekly):** $515k contributed →
**$10.31M (20.0x money multiple, 24.7% IRR)** vs QQQ $4.69M (9.1x, ~19%)
and SPY $2.44M (4.7x, ~14%).

**Where it still loses to QQQ** (16/244 windows, worst -10.6%): 3-5y
windows starting 2010-2013 (peak AAPL-era QQQ, when Apple alone was
~15-20% of QQQ) plus the 2006Q1 3y window; 11 of those 16 still beat SPY.
No losing window after 2016Q4 starts.

### Anti-overfitting / honesty checks

| check | result |
|---|---|
| Leakage audit (panel hard-truncated at 6 random dates, scores recomputed) | max diff = 0.0 — fully causal |
| Reference engine vs fast engine | identical to 4 decimals on 3 windows |
| Schedule offset (all 10 phases of the biweekly grid) | win_qqq 93-95%, no phase dependence |
| Costs 5 → 40 bps | results unchanged (buy-once, hold-forever) |
| k = 1 / 2 / 3 / 4 / 5 | 95% / 93% / 91% / 84% / 82% win_qqq — graceful, concentration is the point |
| Biweekly vs monthly cadence | equivalent (93% vs 93%) |
| Bull size-weight 3-8, bear size-weight 0.5-2 (full cross) | 89-94% win_qqq everywhere — mid-plateau (5, 1) chosen, not argmax |
| Momentum formation variants (2-horizon, 12-1 only) | 87-91% — not formation-sensitive |
| Split-sample | starts 2006-2014: 90% win_qqq, median +21.5%; starts 2015-2023: 99%, +49.2% |
| NASDAQ-100 PIT universe transfer (2015+, own membership data) | beats QQQ DCA and the random control at all 15 start dates (e.g. 2016 start: 8.6x vs QQQ 3.3x vs random 3.4x) |
| Random-pick control (same universe, same engine) | random beats QQQ in 8% of windows, median -15% — universe carries no QQQ edge; SUMMIT's edge is selection |

### Survivorship & data caveats (read before believing)

* Free (Yahoo) data lacks ~26% of historical constituents, concentrated in
  pre-2015 delistings (2005 coverage 57% → 99% today). Mitigations: PIT
  membership masks (no name selectable before/after its membership), a
  random-pick control carrying the same bias (≈ SPY, far below SUMMIT),
  and a mega-cap tilt that concentrates picks in the best-covered corner
  of the universe. Residual upward bias cannot be fully excluded,
  especially in the bear sleeve (distressed names); the repo's
  conventional response is a ~3% CAGR mental haircut on stock-picking
  results — SUMMIT's margins exceed that comfortably.
* ~14 delisted tickers had corrupted Yahoo records (garbage 1000x spikes);
  found and repaired *before* final results (pre-fix numbers were inflated
  by up to +14pp win-rate — treat any uncleaned Yahoo backtest with deep
  suspicion).
* Pre-2004 history (dot-com crash) is not in the panel; the strategy's
  2000-02 behavior is untested. The GFC is the closest available analog.
* Yahoo's adjusted prices embed dividend reinvestment for stocks;
  benchmark ETF files are likewise adjusted — consistent treatment.

### What didn't work (documented negatives)

* Walk-forward LightGBM ranker (20 features): OOS IC ≈ 0.002 (t=0.5),
  learns defensive beta/vol tilts, loses to a single momentum column.
* Chronos-bolt re-ranking of momentum candidates: worse than the matched
  momentum control at every k (-15 to -22pp win-rate).
* Vol-compression/breakout family: negative alpha, monotone in the dose.
* Volume/accumulation: veto-grade at best; redundant after the size tilt.
* Low-vol, anti-lottery (MAX), Sharpe-scaled momentum: fatal vs QQQ.
* Sell rules (HY-OAS panic exit, 300dma stops): worst window -24.5% →
  -62.6% (panic) — never sell.
* Recovery triggers (breadth thrust, VIX relax, 50dma reclaim): GFC
  -3% → -20% (VIX trigger re-enters momentum during bear rallies).
* 52-week-high proximity, trend-smoothness, momentum-acceleration: dead.

---

## 3. Live execution spec (exact rules)

Parameters: `k=2`, cadence = every 10 trading days, all values from
`dca/strategy_dca.py` (BULL_HORIZONS=(63,126,189,252), SKIP=21,
SIZE_WIN=63, W_SIZE_BULL=5, W_SIZE_BEAR=1, DD band -60%..-30%,
BREADTH_FLOOR=0.40).

Every second Friday after the close (any consistent biweekly grid works —
validated across all 10 phases):

1. **Universe**: current S&P 500 members with ≥252 trading days of history.
2. **Regime**: risk-off if SPY close < SPY 200-day MA, OR the 10-day mean
   of (share of members above their own 200-day MA) < 0.40. Else risk-on.
3. **Score** (cross-sectional percentile ranks over the universe):
   * risk-on: `rank(Σ_h rank(close[t-21]/close[t-h] - 1, h ∈ {63,126,189,252}))
     + 5 × rank(mean₆₃(close×volume))`
   * risk-off: among names with (close > 400dma OR 24-month return > 0)
     AND drawdown from all-time high in [-60%, -30%]:
     `rank(-drawdown) + 1 × rank(mean₆₃(close×volume))`
4. **Buy** the top 2 names at Monday's open, half the contribution each
   (plus any cash from rare delisting proceeds). Market or
   open-auction orders; 5-40 bps slippage is immaterial.
5. **Never sell.** Holdings ride through everything, including index
   deletion of a held name (delisting cash recycles into the next buy).
6. Helper: `python -c "import sys; sys.path.append('dca');
   import strategy_dca; print(strategy_dca.current_picks(2))"`

Variants validated: k=1 (max aggression, 95% win_qqq, +43% median, fatter
single-name risk), k=3 (diversified, 91%, +22%), monthly cadence
(equivalent to biweekly at every k).

Current signal (2026-06-12 close): RISK-ON; picks = **MU, SNDK**.

---

## 4. Reproduction

```bash
pip install -r requirements.txt
python dca/download_pit_universe.py        # ~30 min, Yahoo
python -c "import sys; sys.path.append('dca'); import data; data.build_panel(force=True)"
python - <<'PY'
import sys; sys.path.append('dca')
import data, protocol, strategy_dca
P = data.build_panel()
protocol.evaluate_signal(strategy_dca.build_scores(P), "SUMMIT_k2", k=2)
PY
# full battery:
python -c "import sys; sys.path.append('dca'); import validate_final, strategy_dca;
validate_final.validate(strategy_dca.build_scores, 'SUMMIT', k=2)"
```
