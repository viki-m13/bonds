# Validation: the "fair-value z-score" mean-reversion scripts

**Question asked:** validate the submitted strategies the way SUMMIT was validated
(PIT universe, anti-leakage, real benchmark, OOS) and determine whether any of
them are correct, accurate, and as profitable as they claim.

**Verdict: No.** All of the submitted scripts are variants of one signal, and
none of them is a validated, profitable strategy. The apparent edge is an
artifact of (1) survivorship bias, (2) a forward-return function that silently
deletes losers, (3) massively overlapping samples, and — in the "test" /
"selective" variants — (4) look-ahead leakage in the calibration. On a proper
point-in-time S&P 500 universe the signal **underperforms buying on a random
day.** The intraday (1h / 5-min) variants cannot establish multi-year
profitability at all because the data does not exist.

Reproduce: `python3 dca/research/validate_user_strategies.py`
(PIT panel: 720 names, 2004–2026, 502 current members + 218 delisted, SPY market).

---

## 0. What the scripts actually are

Eight scripts, but one strategy. Per ticker, vs SPY:

- `z` = residual of a rolling OLS `log(stock) ~ log(SPY)` (5y beta, 2y resid-std)
- **buy gate** = `z ≤ −2.75` **and** weekly `RSI(14) ≤ 40` **and** drawdown from
  the 252-day high `≤ −30%` **and** 20-day $-volume `≥ $25M`.

"Profitability" is asserted from a *forward-return table*: for horizons of 252 /
756 / 1260 trading days, compare `price.shift(-n)/price − 1` on signal days vs
all other days. The variants only change the **universe** (hand-picked mega-caps
→ live iShares S&P 500 holdings), the **bar size** (daily → 1h → 5m), or add
**extra gates** (regime / "edge" / "strength" calibration).

This is not a backtest. There is no portfolio, no position sizing, no execution
model, no transaction costs, and no benchmark return. "P(gain) of one stock over
5 years" is ~80–90% for *any* large-cap survivor regardless of the signal, so the
only thing that matters is the **delta vs the non-signal baseline** — which the
scripts print but the narrative ignores.

---

## 1. The headline result (PIT, pooled signal-day vs non-signal-day forward returns)

Δ = signal − non-signal. Positive Δ = the signal helps.

| Test | universe | 1Y ΔP(gain) / Δavg | 3Y ΔP(gain) / Δavg | 5Y ΔP(gain) / Δavg |
|---|---|---|---|---|
| **A** hand-picked mega-caps | survivors + biggest winners | **+7.1% / +22.2%** | −7.8% / +8.1% | −0.4% / +67.3% |
| **B** S&P survivors only, no mask | iShares-holdings style | +8.1% / +31.0% | +3.0% / +44.5% | +1.1% / +53.8% |
| **C** PIT, membership mask, drop delistings | correct universe, *their* return fn | +3.7% / +22.7% | **−4.8% / +8.6%** | **−7.4% / +0.4%** |
| **D** PIT + delisting-aware (hold to last price) | correct universe, honest returns | +0.5% / +17.5% | **−10.7% / −6.6%** | **−18.0% / −37.3%** |

Reading down each column, the edge **monotonically collapses** as you remove
bias. By the honest measure (row D), the signal is a strong **anti-signal** at
multi-year horizons: stocks bought after a deep, oversold, "cheap-vs-SPY"
drawdown go on to gain **+68%** over 5Y vs **+106%** for a random buy day, and
win only **63%** of the time vs **81%**. The strategy systematically catches
falling knives.

The lingering positive *average* in rows A–C is the classic deep-value mirage:
deep-drawdown names have fat right tails (the few that round-trip to new highs),
which lifts the mean while the median/win-rate is already negative. Row D shows
even the mean turns sharply negative once delisted names are not deleted.

---

## 2. The four reasons the claim is wrong

### 2.1 Survivorship bias (the big one)
The hand-picked list (`AAPL, NVDA, AVGO, LLY, …`) and the live-iShares-holdings
scripts both test **today's winners**. "Buy the dips on the stocks that we now
know became the largest companies in the world" is guaranteed to look good. Test
A→C quantifies it: the 5Y win-rate edge goes from roughly break-even on the
mega-caps to **−7.4%** on the full PIT universe (which includes the 218 names
that fell out of the index). SUMMIT was built on exactly this discipline — real
historical membership, delisting-aware accounting, and a random-pick control —
*because* this bias is decisive. None of the submitted scripts have it.

### 2.2 `forward_return` silently deletes the losers
`price.shift(-n)/price − 1` is `NaN` whenever the stock stops trading before the
horizon (delisting, bankruptcy, acquisition) — so those rows are dropped from the
stats entirely. On the PIT universe this discards **3,740 / 9,047 / 13,147**
signal outcomes at 1Y/3Y/5Y. These are disproportionately the disasters.
Counting them honestly (row D, "hold to last traded price") swings the 5Y average
edge by **−37 percentage points** (+0.4% → −37.3%). This bug exists in *every*
variant, independent of universe.

### 2.3 Overlapping samples → fake significance
The reported `n` counts every signal *day*. A single drawdown episode fires the
gate on dozens of consecutive days, and each 5Y row reuses ~1,260 overlapping
forward windows. On the PIT universe: **27,940 raw signal-days collapse to ~2,239
distinct episodes** (≥~1 month apart). The true independent sample is ~12× smaller
than advertised; the t-stats implied by `n=27,940` are meaningless.

### 2.4 Look-ahead leakage in the "test" and "selective" variants
These two scripts add a self-calibrating gate that *manufactures* the win-rate:

- **`#ishares 1h test`** — `USE_EDGE_GATE`: `exp_winrate = cum_w / cum_n` where
  `cum_w` cumulates `win = (forward_return(s, 63d) > 0)`. At decision time `t`, a
  signal that fired within the last 63 days contributes a `win` computed from a
  price **after** `t`. The gate "only fire if this setup is winning ≥70%
  lately" is therefore reading the future. It would fail the truncation audit
  (`audit.py`): truncating the panel at `T` changes the gate near `T`.
- **`#ishares 1h selective`** — the global and per-ticker **strength thresholds**
  are chosen by `strength.quantile(q)` and Wilson-LB win-rate over the **entire
  sample**, then applied to that same sample. That is train-on-test. The reported
  "selective" P(gain) is in-sample-optimized and cannot be believed.

Both also inherit 2.1–2.3.

---

## 3. The intraday (1h / 5-min) variants cannot test what they print

`yfinance` intraday history is short. Measured today:

| interval | bars available | calendar span | 1Y horizon | 3Y | 5Y |
|---|---|---|---|---|---|
| `1h` | 3,466 | **728 days** | borderline (~1 independent year) | **all NaN** | **all NaN** |
| `5m` | 3,276 | **58 days** | **all NaN** | **all NaN** | **all NaN** |

The "1Y/3Y/5Y forward stats" the hourly script prints are computed on essentially
no data (3Y/5Y horizons need 5,292 / 8,820 bars; only 3,466 exist). The 5-minute
script has ~2 months of history — *every* multi-year horizon is 100% `NaN`. You
cannot conclude anything about multi-year profitability from two months of
5-minute bars. (The weekly-`RSI(14)` gate is also mostly `NaN` on 60 days of
data, so on 5m the gate degenerates to `z & drawdown` only.)

Minor correctness issues compound this: prices are `ffill`-ed across halts
(fabricating flat bars and fake z-scores), and crypto (`BTC/ETH/FET-USD`) is run
through a SPY-beta regression where the "fair value" residual is economically
meaningless and the 24-bars/day assumption is arbitrary.

---

## 4. How this differs from SUMMIT (the bar these never clear)

| dimension | SUMMIT | submitted scripts |
|---|---|---|
| universe | point-in-time S&P 500 membership, delisting-aware | today's holdings / hand-picked winners |
| survivorship control | random-pick DCA from same eligible set (beats it 93% vs ~8%) | none |
| leakage | truncation audit, bit-identical, zero diff | "test"/"selective" variants leak the future |
| execution | next-open, 5–40 bps cost swept | none (no trades modeled) |
| benchmark | money-multiple vs QQQ/SPY DCA | none (only single-name P(gain)) |
| robustness | 244 windows × offset/cost/k/cadence + NASDAQ-100 transfer + IS/OOS | single in-sample pass |
| sample | independent rolling windows | overlapping daily windows, n inflated ~12× |

---

## 5. Bottom line

- **Correct?** No — `forward_return` deletes losers, the intraday horizons are
  uncomputable, and two variants leak the future.
- **Accurate?** No — the favorable numbers are survivorship + overlap + (for two
  variants) in-sample-fit artifacts.
- **As profitable as claimed?** No — on a proper PIT universe with honest
  delisting accounting, the signal **loses to buying on a random day** at 3Y/5Y
  (−10.7% / −18.0% win-rate, −6.6% / −37.3% average return).

The one mildly interesting fragment is a *short-horizon* (1Y) bounce: a small,
positive win-rate edge survives into the PIT universe (+3.7%) before the
delisting bias is fixed, then evaporates (+0.5% in row D). That is a weak, costs-
and-execution-sensitive mean-reversion bounce, not a multi-year wealth engine —
and it is the opposite of what the scripts advertise. If you want to pursue it,
it must go through the SUMMIT harness: PIT universe, next-open execution with
costs, a real benchmark, and the truncation audit.
