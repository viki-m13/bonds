# Honest improvement sweep for SUMMIT (PIT S&P 500)

Goal: try sector caps, more technical analysis, risk-adjusted (Sharpe-style)
selection, and creative blends — and report honestly what helps, guarding
against overfitting (every candidate scored on the full 244-window grid; the
promising ones re-checked for phase robustness and in-sample vs out-of-sample).

Baseline SUMMIT (biweekly k=2, 5 bps): **win QQQ 93%, median +28.8%, worst
−10.6%, 20.0×.**

## 1. Technical / risk-adjusted selection sweep (19 variants)

Each variant changes only the risk-on score; regime + bear sleeve unchanged.

**What HURT (clear, consistent):**
* **Sharpe-ratio selection** (return/vol, 126d/252d): 85-88% win, median +15-20%.
  Risk-adjusting the momentum demotes the highest-octane winners. ✗
* **Volatility-adjusted momentum / momentum − vol penalty**: 75-89% win. ✗
* **Mean-reversion "fair-value z vs SPY"** (the example signal) as a selector:
  63% win, −43% worst — it's contrarian, the opposite of what works here. As a
  *blend* into momentum it still added tail risk (−43% worst). ✗
* **Anti-lottery (MAX) filter**: median +16.5% (vs +28.8%). ✗
* **RSI-pullback tilt** (prefer oversold): 91% win, +23.5%. ✗

**What was ~neutral / marginally positive (within noise):**
* **Skip extremely overbought (RSI ≥ 80) gate**: 94% win / +29.3% median /
  −11.0% worst. Robust across phases and slightly *better* out-of-sample
  (OOS win 100%, median +49.3%). A sensible, tiny filter — keep-or-leave.
* **Mild trend-quality tilt** (+0.5·rank(% days above 50dma)): 94% win,
  median +26.1%, **worst −8.8%**. Robustness check: better worst in BOTH eras
  (IS −8.8%, OOS +0.5%) and higher OOS median (+52.8%), at the cost of ~1×
  full multiple (18.6× vs 20.0×) and ~2pp lower in-sample median. The one
  tweak that genuinely improves the tail.

**Overfit trap caught:** `mom_12_1_only` (single 12-1 momentum vs the
multi-horizon blend) looked best on the aggregate grid (+30.4% median, −7.8%
worst) — but the IS/OOS split showed it is *worse out-of-sample* (OOS median
+42.8% vs baseline +48.8%). It overfits the in-sample tail; rejected. The
multi-horizon blend is more robust OOS, which is why it's the live default.

## 2. Sector caps

Sectors = current GICS (yfinance proxy; 659/720 covered). SUMMIT's book is
~77% Technology (it rode NVDA/AAPL).

| config | win QQQ | median | worst | full mult | top sector |
|---|---|---|---|---|---|
| baseline | 93% | +28.8% | −10.6% | 20.0× | Tech 77% |
| force 2 picks ≠ sector | 93% | +25.8% | −9.6% | 19.2× | Tech 74% |
| sector cap 50% / yr | 94% | +27.6% | −10.9% | 20.4× | Tech 62% |
| sector cap 40% / yr | 93% | +25.8% | −12.4% | 18.1× | Tech 54% |
| sector cap 30% / yr | 92% | +23.5% | −13.7% | 16.0× | Tech 46% |
| diversify + cap 40% | 93% | +25.2% | **−8.6%** | 18.2× | Tech 53% |

* **Hard sector caps hurt, monotonically.** Tightening the tech cap (50→40→30%)
  lowers median, *worsens* the worst case, and cuts the full multiple — because
  tech led 2009-2026, so forcing out of it sacrifices the winners.
* **A loose 50% cap is near-free** (94% win, +27.6%, 20.4×) — it pulls tech
  from 77% to 62% with essentially no performance cost. Like the single-name
  trim, it's a free *risk-reduction* lever, not a return improver.
* **Forcing two distinct sectors per buy barely diversifies** (tech still 74%)
  because momentum picks tech so consistently that the #1 pick is tech almost
  every period; it just shaves median.

## Overall honest conclusion

SUMMIT sits at a **robust local optimum**. Across this sweep — and the earlier
universe (Russell-1000/mid-cap), ETF, cadence, rebalance and trim studies — the
recurring result is the same: **the strategy's returns come from concentrated
mega-cap momentum, and almost everything that dilutes or "improves" on that
either does nothing (the size tilt keeps it in the leaders) or trades return for
diversification.**

* No tested signal (Sharpe, vol-adjustment, mean-reversion, RSI, MAX, sector
  diversification) robustly raises the win-rate or median.
* The only genuinely *free* knobs are **loose concentration caps** — single-name
  trim ≤33% (already on the page) and a ~50% sector cap — which cut concentration
  risk at ~no cost for investors who want it.
* The only tweak that mildly, robustly improves the **tail** is a small
  **trend-quality tilt** (better worst-case IS and OOS), at the cost of a little
  terminal multiple — a defensible option if reducing the worst 3-year window
  matters more than the headline multiple.

I did not find a change that honestly beats the live configuration on the
metrics that matter (cross-window win-rate and median, robust across phases and
OOS). That is itself a useful, honest result: the design is hard to improve
without giving something up.
