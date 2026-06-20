# Literature & landscape review — signals for concentrated DCA stock selection

(Compiled from the canonical literature plus this project's own experiments;
the original web-research agent stalled, so citations are from memory of the
canonical papers — magnitudes are the commonly reported ones, and every
claim we *acted on* was re-verified empirically on our PIT panel.)

## 1. Cross-sectional momentum
Jegadeesh & Titman (1993): 12-1 momentum (12-month return skipping the last
month) decile spreads ~1%/mo historically. Known properties: skip-month
matters (1-month reversal), crashes violently when the market rebounds from
deep drawdowns (2009: down >50% for winners-minus-losers — Daniel & Moskowitz
2016), and the raw factor has decayed post-2010 in long-short form. Long-only
large-cap implementations retain roughly half the spread. **Our panel
verdict**: the single most useful direction. Best lookback 6-12m with 1-2m
skip; 52-week-high proximity (George & Hwang 2004) and trend-quality scores
underperformed plain momentum badly vs a QQQ benchmark (they tilt low-vol).

## 2. Intermediate / residual momentum
Novy-Marx (2012): months 12-7 carry the signal. Blitz-Huij-Martens (2011):
residual (idiosyncratic) momentum halves crash risk. **Our verdict**:
intermediate momentum was *worse* than 12-1 here; residual momentum vs SPY
was comparable to plain momentum but its edge came partly from an anti-beta
tilt that fails against QQQ. Used only as a cross-check.

## 3. Volatility-managed momentum / crash protection
Barroso & Santa-Clara (2015), Daniel & Moskowitz (2016): scaling momentum by
its own volatility avoids crashes. Market regime filters (200dma) are the
retail-grade equivalent. **Our verdict**: vol-scaling *selection* hurts
(low-vol tilt loses to QQQ everywhere); the regime idea is right but the fix
is to change *what you buy* in bears (see §8), not to stop buying — a DCA
stream that pauses in bears misses the cheapest lots.

## 4. Volatility-compression breakouts (Darvas/Minervini/Bollinger squeeze)
Practitioner lore; thin academic support. **Our verdict (strong, clean
negative)**: every compression formulation — vol ratios, range contraction,
BB-width percentile, ATR contraction, Donchian-with-quiet-vol, base
breakouts — is a *negative* overlay on momentum in S&P large caps at a
biweekly cadence, monotonically in the dose. Both compression AND expansion
tilts hurt symmetrically; short-horizon vol structure carries ~no
cross-sectional alpha here. EDA agrees: parabolic moves come from
high-energy (high-vol, high-beta) names, not coiled springs.

## 5. Volume / accumulation signatures
Gervais, Kaniel & Mingelgrin (2001) high-volume return premium; OBV/Chaikin
accumulation lore. **Our verdict**: every pure volume signal loses to plain
momentum; the only durable use is as a *veto* (demote names under heavy
distribution), worth ~nothing once the size tilt is in. Excluded from final.

## 6. Quality / fundamentals
Piotroski F-score, gross profitability (Novy-Marx 2013). Requires
point-in-time fundamentals we do not have. A price-only "quality" proxy
(long-term uptrend intact) is used inside the bear sleeve. Flagged as future
work with proper PIT fundamental data.

## 7. Low-vol anomaly and the MAX/lottery effect
Baker-Bradley-Wurgler (2011); Bali-Cakici-Whitelaw (2011) MAX effect.
**Our verdict**: actively harmful for this objective. Low-vol and
anti-lottery filters fight the benchmark (QQQ is high-beta growth) and
fight the goal (parabolic winners look "lottery-like" ex ante: top-1%
forward winners sit at the 86th vol percentile and 78th beta percentile).

## 8. What precedes huge runs (extreme-winner studies)
O'Neil/CANSLIM lore and academic extreme-winner work both emphasize: new
highs after bases, big volume, leadership. Our EDA's distinctive finding is
regime-conditionality: **below the market 200dma, deep drawdown-from-ATH
among long-term-healthy names is the best parabolic precursor**
(P(+50% in 6m) = 17% vs 3% base rate); above the 200dma it's worthless and
trend strength + beta dominate. This single empirical fact shapes the final
architecture (momentum sleeve in bulls, discounted-quality sleeve in bears).

## 9. Machine-learned cross-sections
Gu, Kelly & Xiu (2020): trees/NNs lift OOS R² to ~0.3-0.9% monthly on broad
universes (small caps dominate the gains). **Our verdict**: on ~500 large
caps with OHLCV-only features at 6m horizon, walk-forward LightGBM has
mean OOS IC ≈ 0.002 (t≈0.5), learns a defensive beta/vol tilt, and loses
decisively to a single momentum column. Matches the literature's caveat
that ML alpha lives in small/illiquid corners.

## 10. Time-series foundation models (Chronos, TimesFM, PatchTST...)
No credible published evidence of cross-sectional stock-selection alpha.
**Our controlled test**: chronos-bolt re-ranking of the top-30 momentum
candidates *underperforms the momentum control it re-ranks* at every k
(-15 to -22pp window win-rate). Negative, excluded.

## 11. Transaction costs & turnover
Novy-Marx & Velikov (2016): momentum decays fast with costs at monthly
rebalance long-short. A buy-and-never-sell DCA sidesteps this: each lot
pays one half-spread once. Verified: results identical at 5 vs 40bps.

## 12. Survivorship & data integrity
Standard warnings (e.g., Shumway delisting bias) apply doubly to free data.
We found and repaired: recycled tickers (different company under a dead
ticker), garbage spikes on ~14 delisted names (pre-fix backtests were
inflated by up to +14pp win-rate), and partial coverage of pre-2015
delistings (mitigated by PIT membership masks + a random-pick control that
carries the same bias + a mega-cap tilt that concentrates picks where
coverage is ~complete).

## Ranking for this mandate (biweekly, concentrated, hold-forever, vs QQQ/SPY)

1. Multi-horizon momentum with skip-month (core bull selector) — verified.
2. Mega-cap (dollar-volume) tilt — the decisive, under-published ingredient
   vs cap-weighted benchmarks; closes the QQQ concentration gap.
3. Regime switch at SPY 200dma + breadth (when to change sleeves).
4. Discounted-quality rebound buying in bears — the DCA-native crash fix.
5. Residual momentum — diagnostic value only here.
6. Volume veto — marginal, excluded.
7. Compression/breakout, low-vol, anti-lottery — negative here.
8. GBM/foundation-model ranking — negative here.
