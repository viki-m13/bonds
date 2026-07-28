# Corporate strategy — bias & overfitting audit

Professional-grade validation of the corporate dislocation-reversion strategy.
Every check below is reproducible from the committed data
(`corps/data/panel/*.parquet`) and code.

## Summary verdict

| risk | status | how addressed |
|---|---|---|
| Survivorship bias | **controlled** | Data is the full TRACE tape incl. defaulted/called bonds; 70% of bonds stop trading before the sample end and are retained with their final prices. 7–14% of positions exit "stale" (bond stopped trading) and take the realized loss. |
| Selection bias | **controlled** | Universe = **all 55,545 bonds** with ≥20 trading days (99.9% of all bond-days), not a top-N-by-liquidity cut. Removing our earlier "top-8000" shortcut *raised* the excess (+1.38%→+2.06%), confirming the shortcut, if anything, understated — the bias was against us. |
| Look-ahead bias | **controlled** | Trailing median is `.shift(1)` (excludes today); liquidity gate counts trailing-90d dates in `[t−90, t)`; entry is the first ask strictly **after** the signal day; exits use only prices at/before exit. |
| Overfitting (parameters) | **controlled** | Core parameters (60-day window, 3-pt threshold, ~1-yr hold, 90-day/8-day gate) were **transferred verbatim from the muni strategy — zero corporate-specific fitting** — and still produce +2.06% excess. |
| Overfitting (added signals) | **demonstrated** | Two economically-motivated overlays (market-regime gate, per-bond credit filter) were tested and **rejected for failing out-of-sample** (see below). We publish the base, not the in-sample-flattering variant. |
| Data-mining / robustness | **controlled** | Signal is **monotone** in threshold (1/2/3/4 pt) and hold length; IS and OOS both significant on a clean time split. |
| Transaction costs | **modeled** | Fills use the actual daily **bid/ask** from the panel (buy ask, sell bid) — the spread is paid, not assumed away. |

## 1. Survivorship — the data retains dead bonds

The OSBAP panel is built from the complete TRACE tape. Diagnostics on the
55,545-bond universe:

- **70%** of bonds stop trading > 1 year before the sample end (matured,
  called, or defaulted) and remain in the panel with their last prints.
- **8%** ever print below 50 (crash/distress); **4%** have their *final* trade
  below 70 (died distressed, never recovered).
- **7–14%** of strategy positions exit "stale" — the bond stopped trading
  during the hold — and the exit takes the last available bid (a real loss when
  the bond cratered).

A survivorship-biased dataset would show none of this. The strategy's edge is
measured *after* eating these losses.

## 2. Selection — full universe, not a survivor cut

Our first pass pre-filtered to the "top 8,000 bonds by lifetime trading days,"
which favors long-lived survivors. The professional rebuild uses **every bond
with ≥20 trading days** (55,545 bonds, 29.7M bond-days = 99.9% of the data);
tradability is decided **point-in-time** by the trailing-liquidity gate, not by
a bond's full-sample longevity. Result: the excess *rose* (+1.38%→**+2.06%**),
so the earlier cut was conservative, not flattering.

## 3. Overfitting — the transfer test and the rejected overlays

**Transfer test (strongest evidence).** The entire signal specification was
fixed on U.S. **municipal** bonds and applied **unchanged** to corporates —
different asset class, different issuers, different data vendor. A curve-fit
signal does not transfer across markets; this one does (+2.06% excess,
+1.63% OOS).

**Rejected "improvements"** (each judged on OOS excess vs the base +1.63%):

| overlay | IS | **OOS** | verdict |
|---|--:|--:|---|
| base | +3.37% | **+1.63%** | — |
| market-regime gate | +4.21% | +0.94% | **reject** (fit the GFC, degrades OOS) |
| per-bond credit filter | +3.20% | +0.64% | **reject** |
| regime + credit | +3.37% | +0.78% | **reject** |

The regime gate looked great in-sample and "fixed" the 2008 GFC, but it
**hurt out-of-sample** — the textbook overfitting signature. We keep the base.

**Robust levers** (monotone, fair same-control comparison, hold OOS):

| lever | OOS excess | note |
|---|--:|---|
| threshold ≥3 pt | +1.63% | broad operating point |
| threshold ≥4 pt | **+2.96%** | deeper = cleaner signal, fewer trades (capacity/alpha tradeoff) |
| hold ~455 d | +2.78% | more excess, more duration risk |
| **duration ≤5 y** | **+3.51%** | short bonds pull to par → cleaner reversion; ~30% of trades, **repairs the GFC** OOS (see below) |
| dynamic recovery-exit | — | cuts avg hold ~375→240 d at similar *annualized* return — a turnover/risk gain, not extra alpha |

**Selectivity — duration, not credit quality (§8 of the white paper).** A
credit-quant would ask whether a smaller, higher-conviction book beats the
3,100-position full universe. Tested point-in-time (both filters gate the
control identically):

- **Credit quality filtering backfires.** Restricting to top-IG (spread ≤1.5%)
  *kills* the OOS timing edge (−0.25%, p≈1) and even loses money outright
  (2022 rate selloff). The alpha lives in crossover/HY; only the deepest
  distress (spread >5%, falling knives) has no edge and is worth trimming as a
  tail knob.
- **Duration concentrates the edge.** Short-dated (≤5 y) dislocations must pull
  to par → OOS excess **+3.51%** (double the full book) on ~30% of the trades;
  long (>12 y) shows **zero** OOS edge. The focused ≤5 y book returns +268.9% /
  +5.93% CAGR on **1,016** avg positions (vs 3,133), and — the key
  anti-overfitting point — it **repairs the 2008 GFC out-of-sample**
  (era excess +3.46%, p=0.006), the opposite of the rejected regime gate's
  in-sample-only "fix." Monotone, structural, improves IS *and* OOS.

Reproduce: `corps/research/selective.py` (grid) and
`corps/research/selective_equity.py` (equity/era).

## 4. Look-ahead — point-in-time construction

- Signal `price ≤ trailing-60d-median − k`: median is `.shift(1)`, excludes today.
- Liquidity gate: trailing-90d active-day count over `[t−90, t)`, excludes today.
- Entry: first customer-ask print **strictly after** the signal day (≤7d).
- Exit: first bid in `[entry+min_hold, entry+max_hold]`; stale exit uses the
  last bid **at or before** the hard stop. No future information anywhere.
- Matched control draws random entry days from the **same window** with the
  **same** gate and exit logic; significance is a 2,000-sample cluster bootstrap.

## 5. Known limitations (disclosed)

- **Coupon carry** is proxied by each bond's median yield (OSBAP daily rows omit
  the coupon); the excess-vs-control metric nets it out (both legs hold the same
  bond for the same period).
- **Equity-curve drawdown** uses linear intra-trade attribution, so it is
  somewhat smoothed vs a daily mark; total return / CAGR are realized from
  bid/ask fills.
- **Capacity**: the full-universe number includes illiquid names; a live book
  would tier by liquidity, trading fewer, larger positions. Deeper-threshold
  operating points (≥4 pt) concentrate the alpha but reduce breadth.
- **Prices** are OSBAP's cleaned daily VWAP + bid/ask (a reputable academic
  pipeline), not raw executable ticks; fills assume the patient buyer captures
  the posted bid/ask.
