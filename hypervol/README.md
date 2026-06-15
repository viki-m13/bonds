# HYPERVOL — porting the VIX-ETN volatility strategy to Hyperliquid perps

This directory adapts Concretum Group's **"The Volatility Edge: A Dual Approach
for VIX ETNs Trading"** (the eVRP + term-structure "Strategy 4" automated in
their IBKR notebook) to **Hyperliquid perpetual swaps**, with fills, fees, and
funding modeled honestly. It does not just translate the code — it asks whether
the strategy's *economic edge* survives the move to crypto, and validates the
answer with ablations, IS/OOS, cost stress, and parameter robustness.

**TL;DR (honest):**
1. The **literal translation** (use eVRP + a term-structure regime to time
   long/short the coin) is **weak and does not survive out-of-sample** (Sharpe
   ~0.4 in-sample → ~0 OOS). It mostly just reduces drawdown vs buy-and-hold.
2. The **economically correct translation** of "sell the volatility premium" on
   a perp venue is the **delta-neutral funding carry** (long spot + short perp).
   That edge is **real and robust in-sample (Sharpe ~6)** but **non-stationary**:
   funding compressed from ~22%/yr (2023-24) to ~3%/yr (2026), so the OOS Sharpe
   falls to ~1.3 and OOS return to low-single-digits.
3. **The entire Concretum signal apparatus (DVOL/eVRP gating, regime switching)
   adds no value in crypto — it actively *destroys* it.** Funding already prices
   the premium directly, so the best rule is the dumbest one: be on the receiving
   side of funding. This is the most important finding here.
4. **Intraday (§5) re-prices the risk honestly:** the carry's true vol is ~12%,
   not the 2% daily marks suggest (the basis mean-reverts within the day), and in
   today's compressed/dispersed-funding regime a diversified, funding-*sign*
   carry basket (Sharpe ~0.5, ~3.5%/yr) beats the old "always short" rule.

---

## 1. The translation

The VIX-ETN strategy has three ingredients and a sizing rule. Each has a clean
Hyperliquid analog:

| VIX-ETN element | Hyperliquid analog | Source |
|---|---|---|
| Underlying SPY | Coin perp price | HL `candleSnapshot` 1d |
| Realized vol (10d SPY) | 10d realized vol of the coin | computed |
| Implied vol = **VIX** (30d) | **DVOL** (Deribit 30d implied-vol index) | Deribit |
| **eVRP** = VIX − RV | DVOL − RV | both |
| Term structure **VIX vs VIX3M** (roll/contango) | **Funding rate** (perp basis / roll carry) | HL `fundingHistory` |
| **SVXY** (short-vol, +risk beta) | **Long** the perp / receive funding | HL |
| **VXX** (long-vol, −risk/crash beta) | **Short** the perp / pay funding | HL |
| Size ∝ VIX | vol-target or IV-proportional | computed |

Two structural facts drove the design:

* **Funding _is_ the term structure.** A perp trades at a premium to spot
  (positive funding, "contango") ~90% of the time; the funding a position
  pays/receives is exactly the roll carry that VXX/SVXY earn off the VIX futures
  curve. So the VIX↔VIX3M signal maps onto the *sign of funding*, not onto a
  separately-fetched second tenor.
* **Perps never expire ⇒ no roll cost.** Unlike VIX futures (which bleed roll
  every month), a perp position is rolled "for free"; the only costs are
  entering, exiting, and flipping. This is why perp funding-farming is cheap and
  why we model cost on position *changes* only.

### Two strategy families built

* **Directional** (`engine.py`, `mode="directional"`): the faithful beta map —
  SVXY→long coin, VXX→short coin — using the four eVRP × term-structure regimes
  to go long / half-long / short / flat, vol-targeted.
* **Carry** (`carry.py`): the economically-correct "sell the premium" trade —
  **delta-neutral** long spot + short perp to collect funding, in always-on and
  Strategy-4-gated forms.

> A naked single-leg perp "carry" (short the perp without the spot hedge) is also
> tested — it **loses 24%/yr** because you are short a market that rose. That is
> the cautionary result that forces the delta-neutral construction.

---

## 2. Honest execution model

* **Prices:** Hyperliquid perp 1d candles (what actually fills), not Yahoo spot.
  Perp vs spot price level agrees to mean 0.01%, std 0.09% — the basis is small
  and is modeled explicitly, not assumed away.
* **Funding:** HL funding is charged **hourly**; we sum the 24 hourly prints into
  a realized daily funding and debit/credit it on the held notional every day
  (longs pay positive funding, shorts receive it).
* **Fees:** HL base **taker 4.5bps** + **3bps slippage** = 7.5bps/side, applied
  to traded notional on every weight change. Stressed up to 8× (60bps).
* **No lookahead:** weight decided at close `t` from signals dated `≤ t`; it earns
  `ret[t+1]` and pays `funding[t+1]`; turnover billed when the weight changes.
* **Window:** 2023-06 (HL mainnet funding inception) → 2026-06. DVOL is pulled
  back to 2021 for signal robustness but is not tradeable on HL before 2023-06.

---

## 3. Results (BTC+ETH 50/50 portfolio, all costs + funding in)

```
Benchmark      buy&hold perp 50/50            CAGR  +0.3%  vol 53%  Sharpe +0.27  maxDD -61%

Directional    Strat-4 L/S  (vol-target)      CAGR  +9.4%  vol 34%  Sharpe +0.43  maxDD -42%
               Strat-4 L/S  (IV-prop, faithful)CAGR +6.9%  vol 27%  Sharpe +0.38  maxDD -28%

Naked carry    short perp, NO spot hedge       CAGR -23.6% vol 33%  Sharpe -0.65  maxDD -72%   <-- fails

Carry (Δ-neutral) always-on                    CAGR +13.6% vol  2.0% Sharpe +6.29  maxDD -2.2%
               Strat-4 eVRP-gated              CAGR  +9.4% vol  1.8% Sharpe +5.01  maxDD -2.4%
```

### Ablation — adding "intelligence" makes it worse

Sharpe of the delta-neutral carry as signals are added:

```
a) always short perp (never flip)          Sharpe +8.07
b) + flip on daily funding sign            Sharpe +1.98   <-- churn/whipsaw hurts
c) + 7d-smoothed funding flip [our carry]  Sharpe +6.29
d) + eVRP size tilt (Strategy 4)           Sharpe +4.97   <-- the paper's signal *hurts*
```

The premium is so persistently positive that the simplest rule wins; the
DVOL/eVRP regime logic that is the heart of the equity strategy is redundant at
best and harmful at worst in crypto. (We keep the 7d-smoothed-funding version as
the headline because "always short, never flip" is the most exposed to the
unmodeled negative-funding tail.)

### In-sample / out-of-sample — the edge is decaying

```
                       IS (first 60%)          OOS (last 40%, from 2025-02)
delta-neutral carry    Sharpe +9.4  CAGR +21.8%   Sharpe +1.35  CAGR +2.5%
directional L/S        Sharpe +0.82 CAGR +23.6%   Sharpe -0.08  CAGR -8.8%
```

Cause is not a mystery — it is funding compression as the trade crowded:

```
mean annualized funding:  2023 +20%   2024 +23%   2025 +9.6%   2026 +2.8%
```

The directional timing does not survive OOS at all. The carry survives but as a
**low-single-digit, low-vol** return now, not a Sharpe-6 machine.

### Robustness & costs

* Carry Sharpe across a 28-point grid (funding window × rebalance band): median
  6.3, min 2.0 — a genuine in-sample plateau, not a single lucky cell.
* Directional Sharpe across rv-window × target-vol: median 0.32 — weak everywhere.
* Carry degrades gracefully with costs (Sharpe 6.3 → 4.3 at 2×, breaks by 8×).
* **SOL** (no DVOL, funding-only): carry Sharpe 3.9, mean funding +13.7% — the
  premium generalizes beyond the DVOL coins.

---

## 4. What this is NOT (caveats — read before trusting the Sharpe)

* **Daily close model.** It does **not** capture intraday basis blowouts or the
  liquidation cascades that are the carry trade's actual failure mode (the short-
  perp leg vs the spot leg can de-peg violently for hours during squeezes). The
  reported maxDD of −2% is a daily artifact; the real tail is fatter.
* **Capacity / crowding.** The 2023-24 Sharpe is a young-market, high-funding
  phenomenon. As section 3 shows, it has already largely arbitraged away.
* **Two-venue / collateral reality.** Delta-neutral requires spot on one venue
  and the perp short on HL (or HL spot), with margin and transfer frictions not
  in the model. Negative funding (~11% of days) means you sometimes pay.
* **Survivorship.** BTC/ETH/SOL only; no delisted-perp graveyard here.

**Bottom line:** the VIX-ETN strategy's *defensive vol-timing* idea translates
poorly to crypto, but its deeper economic engine — harvesting the volatility/
roll premium — translates **directly into funding carry**, which is real, was
very profitable in 2023-24, and is now a thin, crowded carry. The crypto market
prices its volatility premium through funding so efficiently that the paper's
clever signal stack is unnecessary.

---

## 5. Intraday extension — pricing the tail the daily model can't see

Hyperliquid retains ~5000 1h candles (~208 days), so we pulled an **hourly** book
(HL 1h perp + Binance.US 1h spot + HL hourly funding) for 2025-11 → 2026-06 — the
current, crowded regime — to do two things the daily backtest cannot.
(`fetch_intraday.py`, `intraday.py`, `basket.py`.)

**(a) The carry is ~6× more volatile intraday than daily marks imply.**

```
                 daily-close model      hourly-MTM (same trade, current regime)
BTC carry vol         ~2%                       11.8%   (Sharpe +0.30)
ETH carry vol         ~2%                       12.7%   (Sharpe +0.40)
```

The position's P&L is `Δbasis + funding`; the basis swings hard intraday and
mean-reverts by the daily close, so daily sampling *cancels* the risk that
actually liquidates positions. Worst intraday perp-rich basis spikes over the
window were +0.44% (BTC), +0.88% (ETH) — modest (no de-peg cascade occurred),
implying you must avoid running the perp leg above ~90–175× before *basis alone*
liquidates a delta-neutral book. The binding real-world risk is the directional
move forcing cross-venue margin calls, which is operational, not in this model.

**(b) The regime-adaptive funding-sign rule now beats "always short", and
diversification helps — but the edge is thin and cost-sensitive.**

Funding has compressed and *dispersed*: BTC +3.8%, ETH +5.1%, LINK +9%, NEAR +9%
but SOL −2.8%, DOT −11%, ATOM −14%, APT −15% (annualized, current). So "always be
short the perp" (the 2023-24 winner) now bleeds on the negative-funding coins,
and being on the *receiving side of each coin's funding* adds value again.

Liquid 5-coin basket (BTC, ETH, SOL, XRP, DOGE — coins whose Binance.US spot is
tight enough that the basis is real and not stale-quote noise), hourly, net costs:

```
OLD: always-short  risk-parity        CAGR +2.7%  vol 8.9%  Sharpe +0.35
NEW: sign-follow   risk-parity        CAGR +3.5%  vol 7.7%  Sharpe +0.49   <- headline
```

Robust across funding-smoothing windows 120–336h (Sharpe 0.43–0.49); 72h churns.
**But** it halves at 2× costs (Sharpe 0.20) because flipping sides is expensive,
and 12 of 17 fetched coins had to be dropped — their Binance.US spot is too stale
to trade the basis against, so a live book needs better spot venues per coin.

**Intraday verdict:** the diversified, funding-sign carry is a genuine, robust,
*low* return today (~3.5%/yr, Sharpe ~0.5, sub-1% drawdown on daily marks) whose
true risk lives in intraday basis spikes and cross-venue margin, not in the
benign daily P&L. It is a real cash-management yield, not the Sharpe-6 machine the
2023-24 daily numbers advertised.

---

## 6. Reproduce

```bash
pip install -r requirements.txt pyarrow
python -m hypervol.fetch_data       # HL funding+candles + Deribit DVOL (daily)
python -m hypervol.validate         # full daily report -> results/validation.json
python -m hypervol.plot             # results/equity_curves.png
python -m hypervol.fetch_intraday   # HL 1h perp + Binance.US 1h spot + hourly funding
python -m hypervol.intraday         # intraday tail / liquidation audit
python -m hypervol.basket           # diversified funding-sign carry basket
```

Files: `fetch_data.py` / `fetch_intraday.py` (data), `engine.py` (signals +
directional backtest), `carry.py` (daily delta-neutral carry), `intraday.py`
(hourly tail audit), `basket.py` (diversified intraday carry), `validate.py`
(all daily tests), `plot.py` (figure).
