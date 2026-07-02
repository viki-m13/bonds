# ASCENT — biweekly/monthly stock-selection under a 30-day-min-hold mandate: the honest record

**Mandate (user-specified):** a strategy that says, every other week or monthly,
which stock(s) to buy; every position held ≥ 30 calendar days; selling allowed
after that (cut losers, rotate, or hold forever); objective = highest CAGR,
significantly outperforming DCA into QQQ.

**Verdict up front (read this before the tables):** after an independent
rebuild of the data layer, a stricter harness purpose-built for this mandate,
and ~40 new strategy configurations raced through it with proper controls —
**no selector honestly and durably beats DCA-into-QQQ on this data.** Every
apparent beat decomposes into (a) era-concentration beta, (b) recency
(the lead built in the final 6–24 months of the sample), or (c) luck inside
the random-null distribution. This independently re-confirms the repo's
standing conclusion — and adds a new finding: **the published WAVE champion
(21.5% CAGR / Sharpe 1.41, "current deployment track") does not survive the
rebuild**; two backtest defects account for the gap (§3).

What we ship anyway (§7): the honest primary (QQQ-DCA, buy on arrival) and a
fully-specified stock-form variant for anyone who wants tickers, with its true
risk profile stated (it is a concentration dial, not alpha).

---

## 1. Data layer (all committed / rebuildable)

* **Prices:** Tiingo delisting-inclusive daily adjClose+volume, 1990→2026-06-18,
  ~24k US tickers incl. 8.9k delisted (`dca/research/data/tiingo/`). Monthly and
  weekly panels rebuilt by `scripts/build_panels.py`.
* **Fundamentals:** SEC XBRL quarterly (revenue + 10 concepts, CY2011–CY2025Q4,
  `dca/research/data/sec/`), mapped with an 80-day reporting lag.
* **Insider:** SEC Form-345 bulk TSVs 2010q1–2026q1 re-downloaded and re-parsed
  (`scripts/parse_insider.py`; 498,897 ticker-months — replaces the ephemeral
  `/tmp/wave` panel the WAVE pipeline used).
* **PIT NASDAQ-100 membership:** `data/pit/n100_panel_member.parquet` (2015+).

## 2. Harness (`scripts/engine.py`) — stricter than anything previously in the repo

DCA cash flows ($1k/period, monthly or biweekly), **min-30-calendar-day hold
enforced** (exit signals during the embargo are deferred), 20 bps/side costs on
every trade, delisting haircut −25% (swept 0/−25/−50), liquidity floor
(price ≥ $3, median daily $ volume ≥ $2M), money-weighted IRR + flow-adjusted
TWR/Sharpe/maxDD, benchmark = QQQ-DCA with identical cash flows. Signals use
data through the decision close only.

Benchmark on this harness, 2015-01→2026-06 monthly: **QQQ-DCA MOIC 3.68×,
IRR 21.3%, Sharpe 1.07, maxDD −32.6%** (SPY-DCA 2.55×/15.5% — QQQ remains the
binding bar).

## 3. NEW FINDING — the WAVE champion does not survive an honest rebuild

WAVE (`strategies/WAVE_long_only.md`) is documented at 21.5%/1.41/−17.8% and
marked "CURRENT deployment track". A faithful re-implementation of its exact
pipeline (36 features → walk-forward HistGBM, fwd-12m tercile target → runner
gate → trail-30% + trend exits) on the same data found **two defects**:

1. **Test-time survivor filtering.** Model probabilities were only produced for
   rows with a *complete forward-12-month label* (`fok.dropna()` in
   exp69/78/83). A name is therefore only *pickable at time t* if it still has
   a price 12 months later — names that delist/are acquired within the year are
   silently excluded from the candidate set, and the final 12 months of the
   sample have no picks at all.
2. **Train/test label overlap.** Yearly refits trained on all months
   `< Jan(testyear)`, but a month's fwd-12m label reaches up to 12 months into
   the test year — test-year returns leak into training labels.

Decomposition (identical exp83-style costless fixed-capital sim, 2015–2026):

| variant | CAGR | Sharpe | IC(fwd3m) |
|---|--:|--:|--:|
| V0 faithful replication (both defects present) | 14.8% | 0.90 | +0.102 |
| V1 + fix test-time survivorship (predict all liquid names) | 14.5% | 0.88 | +0.099 |
| V2 + fix label embargo (train ends 12m before test year) | 12.0% | 0.73 | +0.075 |
| QQQ buy&hold same window | 19.6% | 1.06 | — |

Even V0 does not reproduce the published 21.5%/1.41 (residual gap: the original
ran on a `/tmp` feature/price snapshot that no longer exists, plus library
drift), and the honest V2 is **decisively below QQQ**. Under the full
mandate harness (DCA flows + costs + min-hold), ML configs land at
**0.47–0.71× QQQ terminal wealth** — inside the same-harness random-score null
(mean 0.58×). **Recommendation: WAVE's published claims should be treated like
SUMMIT's — a real cross-sectional signal (IC ≈ 0.075 vs absolute returns is
genuine) that does not convert into a QQQ-beat.** A warning banner has been
added to `WAVE_long_only.md`.

## 4. Candidate race (mandate harness, monthly, 2015-01→2026-06)

| config | IRR | Sharpe | maxDD | vs QQQ terminal |
|---|--:|--:|--:|--:|
| QQQ-DCA (benchmark) | 21.3% | 1.07 | −33% | 1.00× |
| ML N12 trail30+trend (WAVE-adapted) | 14.2% | 0.74 | −28% | 0.64× |
| ML+accel N12 / N8 / N16 | 11.3–13.3% | 0.57–0.68 | −27..−29% | 0.53–0.60× |
| ML rank-exit / trail sweeps / minhold60 | 9.3–14.3% | | | 0.47–0.64× |
| ML large-cap (ADV ≥ $20M / $50M / $200M) | 9.3–13.6% | | | 0.47–0.62× |
| ML relative-to-QQQ target | 16.0% | 0.80 | −25% | 0.71× |
| ML fwd-6m target | 6.2% | 0.46 | −30% | 0.39× |
| FACTOR composite (ROA+margins+52wH+lowvol+mom+buyback) | 13.2% | 0.79 | −28% | 0.60× |
| MOM12 broad (control) | −0.6% | 0.02 | −82% | 0.26× |
| RANDOM null through same harness (5 seeds) | | | | 0.33–0.84× (mean 0.58) |
| QUALIFIER (rev-accel/hi-YoY & insider-cluster & uptrend, N20) | 24.3% | 0.79 | −30% | 1.21× |
| MEGACAP-MOM k2 / k5 (momentum + 5×$vol-rank) | 51.6% / 35.0% | 0.91 / 0.70 | −44/−46% | 6.63× / 2.38× |
| NDX-pond MOM12 k8 | 32.7% | 0.98 | −42% | 2.05× |

Novel mechanics tested: cash-parking in QQQ with conviction-gated stock entry
(monotone result: **the higher the conviction bar, the closer to QQQ from
below** — 0.70× → 0.89× → 0.93–0.94× as the bar rises; selection subtracts),
conviction weighting (neutral), rank-relegation exits (hurt), 60-day min-hold
(neutral), add-to-winners vs hold-cash policies (small).

## 5. Gauntlet on the three apparent winners — all die

**Nulls (2015–2026):**
* MEGACAP-MOM k2 6.63× vs **DV-ONLY k2 (no momentum at all) 7.71×** — momentum
  *subtracts*; the entire effect is "hold the most-traded mega-caps in a
  mega-cap decade". Random-pick inside the top-20-$-volume pond: mean 1.98×.
* NDX-MOM k8 2.05× vs NDX-random null mean 0.78× (max 1.42×) — above the null…
* QUALIFIER 1.21× vs random-in-pond mean 1.08×, **max 1.21×** — exactly at the
  null's edge.

**Cutoff-date trajectory (start 2015-01), ratio vs QQQ-DCA:**

| cutoff | MEGA-k2 | MEGA-k5 | NDX-MOM-k8 | QUAL-20 |
|---|--:|--:|--:|--:|
| 2017-12 | 0.94 | 0.71 | 0.90 | 0.98 |
| 2019-12 | 0.88 | 0.69 | 0.80 | 1.12 |
| 2021-12 | 2.06 | 1.17 | 1.26 | 0.89 |
| 2023-12 | 1.41 | 0.82 | 0.97 | 0.96 |
| 2025-12 | 3.88 | 1.32 | 1.31 | 1.26 |
| 2026-06 | 6.63 | 2.38 | 2.05 | 1.21 |

NDX-MOM's lead through 2023 was ≤1.0; **more than half of the final 2.05× was
built in the last six months** (the 2026 AI-memory melt-up: MU/MRVL/SNDK).
Textbook recency artifact (validation-playbook test #4).

**Era extension (price-only strategies, 2000–2014, vs QQQ-DCA):** MEGA-k2
0.96×/0.59×, MEGA-k5 1.20×/0.86×, broad MOM12-k8 0.68×/0.27× (momentum crash),
top-5-$-volume basket 0.95×/0.65× with **−72%/−60% drawdowns** (dot-com:
CSCO/INTC leadership collapse).

**Rolling window grids (36 quarterly DCA starts):** "to-end" beat-rates of
75–100% are the recency illusion; fixed 3-year windows: MEGA-k5 66% beat
(worst 0.66×), NDX-MOM 63% (worst 0.80×), QUALIFIER 40% (worst 0.78×) — coin
flips with fat left tails.

**Cadence:** biweekly re-run of the key configs reproduces every conclusion
(ML 0.67–0.70×; leaders basket 2.3× this era, same trajectory/era failures).

## 6. Why this keeps happening (the mechanism, not a slogan)

QQQ is a cap-weighted, self-rebalancing concentration machine that already
holds the winners in proportion to how much they've won. Any equal-weight
selection from a broader pond holds *less* of the winning beta; any
concentration *within* the pond beats QQQ only while leadership persists and
gives it back (with −50%+ drawdowns) when it rotates. The model's genuine
cross-sectional skill (IC ≈ +0.075) predicts *absolute* returns in the
small/mid-cap pond — a pond that structurally lagged QQQ all sample. Timing,
entry rules, exits, cadence, and ML recombinations of the same features cannot
manufacture the missing pond beta. (46+119 prior experiments; ~40 more here.)

## 6b. Invention round 2 — new mechanisms, built and gauntleted (`run_invent.py`, `run_invent2.py`)

Per the mandate to *invent*, four mechanisms no prior repo work had tried:

* **INV-1 Beta-harvest** (top-8 trailing-beta NDX names — compensated risk as an
  implicit-leverage substitute): 1.86× (2015–26) but the trajectory shows the
  whole lead arrived in the final 6 months (1.04/0.88/1.11/0.78/0.99→1.86), and
  2003–14 = 0.46× with −71% DD. High beta is not proportionally compensated
  (vol drag + crashes) — dead.
* **INV-2 Leadership-persistence switch** (top-10-$-volume set overlap vs 12m ago
  ≥ 0.6 → hold leaders basket, else QQQ): 3.99× (2015–26), beats QQQ in BOTH dev
  and holdout — but the detector almost never fired OFF in-sample (p25 = 0.70)
  and only cushioned 2003–09 to 0.66×. Too slow.
* **INV-4 Faithful staged-exit qualifier** (exp67 ladder under the mandate
  harness): 0.80× — fails in DCA form.
* **INV-6 Concentration timing** (the genuine invention: hold the leaders basket
  only while the *concentration premium itself* is trending — current top-5
  trailing-6m return > QQQ's — else hold QQQ; always invested; disclosed as
  designed after seeing the 2003–09 failure, so judged on untouched eras):

| era | INV-6 vs QQQ-DCA | static basket | note |
|---|--:|--:|---|
| 2000–02 (untouched holdout) | 1.03× (−72% DD) | 1.03× | spread is relative → no crash protection |
| 2003–09 | 0.46× | 0.44× | designed-for failure mode NOT fixed |
| 2010–14 | 0.79× (−15% DD) | 0.78× (−37% DD) | DD halved — timing's real contribution |
| 2015–19 | 1.01× | 0.88× | |
| 2020–26 | 2.58× | 2.34× | the AI/mega-cap era pays |
| 2000–26 full | 2.80× | 1.75× | −81% max DD; 2003-start trajectory sat at 0.5× for 17 years |

Random-picks null with the same timing overlay: mean 0.53× — the era beta lives
in the *leaders* selection, not the overlay. **Conclusion: INV-6 strictly
dominates the static leaders basket (never worse, sometimes better, always
invested) but does not create era-robust outperformance.** With this round, all
four honest mechanism families are exhausted: selection alpha, pond
concentration, compensated beta, and exposure-shape regime timing.

## 7. What to actually do (the deliverable)

**PRIMARY (recommended): DCA into QQQ, biweekly or monthly, buy immediately on
cash arrival.** On every honest test in this repo — including everything new
here — this has the highest expected terminal wealth of anything implementable
from this data. It also trivially satisfies the mandate (min-hold is moot; QQQ
never triggers a cut).

**STOCK-FORM STRATEGY — "ASCENT" (the concentration-timed leaders basket,
INV-6). This is the strategy for anyone who wants tickers; size it knowing its
era profile (§6b table):**
* **Regime check** (each buy date, causal): current top-5 NDX names by median
  daily $ volume — is their equal-weight trailing-6-month return (3-month
  smoothed) above QQQ's trailing-6-month return? Is the top-10 leadership set
  ≥ 60% overlapping with 12 months ago?
  * **Both yes → regime ON:** the contribution buys the **top-5 leaders**
    (equal split; adding to an existing holding is fine), trend-gated
    (price > 10-month MA).
  * **Otherwise → regime OFF:** the contribution buys **QQQ**, and (after
    their 30-day embargo) stock holdings are rotated back into QQQ.
* **Exits** (only after 30 days held): close below the 10-month MA, or −30%
  from the peak since entry, or regime OFF. Proceeds redeploy at the next
  buy date. Never remove the trend exit — it is what caps rotation damage.
* Honest expectations: 5.06× QQQ-DCA terminal wealth 2015–2026 and ≥ the
  static basket in every tested era, BUT 0.46–0.79× through 2003–2014 and a
  −72% drawdown through 2000–02. It outperforms **if and only if concentrated
  mega-cap leadership eras persist/recur**. That is a bet, honestly priced —
  not extracted alpha.
* **Current state (data through 2026-06-18): regime ON** (leaders spread
  +191%, persistence 0.60). **Buy list: MU, NVDA, MRVL, SNDK, INTC** (next in
  line: AVGO, GOOGL). See `current_picks.json`; regenerate with
  `scripts/current_picks.py` after refreshing prices (needs `TIINGO_KEY`).

**What NOT to do:** deploy the ML/insider/qualifier pickers expecting a QQQ
beat (they are 0.4–0.7× after honesty corrections); tighten stops below the
30-day embargo (whipsaw); trade the "to-end 100% win-rate" numbers (recency).

## 8. Reproduce

```bash
export ASCENT_WORK=/tmp/ascent_work   # scratch dir for panels/caches
cd dca/research/strategies/ascent/scripts
python3 build_panels.py               # ~1 min  (monthly/weekly panels from Tiingo chunks)
#  fetch SEC Form-345 zips (curl loop, ~65 files) into $ASCENT_WORK/../sec_insider
python3 parse_insider.py              # ~30 s
python3 features.py                   # ~2 min  (36-feature panel)
python3 ml_train.py                   # ~2 min  (walk-forward GBM, embargoed)
python3 run_candidates.py             # main race + random null
python3 run_novel.py run_angles.py run_megacap.py run_biweekly.py
python3 diagnose.py                   # WAVE defect decomposition
python3 validate_finalists.py         # nulls + trajectory + era + window grids
python3 current_picks.py              # emit today's lists
```

## 9. Caveats

Tiingo lacks ~50 OTC-Q terminal bankruptcies (results mildly optimistic —
strengthens, not weakens, the negative verdict). June-2026 bar is partial
(through 06-18). Same-close execution convention (lag-1 robustness run shows
the conclusions are execution-insensitive; costs swept 5–50 bps). The NDX PIT
membership panel starts 2015 — era extensions use the broad liquid pond.
2026Q1 fundamentals not yet in the SEC cache (affects freshness of the ML
reference list only, not any conclusion).

*Not investment advice. Past performance does not guarantee future results.*
