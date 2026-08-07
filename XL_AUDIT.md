# KEYSTONE-XL & GRANITE-XL — trade-desk audit (2026-08-07)

Full adversarial audit of the two XL strategies — the muni book (KEYSTONE-XL,
`munis/`) and the corporate book (GRANITE-XL, `corps/`) — done the way a
hedge-fund PM would demand it: reproduce everything from the committed data,
hunt for look-ahead / survivorship / accounting bias, and quantify what the
backtest promises that a live book would *not* deliver. Everything below was
measured, not asserted; the audit scripts' methods are described inline and
all published artifacts were regenerated after the fixes.

## Verdict in one paragraph

Both strategies reproduce **bit-for-bit** from the committed data, the core
dislocation-reversion alpha is real (excess vs matched controls survives a
stricter two-leg bootstrap at p<0.001 in all four windows), and no fatal
look-ahead was found in entries, signals or eligibility. The audit did find
**one implementation bug that made a published risk control fictitious**
(KEYSTONE-XL's issuer cap — now fixed, all artifacts regenerated), **one
material accounting bias in GRANITE-XL's headline** (the median-YTW carry
proxy adds ≈ +1.3 to +2.9 pp of CAGR that real coupons would not have paid —
corrected figures below), a **mild same-day-mid look-ahead in the recovery
exit** (≈ −0.4 to −0.9 pp per trade if executed with a strictly-lagged
trigger; roughly CAGR-neutral at book level), and several **presentation
biases** (win rates flattered by right-censoring, monthly Sharpe flattered by
stale-mark smoothing — the honest annual-frequency Sharpe is ~0.5–0.65, not
1.0+). After corrections the strategies still clearly beat their benchmarks;
they are just less shiny than the headline numbers: **GRANITE-XL OOS ≈ +14.2%
CAGR / Sharpe(m) 0.86 / Sharpe(a) ~0.6 / maxDD −35%** (was +17.1% / 1.03),
**KEYSTONE-XL OOS ≈ +9.3% CAGR on a book 40% smaller than previously
published** (was +9.4% on the un-capped book). Neither strategy is currently
executable live off this repo without a data-feed decision (see §7).

---

## 1. What reproduces (verified)

| check | result |
|---|---|
| KEYSTONE-XL full pipeline rerun (`keystone_xl.py`) | identical to committed `results/keystone_xl.json`, line for line |
| GRANITE-CL fills IS (2,018) and OOS (2,733) rebuilt from the committed parquet panel | **100% match** with committed `fills_granitecl_*.json` — same entries, exits, prices, coupons, returns |
| GRANITE-XL OOS3 table (base / sized / dynamic / sized_dynamic) | CAGR and Sharpe reproduce to the 4th decimal (+17.13% / 1.03 sized_dynamic) |
| Muni +limit excess significance, both-leg cluster bootstrap (strategy leg by bond cluster AND control leg resampled — the published p held the control mean fixed) | IS +3.36% p<0.0005, OOS +4.00% p<0.0005 |
| GRANITE-CL OOS excess, same two-leg bootstrap | +5.83% p<0.0005 |

The committed data is sufficient, the pipelines are deterministic, and no
number in the published JSONs disagreed with a fresh run. That is rare and
worth saying plainly.

## 2. Bug (fixed): KEYSTONE-XL's issuer cap did nothing

The cap ("max 1 concurrent position per issuer") was implemented as
`six[:6]` — correct for corporate CUSIPs, where the first 6 characters *are*
the issuer — but muni securities are keyed by EMMA's **opaque 33-char hash
ids**. Among the 3,085 traded bonds, 3,082 have unique 6-char prefixes: the
cap could never bind (it removed exactly 1 of 1,058 fills across IS+OOS).
The real issuer structure is heavily concentrated — 1,247 distinct issuers,
with e.g. 83 NYC Transitional Finance Authority bonds — so the published
"+issuer" stage and the tearsheet's "1 position/issuer" claim, and the page's
"~835 issuers" diversification stat, were all fictitious.

**Fix applied**: `keystone_xl.issuer_cap` (and the quant-appendix
concentration stats) now key on the real issuer parsed from the EMMA
universe description. All muni artifacts regenerated. Effect:

| KEYSTONE-XL | published (no-op cap) | **fixed (real cap)** |
|---|---|---|
| IS trades / mean / CAGR | 484 / +5.50% / +7.98% | **325 / +5.27% / +7.84%** |
| OOS trades / mean / CAGR | 573 / +6.92% / +9.44% | **336 / +6.85% / +9.29%** |
| full-window equity (2012–26) | +7.97% CAGR, −4.7% maxDD | **+7.82% CAGR, −4.8% maxDD** |

The alpha survives intact on the properly-capped book — the bug was
diversification-flattering, not return-flattering — but the published trade
counts, capacity and concentration claims were wrong and are now corrected.

## 3. Material bias: GRANITE-XL's carry proxy inflates the headline

The OSBAP panel does not carry the coupon; the engine credits daily accrual
at each bond's **median yield (clipped 1–12%)**. For discount/distressed
bonds — exactly what this strategy buys — yield ≥ coupon, so the proxy pays
phantom carry. The repo already contains a validated coupon-recovery method
(price/yield/maturity inversion, near-par error 0.05 pt, 99.95% premium-bond
sign consistency) — but it was **never wired into the GRANITE fills** (and
had bit-rotted: it crashed on the current cache schema; fixed in this audit).
Re-pricing the *identical* XL fills with recovered real coupons:

| GRANITE-XL (depth-wt + recovery) | published (YTW proxy) | **real coupons** |
|---|---|---|
| IS 2003–15 | +19.24% CAGR / Sharpe(m) 1.15 | **+17.95% / 1.09** |
| OOS 2016–25 | +17.13% CAGR / Sharpe(m) 1.03 | **+14.19% / 0.86** |
| full sample | +16.62% CAGR / Sharpe(m) 1.03 | **+14.80% / 0.94** |
| mean/trade (full) | +6.22% | +5.79% |

The gap is largest OOS (−2.9 pp CAGR) because 2016–24 yields ran far above
coupons. The recovered coupons are themselves conservative for premium
callables (YTW-based, biases carry *down*), so the truth is likely between
the columns — **plan around ≈ +14–15% OOS CAGR, not +17%**. The
excess-vs-control numbers are unaffected (both legs hold the same bond), and
the GRANITE-CL 1-year book is barely affected (+10.98% → +9.97% mean/trade
OOS; IS unchanged). This is a GRANITE-XL-headline issue, not a
signal-validity issue.

## 4. Mild look-ahead: the recovery exit's same-day mid

The XL recovery exit sells at the first customer-bid print (≥21d) once "the
day's mid has recovered to the entry-day trailing median". The day's mid is a
full-day aggregate (inter-dealer/VWAP) — not knowable before executing at
that same day's bid print. Re-running with a strictly-prior-day trigger
(executable live):

- **Corps**: per-trade mean −0.93 pp IS / −0.10 pp OOS; **book CAGR is
  within noise** (IS 19.24→19.16, OOS 17.13→17.85 — holds shorten too), so
  no NAV-level distortion.
- **Munis** (fixed-cap book): per-trade mean −0.72 pp IS (+5.27→+4.55),
  −0.39 pp OOS (+6.85→+6.46); win rates essentially unchanged.

Verdict: not a thesis-breaker, but live execution should use the lagged
trigger and expect the lagged numbers, i.e. **≈ 0.1–0.9 pp/trade less than
the published per-trade means** (munis sit at the −0.4 to −0.7 pp end).

## 5. Presentation biases a PM should deflate

- **Muni OOS win rates (97–98%) are flattered twice.** (a) Right-censoring:
  muni entries run to the last data day; entries after ~2025-04 cannot
  complete the 1-year hold, exit "stale" at the last print and are credited
  the **full 455-day coupon accrual** — that slice prints 99–100% wins by
  construction. Cutting entries at data-end−455d (the corps convention, which
  corps already follows): win 97→93%, mean +8.58→+8.63%, excess actually
  *rises* (+4.0→+5.7%). (b) Universe survivorship: the EMMA universe is
  "bonds that traded 2025-07→2026-07", so 2023–24 entries are conditioned on
  the bond surviving to 2025+; the survivorship-tilted OOS slice shows +5.5%
  excess vs +2.5% in the final survivorship-free year. **The durable planning
  numbers are the IS excess (+3.6% capped) and the final-year OOS (+2.5%),
  not the blended +4%.** The published mean returns are barely affected; the
  win rates and p-values are the flattered quantities.
- **Monthly Sharpe is smoothed.** XL book monthly returns have lag-1
  autocorrelation 0.38 (stale marks). The engine honestly computes annual
  Sharpe too — **0.53 full / 0.55 IS / 0.64 OOS** vs monthly 1.03/1.15/1.03 —
  but only the monthly number was displayed on the pages. Live, marked on
  liquid prices, expect the annual-ish number.
- **The NAV assumes free daily re-weighting at mid** (equal/depth weights
  renormalized daily). Implied one-way turnover ≈ 1.7× NAV/yr ≈ 1.1%/yr drag
  at per-bond half-spreads *if actually traded*. Measured alternative: a
  **no-rebalance drift-weight book returns +18.05% CAGR with −35.2% maxDD**
  (vs +16.62%/−41.2% renormalized) — so the construction is not
  return-flattering, but the published *path* is not what a live book would
  print. Run drift weights live.
- **Book concentration at the edges.** The "fully-invested" NAV rides 1–6
  open positions in early 2003 and late 2024–25 (1.9% of invested days have
  <10 positions); those segments of the equity curve are single-name risk,
  not strategy performance.

## 6. What was checked and came back clean

- Signals use `shift(1)` trailing medians; liquidity gate counts `[t−90, t)`;
  entries are strictly-after-signal asks (≤7d); exits use only prices at/
  before exit; the limit filter uses the latest *prior* mid (median age 1
  day, never >7d). No entry/eligibility look-ahead found.
- Corps survivorship: full 55,545-bond TRACE universe including dead bonds;
  14–16% stale exits eat real losses; corps entries already stop 455d before
  panel end (no right-censoring).
- Threshold monotonicity, IS→OOS persistence, era robustness, the reverse
  transfer of the limit rule to munis (+3.36% IS / +4.00% OOS excess with
  cap-matched controls) — all reproduce.
- Excess significance is *not* an artifact of the fixed-control-mean
  bootstrap shortcut: resampling both legs leaves every headline p <0.001.
- Slippage sensitivity exists and is sane (GRANITE-XL CAGR 16.6→13.9% at a
  0.5 pt haircut; KEYSTONE-XL 8.2→7.0%). Capacity estimates: ~$77M implied
  AUM (corps, 25% participation), muni sleeve is boutique-sized.

## 6b. Second-pass checks (data integrity, stability, small defects)

- **Panel integrity.** Corp: 29.67M rows, 210 duplicate (bond,day) rows
  (negligible), 0.06% of mids <10, mid outside the day's [bid,ask]±0.5 on
  1.0% of two-sided days; **7.8% of two-sided days print ask<bid**
  (trade-derived sides can cross — execution noise, works both ways, and the
  median spread paid is still 0.31 pt / p90 2.1 pt). Muni: 0 duplicates,
  3.1% buy<sell violations, median spread 0.66 pt — consistent with the
  repo's own VALIDATION.md.
- **Control stability.** GRANITE-CL OOS excess across 5 control RNG seeds:
  +5.04% to +5.99% (published +5.48% mid-range). Conclusion unchanged by
  seed choice.
- **No single-name driver.** XL full book: 4,582 fills across 4,052 bonds
  (max 4 fills/bond); the top-10 bonds contribute 12.8% of total weighted
  per-trade return.
- **Small spec inconsistency found: the XL recovery exit can outlast the
  base exit** that the cooldown and issuer cap were computed against (the
  455-day stale path extends the hold). Result: 98 of 4,582 corp XL fills
  (2.1%) overlap a same-bond position, and 9 of 656 muni XL fills (1.4%)
  overlap a same-issuer position — technically violating the 1-per-issuer
  rule the entries were filtered on. Effect is de-minimis at book level, but a
  live book should re-check the cap at entry against *actual* open
  positions, not the 1-year-book schedule (same conclusion for munis).
- **XL-stage excess caveat (munis).** The keystone_xl.py "excess" for the
  recovery-exit stage compares a ~230–310d-hold strategy against the
  1-year-hold control (+2.37% OOS); it is a conservative apples-to-oranges
  comparison retained for continuity — the paired-entries MTM comparison is
  the decision metric, as in corps.

## 7. Live-readiness — the honest gap list

1. **No live corporate data feed.** The OSBAP panel ends 2025-03 and updates
   ~annually; the FINRA API credential in use cannot access trade-level
   TRACE (403 on all `trace*` datasets). GRANITE-XL is therefore **not
   executable live from this repo today** — it needs a TRACE-derived daily
   bid/ask/mid feed per CUSIP (WRDS TRACE, a vendor, or an upgraded FINRA
   credential) wired into the `six/date/mid/s_px/p_px` schema.
2. **Muni feed is manual.** EMMA download scripts work but nothing schedules
   them; the daily GitHub workflow refreshes only the ETF strategies. The
   published muni pages are a static snapshot as of 2026-07-10. The
   `current_picks.py` screen is live-runnable after a manual
   `download_trades.py` refresh (~30 min).
3. **Execution realism.** Backtest fills are day-level par-weighted prints
   (munis) / trade-derived bid-ask (corps). A live buyer works limit orders
   against dealer offers; the exec-model haircut grid is the right way to
   budget this — assume the 0.125–0.25 pt haircut row, not h=0.
4. **Recovery exit**: use the lagged-mid trigger live (§4).
5. **Sizing**: depth-weighting (w = depth/3 capped [0.5,2]) and the issuer
   cap are now both real, implementable rules; the NAV should be run
   drift-weight (no daily renormalization).

## 8. Changes made in this audit

- `munis/research/keystone_xl.py` — issuer cap now uses real issuers
  (`issuer_of()`); all muni XL artifacts regenerated
  (`results/keystone_xl.json`, `docs/keystone_xl_curve.json`,
  `docs/exec_muni.json`, `docs/quant_muni.json`).
- `munis/research/quant_appendix.py` — issuer concentration on real issuers
  (was hash prefixes; "835 issuers" → true count on the capped book).
- `corps/research/augment4_coupon.py` — repaired (ytw no longer in the
  cache; now re-attached from the panel), re-run and re-validated; the cache
  again carries `coupon_inv` for coupon-sensitivity work.
- `docs/granite_xl_data.json` + page blobs re-generated; tearsheet prose
  corrected (KEYSTONE-XL numbers on the real-capped book) and both
  tearsheets now disclose the coupon-proxy correction and annual Sharpe.
- `corps/CORP_AUDIT.md` §6 and `munis/research/FINDINGS.md` audit addendum
  added, pointing here.

## 9. What to expect live (planning numbers, post-audit)

| | KEYSTONE-XL (munis) | GRANITE-XL (corps) |
|---|---|---|
| Book | ~336 trades / 3.5y OOS, ~30 concurrent | ~200–400 concurrent, ~208 fills/yr |
| Mean/trade | +5.0–6.5% (lagged exit) | +5.5–6.0% (real coupons, lagged exit) |
| Excess vs random-in-same-bond | +2.5–3.6% | +3–5% |
| CAGR | +7.5–9% (bull-tilted OOS; through-cycle nearer +6–8%) | **+14–15%** (real coupons; +17% was proxy-flattered) |
| Sharpe to budget | n/a (marks too sparse) — use maxDD | **~0.6 annual-frequency** (monthly 0.86–0.94 is smoothed) |
| maxDD | −5% on smoothed marks; assume 2–3× on real marks | **−35 to −42%** |
| Failure mode | sustained rate selloff (2022: −2% excess, 35% win) | systemic credit crisis (GFC excess ≈ 0) |
| Live feed | manual EMMA refresh (works) | **missing — blocker** |

Bottom line: the research is honest by construction and now honest in
implementation; the remaining distance between backtest and live is (a) the
corrected carry, (b) execution haircuts you must budget explicitly, and
(c) infrastructure — the corporate strategy has no live tape yet.
