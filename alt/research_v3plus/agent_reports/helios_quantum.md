# Agent report: HELIOS + QUANTUM

## What each sleeve is
HELIOS: cross-sectional momentum close[t-42]/close[t-189]-1 on 13 unlevered underlyings (SPY,QQQ,TLT,IEF,GLD,USO,XLK,XLE,XLF,SMH,VNQ,EEM,FXI); filter mom>0 AND close>200d SMA; macro gate VIX 252d z<1.5 AND HY OAS 20d chg<+0.3 (TLT/IEF/GLD bypass); top-2 held 50/50 expressed via matched 2x/3x LETF (UPRO,TQQQ,TMF,TYD,UGL,UCO,TECL,ERX,FAS,SOXL,DRN,EDC,YINN); residual BIL. Weekly Friday rebalance, ~28x ann turnover, 5bps x sum|dW|. Signal close[t], fill open[t+1], booked at t as open[t+2]/open[t+1]-1. IS SR 0.68, OOS 0.88, MDD -62%, 44% vol.

QUANTUM: XGBoost regressor (400 trees, depth 4) predicting fwd N-day log c2c return from ~28 features (mom lags 5/21/63/252, vol, rolling sharpe, excess-vs-SPY, dist-200dma, x-sec pct ranks; VIX, HY OAS, T10Y2Y, SPY MA spread), all shift(1). Universe 17 LETFs. Rebalance every N=21 days, top-K=3 equal weight, cash 0%. 10bps/side on target-weight turnover. Model fit ONCE on IS 2010-2018, frozen (quantum_model.pkl); (N,K) via 4-fold expanding CV rank-IC in IS (best IC 0.023). IS SR 2.73, OOS 0.87, FULL 1.72.

Price data: split/div-adjusted OHLC both Open and Close (consistent). FRED = latest vintage full history.

## CRITICAL
C1. QUANTUM published series 2010-2018 is IN-SAMPLE model output (quantum_strategy.py:566-573: final model trained on all IS at :517,:550, then predicts over full window incl IS; written to quantum_returns.csv :585). IS 2.73 vs OOS 0.87 = memorization gap. PHOENIX blends it at w=0.152 with IS-inverse-vol weights fit on contaminated IS. Only honest QUANTUM number: OOS 0.87. Fix: walk-forward refits or exclude pre-2019 QUANTUM from fitting/headlines.

## HIGH
H1. Cross-sleeve return-dating misalignment (helios_strategy.py:219 r_fwd = opens.shift(-2)/opens.shift(-1)-1 booked at t; quantum books true day-t). Corr HEL/QUA as-booked -0.04 vs +0.11 realigned. Overlay multiplier applied to HEL returns realized t+1->t+2 — a live trader cannot apply the backtest's multiplier; risk-overlay results not reproducible as stated.
H2. QUANTUM "(N,K) via CV" claim half-false: K never used in CV loop (quantum_strategy.py:241-268); identical ICs for all K at fixed N (quantum_metrics.json all_scores); strict > and iteration order => K=3 always wins. K unvalidated free parameter.

## MEDIUM
M1. Survivorship/hindsight universes (helios:54-68, quantum:66-71): hand-picked 2026-era lists of surviving LETFs; pairing map (SMH->SOXL, FXI->YINN) 2026 choice. NO synthetic pre-inception backfill (verified: CSV first dates = real inceptions — UPRO 2009-06-25, TQQQ 2010-02-11, NUGT 2010-12-08). Bias is selection, not fabricated data. robust_survivorship.py targets other sleeves.
M2. HELIOS implicitly re-trues to 50/50 daily at zero cost (w[t]*r_fwd[t] with ffilled constant weights, :194-200,:232); turnover counts only weekly target changes; drift trades absent from W.diff(). DESIGN.md "28x incl drift" not what code computes.
M3. QUANTUM drops overnight close[d-1]->open[d] return of outgoing holdings each rebalance (~12 nights/yr); exit effectively at close[d-1] = same close feeding day-d signal (same-bar exit). Mildly conservative probably, but accounting hole.
M4. QUANTUM turnover on stale 21-day-old target weights, not drifted weights -> TC understated.
M5. HELIOS params IS-Sharpe-tuned with residue: docstring VIX z<0.75 vs code 1.5 (:18 vs :84 VIX_Z_CAP=1.5 "softer gate chosen via IS"); "6-month momentum" (:11) vs MOM_LB=189 comment "9-month" (:76). ~8 free params. Mitigant: OOS 0.88 > IS 0.68; no HELIOS/QUANTUM-specific grid scripts found. Repo-level multiplicity (~60 scripts, 5 survivors blended) applies.
M6. FRED latest-vintage not point-in-time; HELIOS uses same-day HY OAS at close-t signal — OAS published next morning; implementable (fill t+1 open) but tight; revisions could flip gate near 0.3 threshold. data/pit/ exists but only equity membership panels. QUANTUM cleaner (all macro shift(1)).
M7. HELIOS last-2-rows ship 0.0-minus-cost placeholders daily (skipna sum :232); verified 2026-06-30/07-01 = 0.0. refresh_all repair mechanism correct (lookahead 2/1/0 for HEL/ORI/QUA) but phoenix_production publishes NAV incl 2 placeholder-zero HELIOS days every day.

## LOW
L1. QUANTUM loader no dedup of dates (:85-88) unlike helios (:97).
L2. HELIOS ffills opens up to 3 days (:132-134) -> fills at stale prices for illiquid names; skipna zeroes mid-history missing days.
L3. Costs optimistic for YINN/EDC/ERX/TYD (TYD ~$1-2M ADV, uninvestable at size); NUGT/YINN spreads blow out in stress. Design doc admits (:90-92).
L4. HELIOS skips entire week when Friday is holiday (:178-180), no Thursday fallback.
L5. QUANTUM early-window dead zone: until mid/late-2010 few names pass dropna; 0% if len(sl)<K; first_valid computed, never used (:462). Dilutive.
L6. build_weights(live_extend=True) fabricates next-bday row by copying last row (hel:284-289, qua:306-311); live Friday picks can use Thursday ffilled closes, diverging from backtest Friday-close convention.

## Affirmed clean
- HELIOS signal chain leak-free (shift(42)/shift(189), backward rollings; no shift(-1) in signal path). refresh_all lookahead accounting correct.
- QUANTUM feature timing clean; CV embargo correct (val_start = train_end + N); model frozen; 2019+ genuinely OOS.
- No pre-inception synthetic LETF data in data/etfs/ (synthetic_letf_build.py writes to separate pickle).
- Adjusted prices consistent across Open/Close.
- QUANTUM cash at 0%, HELIOS TC on initial BIL entry: conservative.
- Freeze-history append design sound.
