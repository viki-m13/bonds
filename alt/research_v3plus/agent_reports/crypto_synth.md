# Agent report: CRYPTO sleeve + synthetic data

## What the sleeve is
phoenix_v2_crypto.py: weekly TSMOM on Grayscale trusts (NOT spot). Universe GBTC (2015-05-11+), ETHE (2019-06-14+); cash=BIL; pre-2015 100% BIL. Signal mom63 = close.pct_change(63).shift(1) (:74). Friday rebalance at open (:79). Macro gate (:155-157/:197-199): SPY>200dma AND 200dma 20d slope>0 AND HY OAS 20d chg<1.0 AND VIX<30, all shift(1); off->BIL. P&L: opn.pct_change().shift(-1)*W - 10bps*sum|dW|, then port_ret.shift(1) (:116) => CSV date-t = P&L over open[t-1]->open[t]. Blend weight 0.101 from IS inverse-vol (variant_B in phoenix_v2_crypto.json matches BLEND_WEIGHTS). Standalone: full SR 0.84, CAGR ~40%, MDD -71.5%; adds ~+0.11 SR to blend (2.106->2.217). Live: build_weights(use_live_proxy=True) reassigns GBTC weight to IBIT from 2024-01-11 (:119-170); live_signal.py:238 consumes.

## Synthetic builders (QUARANTINED from production — affirmatively verified)
synthetic_letf_build.py -> data/etfs_extended/ (17 LETFs, 2005->inception): r = L*r_und - (L-1)*FEDFUNDS/252 - 90bps/252, o2o and c2c compounded separately, single close-vs-open scale factor at splice, High/Low=max/min(O,C), Vol=0. Missing: swap spread (~40-60bps x (L-1) => 3x funds ~0.5-1.5%/yr too cheap), internal rebalance drag, tracking noise. Vol drag correctly captured. FEDFUNDS monthly avg ffilled from month-start = mild lookahead. Splice artifact: UPRO c2c +7.04% phantom jump at 2009-06-25 splice (o2o +0.38%). ERX built at 2x but was 3x until Mar 2020. Correlation validation scale-invariant (can't catch wrong leverage/drag). Only consumer: phoenix_extended.py (GFC stress test, not in production path). synthetic_bil_build.py: FEDFUNDS-15bps accrual, ~+14bps/yr over-accrual (261 bdays at 1/252) roughly cancels drag; Open==Close.
All five production sleeves read data/etfs only (van:37, ori:49, hel:44, qua:61, cry:25).

## CRITICAL
C1. Cross-sleeve booking misalignment (three conventions: VAN/CRY open[t-1]->open[t] at t; QUA c2c at t; ORI open[t]->open[t+1] at t; HEL open[t+1]->open[t+2] at t). Deflates correlations (CRY vs ORI 0.001 etc. despite shared macro gate), smooths blend vol, inflates blend Sharpe. Overlay lookahead: raw[t-1] contains HEL returns realized through open[t+1]; multiplier applied to VAN/CRY/QUA date-t returns whose windows already ended => overlay reacts up to 2 days before real time in crashes — exactly the episodes it's credited for. live_signal compute_overlay_mult inherits stale trailing rows => live != backtest on volatile days.

## HIGH
H1. Backtest P&L is Grayscale premium/discount, not BTC/ETH; live instrument (IBIT) can't reproduce. Top days: GBTC o2o +68.2% (2017-12-26), +56.9% (2017-05-25), +39.0% (2017-12-22) = premium explosions. ETHE +400% single print 2019-06-20. 2023-24 discount-closing rally = one-time arb. main() universe is GBTC/ETHE ONLY — IBIT never in backtest; production CSV holds GBTC after Jan 2024 while live trades IBIT (GBTC 1.5% ER vs IBIT 0.25%; ETHE kept at 2.5% vs ETHA). 0.101 weight calibrated on a stream live cannot earn.
H2. TC 10bps unrealistic for OTCQX trusts (GBTC spreads 50-300bps 2015-17; ETHE first days 1012 shares then 0 volume, frozen OHLC).
H3. Survivorship/pseudo-OOS: {GBTC, ETHE} = the two winners; ETCG (-95%), LTCN, BCHG, GDLC absent. OOS fully realized when written; sleeve added for its full-sample lift (in-sample selection of the sleeve itself).

## MEDIUM
M1. Adjustment seam (confirmed independently): BIL Adj Close populated only from 2026-04-06; BIL c2c shows 0 ex-div drops/yr 2023-25 but 3 days < -0.15% in 2026; BIL 2026 YTD 0.88% vs ~2%+ expected. Monthly BIL ex-div books spurious ~-25bps loss to any cash-holding sleeve; all dividends after seam lost (SPY 1.2%/yr, TLT 4%/yr, BIL 4.5%/yr).
M2. 0.101 weight rests on unrepresentative IS vol: crypto column is BIL for 5.5 of 9 IS years => measured IS vol 63% vs 90-100% active-period; also premium-era vol ~2x IBIT vol. Weight hardcoded in 4 places (phoenix_production:34, refresh_all:182,253,347, live_signal:58).
M3. Overfitting residue: stale docstring (SR 2.37/CAGR 57.4%/20% target/2.0x cap/gap 0.28 vs actual 15%/1.0x/2.34/36.8%/0.434); grids: phoenix_v2_grid.csv (226), letf_sweep_crypto.csv (130, sweeps BTC lookback 63/126 + vol targets), phoenix_v2_crypto_overlay.json, phoenix_lite_grid, phoenix_enh_grid. Every distinguishing crypto param appears among swept dimensions.
M4. Synthetic builder flaws (contained, see above).
M5. No RF subtraction in any Sharpe; crypto sleeve in BIL most days => cash carry booked as alpha (~5% 2023-24); OOS Sharpes flattered.
M6. Garbage early ETHE prints (zero-volume frozen, +400% open) feed mom63 for 63 sessions; ffill(limit=5) propagates stale opens. Non-tradable marks drive first ETHE trades.

## LOW
L1. Docstring omits spy_ma slope + VIX<30 gate terms (:8-9 vs :157,:199).
L2. Implicit costless daily re-truing to target weights between Fridays (W const :97, W·r daily :112, TC only on rebal diffs :113).
L3. Synthetic BIL accrual quirk (+14bps/yr, offsets drag).
L4. live_extend +1 BDay can be exchange holiday (as_of label wrong, benign).
L5. Regime/data logic duplicated between main() and build_weights() — can silently desync backtest vs live gate.
L6. CAGR len/252 on stub years — cosmetic.

## Affirmatively clean
- Intra-sleeve timing leak-free: signal close[t-1] -> execute open[t] -> earn open[t]->open[t+1] -> rebooked t+1. lookahead_days=0 for CRYPTO in refresh_all is CORRECT.
- Live weight timing matches backtest information set exactly (instrument differs, timing doesn't).
- Calendar: no weekend rows; weekend BTC moves captured once in Monday o2o; crypto index subset of production index — nothing dropped/double-counted.
- 252-day annualization correct for exchange-listed vehicles.
- Synthetic data quarantined from production.
- Vol drag modeled correctly in synth builder; calibration diagnostics recorded honestly.
- Dedup in every loader; ffill bounded; inf-guard on inv-vol; GBTC/ETHE splits correctly adjusted.
