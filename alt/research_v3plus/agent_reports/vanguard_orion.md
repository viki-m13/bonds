# Agent report: VANGUARD + ORION

## What each sleeve is
VANGUARD: long-only rotation over 4 LETFs {QLD, UGL, TMF, TYD}. Monthly (1st bday) re-rank: eligible if 189d ret>0 AND close>200d SMA (close[t-1]); inverse 60d vol weights. Daily macro gate (4 triggers: HY-OAS 20d slope>0.30 or 5d>0.25; VIX 60d z>1.2 or VIX>30; T10Y2Y<0 falling; SPY<200SMA; 5d-smoothed count -> participation 1.0/0.75/0.5/0.25/0, lag 1) x constant gross=1.5. Open fills, o2o returns, 5bps. SR 0.96 full (IS 0.96/OOS 0.95), CAGR 24.8%, MDD -41%, turnover 15.6x/yr. Books day-t = open[t-1]->open[t] (vanguard_strategy.py:196-199).
ORION: 50/50 two sleeves over 16 LETFs. RISK (12): top-4 by 252d log-mom above 200MA (lag 1d), eq wt, zeroed daily when gate off (VIX>30 or HY OAS>7, lag 1d). SAFE (TMF/UBT/TYD/UGL): top-2 by 0.7 z(mom)+0.3 z(-60d vol). Weekly Wednesday, open fills, 5bps. SR 0.85 full (IS 0.68/OOS 1.02). Books day-t = open[t]->open[t+1] (orion_strategy.py:241 o2o = opens.pct_change().shift(-1)).
Design docs honestly admit failing Sharpe>=2 targets.

## CRITICAL
1. Cross-sleeve dating mismatch (VAN t=open[t-1]->open[t]; ORI t=open[t]->open[t+1]). corr(VAN[t],ORI[t-1])=0.58 vs corr(VAN[t],ORI[t])=-0.02. ORI[t] corr 0.50 with QLD open[t]->open[t+1]. Consequences: (a) fake diversification, blend Sharpe overstated; (b) GENUINE 1-day leak in overlay path: overlay mult for day t uses raw[t-1] which contains ORION's open[t-1]->open[t] return known only at open[t]; that multiplier scales VAN's day-t return accruing open[t-1]->open[t], whose position was set at open[t-1] — before the info existed. Fix: single dating convention, then refit weights/overlays.

## HIGH
2. Mixed adjustment regime in price CSVs: Adj Close empty through 2026-04-03, populated 2026-04-06+ with Close==Adj Close. Bulk history = auto_adjust=True (adjusted OHLC); incremental fetcher appends RAW rows (live_signal.py:91 auto_adjust=False), re-fetch window max 14 days (live_signal.py:82-86). Strategies read Open/Close only. => all dividends with ex-date after ~2026-04-06 booked as price losses (TMF/UBT/TYD ~4%+/yr distributors; ORION SAFE=50% of book; VAN holds TMF/TYD/UGL). Splits older than 14 days before a fetch would splice unadjusted onto adjusted (UCO/YINN/ERX split often). Freeze mechanism locks errors in forever. Pre-2026-04 history affirmatively clean.
3. In-sample universe + leverage calibration: VANGUARD_DESIGN §2.1 admits wider baskets "produced lower Sharpe" => CORE {QLD,UGL,TMF,TYD} picked on realized full-sample Sharpe. LEV_UNIVERSE list (:52-57) is dead code in backtest (only CORE loaded :251,:274) — makes screening look broad. gross=1.5 calibrated to hit 20% CAGR target on full sample (:16). Universes all-survivors, no delisted LETFs; NO synthetic pre-inception data (SOXL.csv starts 2010-03-11; late-inception names sit out via momentum warmup). VAN ~18-20 free params; ORI ~13. orion:80 claims "set by inspection of IS only" — no IS-only harness in repo to substantiate. No VAN/ORI-specific grid scripts found (affirmative). Wednesday DOW + 189d lookback classic overfit knobs; design-doc sweep (0.90-0.96 across lookbacks) mitigates VAN lookback, not universe.

## MEDIUM
4. ORION: 256 dead warm-up zero days inside scored window (prices sliced to START_DATE before 252d mom; first nonzero 2011-03-16) — IS Sharpe diluted (conservative but pads IS/OOS gap stat). Last row fabricated 0.0 (net=(gross-tc).fillna(0.0) :245); repaired next cron but production always consumes latest date with fake ORION 0. fillna(0) would also zero mid-sample NaNs.
5. HY-OAS publication lag: both sleeves trade open[t] using BAMLH0A0HYM2[t-1], which often posts to FRED around/after 9:30 open. Hours-scale availability risk, not hard leak. Recommend extra day lag.
6. TC realism: flat 5bps one-way on thin LETFs at the open (TYD ADV <50k shares; UGL/DRN/EDC/YINN/UCO 10-40bps spreads). ORION gate applied daily on top of weekly freeze (:206-207) — flickering gate liquidates/re-enters half the book on consecutive days at optimistic cost. Realistic costs ~1-3%/yr off VAN. No shorting anywhere (clean).
7. VANGUARD 1.5x gross with zero financing cost (:236,:264): missing ~1-2.5%/yr funding in 2022-26. Partial offset: cash earns 0 when participation<1 (conservative). At PHOENIX level 0.236 weight keeps aggregate gross<=1 (self-funding possible), but standalone CAGR 24.8% assumes free leverage.
8. VANGUARD synthetic non-trading days: bdate_range reindex + ffill(limit=2) (:90-92); 153 fake dates incl 11 Jan-1 rows; January rebalance regularly "executes" at stale Dec-31 open on a holiday. P&L impact tiny (mean |ret| 4e-5), Sharpe dilution conservative; live reconciliation mismatch in January.

## LOW
9. Inconsistent NaN semantics: c_spy = ~(spy > 200MA) fires risk-off on NaN (:131); other 3 triggers fillna(0)=risk-on. Unintentional asymmetry, conservative direction.
10. Silent NaN swallowing in P&L ((w_lag*o2o).sum skips NaN :199; ORION net.fillna(0) :245). No bare excepts anywhere (clean).
11. Minor: rank(method="first") ties by column order; VAN load_etf no dedup (fetcher dedups on write, 0 dups verified); VAN cost booked one day late (cost_lag shift(1) :202) — cosmetic.

## Affirmatively clean
- VAN signal chain: all signals on closes.shift(1), participation shift(1), weights shifted again in backtest — one clean lag, no shift(-1).
- ORI signal chain: all signals shift(1)+; weekly freeze on lagged signals; x-sec ranks per-day; first row zero weights.
- No synthetic pre-inception data either sleeve.
- Pre-2026-04 prices properly split/div-adjusted (adjusted opens as fills = correct total-return convention).
- No duplicate dates in inspected CSVs; FRED files clean.
