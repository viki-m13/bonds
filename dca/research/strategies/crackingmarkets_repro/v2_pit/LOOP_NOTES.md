# /loop research notes — intraday strategies toward 25% CAGR / Sharpe 2

Target: proprietary strategy, **CAGR ≥25% AND Sharpe ≥2, PIT data, honest costs**.
Data: 5-min bars, 7 ETFs (QQQ/SPY/IWM/DIA/GLD/TLT/XLF), 2016-01→2026-04, RTH+VWAP.

## Iteration 1 (2026-07-02) — ORB reproduction & cost frontier

Built `scripts/50_orb_intraday.py` (Zarattini/Barbon/Aziz 5-min ORB: first-bar
direction, enter 9:35, stop at OR opposite extreme, EOD exit, risk-1%/trade,
4x intraday cap).

**Findings:**
1. Gross (cost-free) ORB5-QQQ 2016–2026: **+28.3% CAGR / 0.97 Sharpe / −31% DD**
   — the paper's headline shape reproduces.
2. **The entire edge lives below ~0.85bp/side of friction**:
   0bp → 28.3%/0.97; 0.25bp → 19.0%/0.72; 0.5bp → 10.3%/0.47; 1bp → −5.1%;
   2bp → −29.8%. Levered sizing (≈4x most days, ~250 trades/yr, ~55% stop-outs)
   multiplies costs ~30×/yr. QQQ best-case real friction ≈ 0.25bp/side
   (1c spread at $400 + IBKR commission) **but stop-market fills in fast tape
   are worse than 0.5bp** — the strategy is hostage to execution quality.
3. Other ETFs are worse (SPY/DIA/XLF/GLD/TLT all negative even at 1bp).
4. **Conviction filters do not help**: RVOL>1.5/2.0, OR-size>median, and
   combos all LOWER Sharpe (edge is thin+broad, not concentrated). High-ATR
   regime filter keeps 0.60 Sharpe on 46% of days (best per-trade edge) but
   is still far from target.
5. ORB variants (15/30-min break entries) are weaker than the 5-min first-bar
   version at every cost level.

**Verdict so far:** honest ORB-on-ETFs ceiling ≈ 19%/0.72 at *optimistic*
friction; ≤0 at conservative friction. Not a path to 25%/2.0 by itself.

## Iteration 2 (2026-07-02) — intraday momentum, VWAP reversion, ORB ensembles

`scripts/51_intraday_momentum.py` + inline studies:
1. **Intraday momentum (GHLZ) is DEAD 2016–2026**: all 7 ETFs, all signal
   variants (first-half-hour, penultimate, agree), ≈0 or negative even at
   0.25bp/side. Textbook post-publication decay (paper sample was 1993–2013).
2. **Afternoon VWAP reversion: no edge** (triggers <1% of days at usable
   thresholds; returns ~0).
3. **Cross-ETF ORB: only QQQ carries the edge** — SPY/IWM/GLD/TLT dilute
   (4-ETF EW: 3.1%/0.29 vs QQQ-only 19.0%/0.72 at best-case costs).
4. **ORB-QQQ corr to the EOD 6-sleeve book = −0.07** (genuinely orthogonal).
   Best honest combined portfolio so far: **25% ORB + 75% EOD-k2 =
   13.8% CAGR / 1.03 Sh(d) / 1.27 Sh(m) / −20% DD** — contingent on
   0.25bp/side ORB friction (institutional-grade execution).

## Queue for next iterations
- [ ] ORB exit engineering (dev/holdout): time-of-day exit, trailing stop,
      profit target multiples — can gross/trade rise enough to survive 1bp?
- [ ] Overnight session anomaly (close→open long QQQ; 1 trade/day, daily
      OHLC suffices, 25yr history available).
- [ ] Final loop verdict: honest 25%/2.0 feasibility statement + best
      deployable portfolio; fold into FINDINGS_v2.
