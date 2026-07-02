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

## Queue for next iterations
- [ ] **Intraday momentum (Gao-Han-Li-Zhou)**: first-half-hour return predicts
      last-half-hour; trade 15:30→close only. 1 round-trip/day max, unlevered
      base — structurally cost-robust. Test on all 7 ETFs.
- [ ] Afternoon VWAP reversion; noon-breakout variant of intraday TSMOM.
- [ ] Cross-ETF intraday ORB ensemble at best-case cost (corr across ETFs?).
- [ ] Combine best intraday sleeve(s) with the EOD 6-sleeve book (corr ≈ 0
      expected) — measure ensemble Sharpe lift.
- [ ] Honest final statement of the 25/2.0 feasibility with this data.
