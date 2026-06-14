# bonds

Bond trading model using historical bond market data.

## Strategies

This repo hosts several systematic strategies with live, self-updating
factsheets at https://viki-m13.github.io/bonds/ :

- **SUMMIT** — a concentrated biweekly/monthly **DCA stock-selection** strategy
  on point-in-time S&P 500 data (beats QQQ-DCA in 93% of rolling windows). Full
  project record, research log, and reproduction steps in [`dca/README.md`](dca/README.md);
  live page `docs/summit.html`.
- **ROTATOR** — an external leadership-rotation strategy, independently
  replicated and compared on the identical harness (`docs/rotator.html`).
- **PHOENIX / APEX** and other leveraged-ETF strategies — see `alt/`, `apex/`,
  and `docs/`.
- **IGNITION** — a price-action/volume strategy that tries to buy stocks
  **before they go parabolic** (+50% / 6 months). Grounded in an extensive
  academic + FinTwit + retail research sweep, validated with an honest IS/OOS
  event study and a survivorship-matched random control. Full record in
  [`parabolic/README.md`](parabolic/README.md).

## Data Sources

### 1. Bond ETFs (via Yahoo Finance)
Historical OHLCV price data for 27 bond ETFs spanning:
- **Treasury**: SHY, IEI, IEF, TLH, TLT, GOVT, SPTL, VGLT (short to ultra-long duration)
- **Corporate IG**: LQD, VCIT, VCSH, IGIB
- **High Yield**: HYG, JNK, USHY
- **Aggregate**: AGG, BND
- **TIPS**: TIP, SCHP
- **Municipal**: MUB, VTEB
- **Emerging Market**: EMB, VWOB
- **Floating Rate**: FLOT
- **MBS**: MBB, VMBS

### 2. iBonds / Target Maturity ETFs (via Yahoo Finance)
36 target-maturity bond ETFs that behave like individual bonds:
- **iShares iBonds Corporate**: IBDQ-IBDY (2025-2033 maturities)
- **iShares iBonds Treasury**: IBTF-IBTM (2025-2032 maturities)
- **iShares iBonds High Yield**: IBHF-IBHI (2026-2029 maturities)
- **Invesco BulletShares Corporate**: BSCQ-BSCY (2026-2034 maturities)
- **Invesco BulletShares High Yield**: BSJQ-BSJV (2026-2031 maturities)

### 3. US Treasury Yield Curves (from Treasury.gov)
Daily yield curve rates for all maturities (1mo through 30yr), including:
- Nominal yield curve
- Real yield curve (TIPS)

### 4. FRED Economic Data
30+ time series including:
- Treasury yields at all maturities
- Term spreads (10Y-2Y, 10Y-3M)
- Credit spreads (IG OAS, HY OAS, AAA, BBB)
- Fed funds rate, breakeven inflation, VIX, dollar index

## Setup

```bash
pip install -r requirements.txt
python scripts/download_all.py
```

## Project Structure

```
bonds/
├── data/
│   ├── etfs/          # Bond ETF daily OHLCV data
│   ├── treasury/      # Treasury yield curve data
│   └── fred/          # FRED economic/rates data
├── scripts/
│   ├── download_all.py
│   ├── download_bond_etfs.py
│   ├── download_ibond_etfs.py
│   ├── download_treasury_yields.py
│   └── download_fred_data.py
└── requirements.txt
```
