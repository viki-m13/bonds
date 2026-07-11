# Data validation — EMMA individual muni trades

Files: **1416** securities, **2,024,432** trades.

## 1. Structural checks
- Failures: **109**
  - A0105E0A276E85EBB52BEA43EF0AFA8D5: bad px=0 ytw=9 par=0 side=0
  - A08B97C3C0D5BA44CFDA81E8350CF8D35: bad px=0 ytw=0 par=6 side=0
  - A0F0A964278F12BCED1CE6746B88DA560: bad px=0 ytw=0 par=1 side=0
  - A104F4245359B29BC6822C9C0791CC295: bad px=2 ytw=0 par=0 side=0
  - A113263E67AAA6517C61D1D8798E094D5: bad px=0 ytw=0 par=4 side=0
  - A12F5EDC196D96102C98059AD85B99D67: bad px=0 ytw=0 par=3 side=0
  - A16C96723E9E81B97490C843B99FA4AA3: bad px=0 ytw=0 par=2687 side=0
  - A1750711774022AD19BCAC08A0FF4BE46: bad px=0 ytw=0 par=7 side=0
  - A1C859AC8F2D0557A91008690D45A8E20: bad px=0 ytw=0 par=1 side=0
  - A1EA4B800FE0157C0C01ADA00DBEF3ACE: bad px=0 ytw=0 par=2668 side=0
  - A1EE6D474064A014102D229D4A5151446: bad px=0 ytw=0 par=8 side=0
  - A2145203AF64CAF79E53C745F04D81C2A: bad px=0 ytw=0 par=1 side=0
  - A236D1654D617E8721E001DA00D210357: bad px=0 ytw=0 par=1 side=0
  - A272B78C1F442058208DB37BFC2A56AB0: bad px=0 ytw=0 par=2 side=0
  - A2AF676912D5CB5EF5D0ACBB23F501A3F: bad px=0 ytw=0 par=6 side=0
  - A2BADD6A9B0628AD00D4531DB83915509: bad px=0 ytw=0 par=4 side=0
  - A3279A8778088D1537C522D221366E257: bad px=0 ytw=0 par=6 side=0
  - A33E1DC77E6AD08101995E38CDA2E66B7: bad px=0 ytw=0 par=2 side=0
  - A34B4D20BEF8EC7364BACBD1427EA8A5A: bad px=0 ytw=0 par=4 side=0
  - A35674E453E1DB4E58DA9331A40D3ABC5: bad px=0 ytw=0 par=7 side=0

## 2. Side semantics (customer buy S ≥ customer sell P, same day)
- Bond-days with both sides: **195,969**
- median(S) ≥ median(P): **96.9%**
- Median same-day retail spread (S−P): **0.437** points; mean 0.913

## 3. Price–yield inversion (daily changes, per bond)
- Bonds tested: **1226**
- Median corr(Δprice, Δytw): **-0.979**; share < 0: 99.9%

## 4. Cross-endpoint check vs FindSimilarSecurities summary
- Sampled securities: **150**
- Min/max price match (±0.011): **100.0%**
- Trade-count match (±max(2,2%)): **100.0%**

## 5. Coverage
- Median span per bond: **1425** days; median active days: **235**
- First-trade year distribution: {2005: 3, 2006: 2, 2007: 1, 2008: 3, 2009: 10, 2010: 7, 2011: 4, 2012: 10, 2013: 11, 2014: 13, 2015: 27, 2016: 85, 2017: 87, 2018: 66, 2019: 99, 2020: 101, 2021: 86, 2022: 118, 2023: 117, 2024: 228, 2025: 254, 2026: 84}
- Par sizes multiple of 5k: **93.6%**
