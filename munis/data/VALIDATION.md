# Data validation — EMMA individual muni trades

Files: **3085** securities, **5,023,257** trades.

## 1. Structural checks
- Failures: **233**
  - A0105E0A276E85EBB52BEA43EF0AFA8D5: bad px=0 ytw=9 par=0 side=0
  - A060E9138CD7CB4BA5AB28814D1659190: bad px=0 ytw=0 par=2 side=0
  - A06D57C11312F2AE6ADBC3ABEDED2EB11: bad px=0 ytw=54 par=0 side=0
  - A08B97C3C0D5BA44CFDA81E8350CF8D35: bad px=0 ytw=0 par=6 side=0
  - A0953D5B026D97FF63E349181ECC800AD: bad px=0 ytw=0 par=3 side=0
  - A09A360520319AFA2B42898086474982D: bad px=0 ytw=0 par=2 side=0
  - A0AAC652310463E5CAE3CF934F754C706: bad px=0 ytw=18 par=0 side=0
  - A0CEA5E8706D46A2267B8479D67F9DD62: bad px=0 ytw=0 par=2 side=0
  - A0D7BF7A96AC55A916F2855B1CF0D9B69: bad px=0 ytw=0 par=4 side=0
  - A0EC5B3305B4FDA23E92FAD5114D8FC16: bad px=0 ytw=0 par=4 side=0
  - A0EF8B56D8422CCF347F0B9DA947F531E: bad px=0 ytw=0 par=2 side=0
  - A0F0A964278F12BCED1CE6746B88DA560: bad px=0 ytw=0 par=1 side=0
  - A0F5620F0A7315C230F9FA2BA48EE754A: bad px=0 ytw=0 par=2 side=0
  - A104F4245359B29BC6822C9C0791CC295: bad px=2 ytw=0 par=0 side=0
  - A108DCA2CA90F2C52A6340456575D5CC2: bad px=0 ytw=0 par=4 side=0
  - A113263E67AAA6517C61D1D8798E094D5: bad px=0 ytw=0 par=4 side=0
  - A12DF7344F19FC81C911356A67BCD6AF2: bad px=0 ytw=0 par=1 side=0
  - A12F5EDC196D96102C98059AD85B99D67: bad px=0 ytw=0 par=3 side=0
  - A136E528D0F2A4DBF80DDBC90B5BC855C: bad px=0 ytw=0 par=3 side=0
  - A16C96723E9E81B97490C843B99FA4AA3: bad px=0 ytw=0 par=2697 side=0

## 2. Side semantics (customer buy S ≥ customer sell P, same day)
- Bond-days with both sides: **467,680**
- median(S) ≥ median(P): **96.7%**
- Median same-day retail spread (S−P): **0.500** points; mean 0.931

## 3. Price–yield inversion (daily changes, per bond)
- Bonds tested: **2765**
- Median corr(Δprice, Δytw): **-0.978**; share < 0: 100.0%

## 4. Cross-endpoint check vs FindSimilarSecurities summary
- Sampled securities: **150**
- Min/max price match (±0.011): **98.7%**
- Trade-count match (±max(2,2%)): **92.7%**

## 5. Coverage
- Median span per bond: **1399** days; median active days: **273**
- First-trade year distribution: {2005: 8, 2006: 3, 2007: 1, 2008: 4, 2009: 16, 2010: 16, 2011: 10, 2012: 16, 2013: 23, 2014: 23, 2015: 63, 2016: 174, 2017: 204, 2018: 139, 2019: 208, 2020: 219, 2021: 220, 2022: 251, 2023: 284, 2024: 518, 2025: 535, 2026: 150}
- Par sizes multiple of 5k: **97.3%**
