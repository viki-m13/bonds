"""Catalog of leveraged ETFs + crypto assets available in our data store.

Every ticker listed is verified present in data/etfs/*.csv.

LETFs are tagged with:
  theme    — equity / bond / gold / semis / energy / real-estate / intl / biotech / vol / currency / crypto
  lev      — nominal leverage factor (negative = inverse / short)
  underly  — underlying proxy for 200d-SMA trend filter
  first    — first-date available (informational)

Crypto assets: spot only (BTC_USD, ETH_USD, SOL_USD, ADA_USD). Plus the 2× BTC
LETF BITX (limited history, 2023+).
"""

LETF_CATALOG = {
    # --- equity LONG ---
    "UPRO": {"theme": "equity",   "lev": 3,  "under": "SPY", "first": "2009-06-25"},
    "SSO":  {"theme": "equity",   "lev": 2,  "under": "SPY", "first": "2006-06-21"},
    "TQQQ": {"theme": "tech",     "lev": 3,  "under": "QQQ", "first": "2010-02-11"},
    "QLD":  {"theme": "tech",     "lev": 2,  "under": "QQQ", "first": "2006-06-21"},
    "TECL": {"theme": "tech",     "lev": 3,  "under": "XLK", "first": "2008-12-30"},
    "SOXL": {"theme": "semis",    "lev": 3,  "under": "SMH", "first": "2010-03-11"},
    "FAS":  {"theme": "finance",  "lev": 3,  "under": "XLF", "first": "2008-11-19"},
    "EDC":  {"theme": "em",       "lev": 3,  "under": "EEM", "first": "2008-12-30"},
    "YINN": {"theme": "china",    "lev": 3,  "under": "FXI", "first": "2009-12-03"},
    "DRN":  {"theme": "reit",     "lev": 3,  "under": "VNQ", "first": "2009-07-16"},
    "LABU": {"theme": "biotech",  "lev": 3,  "under": "XBI", "first": "2015-05-28"},
    "ERX":  {"theme": "energy",   "lev": 2,  "under": "XLE", "first": "2008-11-19"},

    # --- bonds LONG ---
    "TMF":  {"theme": "bond-long","lev": 3,  "under": "TLT", "first": "2009-04-16"},
    "TYD":  {"theme": "bond-mid", "lev": 3,  "under": "IEF", "first": "2009-04-16"},
    "UBT":  {"theme": "bond-long","lev": 2,  "under": "TLT", "first": "2010-01-21"},

    # --- commodities LONG ---
    "UGL":  {"theme": "gold",     "lev": 2,  "under": "GLD", "first": "2008-12-03"},
    "NUGT": {"theme": "gold-miners","lev":2, "under": "GLD", "first": "2010-12-08"},
    "UCO":  {"theme": "oil",      "lev": 2,  "under": "USO", "first": "2008-11-25"},

    # --- equity SHORT (inverse) ---
    "SPXU": {"theme": "equity-short","lev": -3,"under":"SPY","first":"2009-06-25"},
    "SDS":  {"theme": "equity-short","lev": -2,"under":"SPY","first":"2006-07-13"},
    "SQQQ": {"theme": "tech-short","lev": -3,"under":"QQQ", "first":"2010-02-11"},
    "SOXS": {"theme": "semis-short","lev": -3,"under":"SMH","first":"2010-03-11"},
    "TECS": {"theme": "tech-short","lev": -3,"under":"XLK", "first":"2008-12-30"},
    "FAZ":  {"theme": "finance-short","lev":-3,"under":"XLF","first":"2008-11-19"},
    "DUST": {"theme": "gold-miners-short","lev":-2,"under":"GLD","first":"2010-12-08"},
    "LABD": {"theme": "biotech-short","lev":-3,"under":"XBI","first":"2015-05-28"},
    "YANG": {"theme": "china-short","lev":-3,"under":"FXI","first":"2009-12-03"},

    # --- bond SHORT ---
    "TMV":  {"theme": "bond-short","lev": -3,"under": "TLT","first":"2009-04-16"},
    "TBT":  {"theme": "bond-short","lev": -2,"under": "TLT","first":"2008-05-22"},
    "TBF":  {"theme": "bond-short","lev": -1,"under": "TLT","first":"2009-08-20"},

    # --- vol / tail ---
    "UVXY": {"theme": "vol-long", "lev": 1.5,"under":"VIXY","first":"2011-10-04"},
    "SVXY": {"theme": "vol-short","lev":-0.5,"under":"VIXY","first":"2011-10-04"},
    "VIXY": {"theme": "vol",      "lev": 1,  "under":"VIXY","first":"2011-01-04"},
    "TAIL": {"theme": "tail",     "lev": 1,  "under":"SPY", "first":"2017-04-06"},
}

CRYPTO_CATALOG = {
    "BTC_USD": {"theme": "crypto", "lev": 1, "under": None, "first": "2014-09-17"},
    "ETH_USD": {"theme": "crypto", "lev": 1, "under": None, "first": "2017-11-09"},
    "SOL_USD": {"theme": "crypto", "lev": 1, "under": None, "first": "2020-04-10"},
    "ADA_USD": {"theme": "crypto", "lev": 1, "under": None, "first": "2017-11-09"},
    "BITX":    {"theme": "crypto-lev", "lev": 2, "under": "BTC_USD", "first": "2023-06-27"},
}

# Pre-curated LONG-only LETF basket that has data from 2011-01-01 onward.
LETF_LONG_2011 = [
    "UPRO", "TQQQ", "TMF", "UGL", "SOXL", "TECL", "SSO", "QLD",
    "FAS", "EDC", "DRN", "ERX", "TYD", "UCO", "YINN", "NUGT", "UBT",
]

# LONG + SHORT set for richer strategies (still from 2011-01-01 onward)
LETF_LONGSHORT_2011 = LETF_LONG_2011 + [
    "SPXU", "SQQQ", "SOXS", "TECS", "FAZ", "TMV", "TBT",
]

# Long-only + crypto (post-2019 common window for reliable data)
LETF_CRYPTO_2019 = LETF_LONG_2011 + ["BTC_USD", "ETH_USD"]

# Long-only + broader crypto (post-2020 common window)
LETF_CRYPTO_2020 = LETF_LONG_2011 + ["BTC_USD", "ETH_USD", "SOL_USD", "ADA_USD"]


if __name__ == "__main__":
    from hydra_core import load_etf
    print(f"LETF catalog ({len(LETF_CATALOG)}):")
    for t, meta in LETF_CATALOG.items():
        s = load_etf(t)
        if s is None:
            print(f"  {t:6s}  MISSING")
            continue
        print(f"  {t:6s}  {meta['theme']:18s}  {meta['lev']:+.1f}x  "
              f"under={meta.get('under','?'):6s}  {s.index[0].date()}..{s.index[-1].date()}")
    print(f"\nCrypto catalog ({len(CRYPTO_CATALOG)}):")
    for t, meta in CRYPTO_CATALOG.items():
        s = load_etf(t)
        if s is None:
            print(f"  {t:10s}  MISSING")
            continue
        print(f"  {t:10s}  {meta['theme']:12s}  {meta['lev']:+.1f}x  "
              f"{s.index[0].date()}..{s.index[-1].date()}")
