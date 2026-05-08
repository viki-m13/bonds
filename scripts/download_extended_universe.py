"""Download extended universe for MERIDIAN: small/mid cap, international ADRs, small cap tech.

WARNING: this list is hand-picked from currently-listed liquid names. Any results
on this universe carry HEAVY survivorship bias — see meridian_extended_test.py
for bootstrap haircut calibration.

Each list is approximately limited to names with ≥2010 US-traded history.
"""
import os
import yfinance as yf
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "stocks_extended")
os.makedirs(OUT, exist_ok=True)

# US small/mid cap (S&P 400 / Russell 2000), broad sector spread
SMALL_MID = [
    "ANET", "ALGN", "BLDR", "CDW", "CHRW", "DECK", "DKS", "DXCM", "EXEL",
    "FDS", "FFIV", "HEI", "HUBB", "JBL", "KEYS", "LECO", "LSCC", "MASI",
    "MIDD", "MKSI", "MKTX", "MOH", "ODFL", "ON", "PAYC", "RGEN", "ROL",
    "SAIA", "SLAB", "SNX", "TREX", "TYL", "ULTA", "WAB", "WST", "ZBRA",
    "POOL", "MANH", "EXP", "NDSN", "MEDP", "FIVE", "MTZ",
    # MidCap names with long history
    "AAON", "AIT", "ALLE", "AMCR", "AOS", "APH", "ATR", "AVNT", "AVT",
    "AYI", "BCO", "BDX", "BR", "BRO", "BURL", "CASY", "CCK", "CFR",
    "CGNX", "CHE", "CIEN", "CLH", "CMS", "CNC", "COLM", "CR", "CSL",
    "CW", "DAR", "DKL", "DLB", "DOX", "DPZ", "EHC", "EME", "ENS",
    "EXLS", "FAF", "FCN", "FELE", "FLO", "FMC", "FRT", "FTI", "FTV",
    "GGG", "GPC", "HALO", "HEI", "HRC", "HSY", "HUBB", "IDXX", "IEX",
    "INGR", "ITT", "JEF", "JKHY", "KEX", "KMT", "KNX", "LAD", "LAMR",
    "LAZ", "LFUS", "LHX", "LITE", "LNC", "LPLA", "LULU", "MAS",
    "MGEE", "MORN", "MPWR", "MSA", "MSCI", "MTH", "MTN", "MTRN",
    "NHI", "NJR", "NLY", "NNN", "NRG", "NUE", "OFC", "OGE", "OHI",
    "OLN", "OMI", "OSK", "OZK", "PB", "PEG", "PEN", "PFGC", "PINC",
    "PNW", "POR", "POWI", "PSN", "PTC", "PVH", "PWR", "QGEN", "RBC",
    "REG", "REXR", "RGA", "RHI", "RJF", "RL", "RLI", "RNR", "RPM",
    "RS", "SAFM", "SAH", "SAIC", "SBNY", "SBRA", "SCHL", "SCI", "SE",
    "SEDG", "SEE", "SEM", "SF", "SFM", "SGEN", "SHO", "SHOO", "SIGI",
    "SITE", "SJM", "SLM", "SMG", "SNA", "SNV", "SNX", "SON", "SPGI",
    "SPR", "SR", "SRE", "STE", "STLD", "STZ", "SU", "SUI", "SUN",
    "SWAV", "SWK", "SWKS", "SYY", "TBI", "TDG", "TDY", "TEX",
    "TFII", "TFX", "THC", "THG", "THO", "TJX", "TKR", "TMHC", "TOL",
    "TPR", "TPX", "TRGP", "TRI", "TRMB", "TRN", "TROW", "TRP", "TRTN",
    "TRU", "TRV", "TS", "TSCO", "TT", "TTC", "TTEK", "TTM", "TTMI",
    "TTWO", "U", "UA", "UAA", "UBSI", "UDR", "UFS", "UGI", "UHS",
    "UI", "ULBI", "UMBF", "UNF", "UNH", "UNM", "URBN", "URI", "VFC",
    "VICI", "VLO", "VMC", "VMI", "VNO", "VOYA", "VRSK", "VRSN", "VRT",
    "VRTX", "VSAT", "VST", "VTR", "VTRS", "VVV", "WAFD", "WAL", "WAT",
    "WBA", "WBC", "WBS", "WBD", "WBT", "WCN", "WDC", "WDFC", "WEC",
    "WELL", "WERN", "WEX", "WGO", "WH", "WHR", "WIRE", "WK", "WLK",
    "WPC", "WRB", "WRK", "WSBC", "WSC", "WSFS", "WSM", "WSO", "WST",
    "WTRG", "WTS", "WU", "WWD", "WWE", "WWW", "WY", "WYND", "WYNN",
    "X", "XEL", "XPO", "XRAY", "XYL", "Y", "YETI", "YUM", "ZBH", "ZION",
    "ZTS", "ARW", "ATO", "AVY", "BC", "BWA", "CACI", "CAH", "CARR",
    "CCL", "CHRW", "CINF", "CMS", "CNDT", "CNX", "COTY", "CPB", "CRL",
    "DPZ", "DRI", "EQT", "EVR", "EXR",
]

# Small cap tech (broader interpretation: $1B-$20B at 2010, tech-heavy)
SMALL_TECH = [
    "EPAM", "WIX", "SPSC", "BL", "QLYS", "PCTY", "MANH", "TYL", "PAYC",
    "NTNX", "RNG", "FIVN", "ESTC",
    "EVR", "CACI", "LOPE", "GWRE",
    # Additional small/mid-cap tech with history
    "VEEV", "TWLO", "OKTA", "ZS", "DDOG", "MDB", "SNOW", "NET", "CRWD",
    "TEAM", "DOCU", "ZM", "HUBS", "PLTR", "AI", "SMCI", "IOT",
    "BSY", "PATH", "S", "GTLB", "FROG", "BILL", "DT", "ENV",
    "BMI", "CGNX", "PRGS", "QTWO", "ALRM", "BAND", "BRZE", "ASGN",
    "CIEN", "COHU", "CYBR", "VRNS", "QLYS", "TENB", "PFPT",
    "FIVN", "MIME", "WK", "JNPR", "FFIV", "NTNX", "SUMO",
    "DOMO", "ZUO", "EVER", "PD", "CSGS", "AMSC", "ANSS", "PEGA",
    "TRMB", "ADTN", "VRSN", "INTU", "ADSK", "ANSS", "FLT", "GLOB",
    "OTEX", "DXC", "EPAM",
]

# International ADRs (US-traded), big enough to be liquid
INTL = [
    "TSM", "ASML", "NVO", "SAP", "TM", "HMC", "BTI", "DEO", "UL",
    "BHP", "RIO", "TTE", "SHEL", "BCS", "HSBC", "MFG",
    "AZN", "GSK", "NVS", "RY", "TD", "BNS", "BP", "STM",
    "BABA", "BIDU", "JD", "PDD", "NIO", "TCEHY", "NTES",
    "ICLR", "ABEV", "VALE", "PBR", "ITUB",
    # Additional intl ADRs
    "TEVA", "SNY", "NMR", "MUFG", "SMFG", "RELX", "FMS", "PHG",
    "VOD", "ORAN", "E", "ENB", "TRP", "CNQ", "SU", "CNI",
    "WCN", "CP", "MFC", "BMO", "WIT", "INFY", "IBN",
    "HDB", "NTDOY", "SONY", "MUFG", "SMG", "NICE", "CHKP",
    "TLK", "PHI", "VIV", "TLK", "STM", "BIDU",
    "ATI", "AGCO", "LH", "ESS", "EXR",
]

ALL = sorted(set(SMALL_MID + SMALL_TECH + INTL))
print(f"Downloading {len(ALL)} tickers to {OUT}")

bad, ok = [], []
for t in ALL:
    fp = os.path.join(OUT, f"{t}.csv")
    if os.path.exists(fp):
        ok.append(t); continue
    try:
        df = yf.download(t, start="2009-01-01", end="2026-05-08",
                         progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            bad.append(t); continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "Close"]].dropna()
        if len(df) < 250:
            bad.append(t); continue
        df.index.name = "Date"
        df.reset_index().to_csv(fp, index=False)
        ok.append(t)
        print(f"  OK {t}: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} bars)")
    except Exception as e:
        print(f"  FAIL {t}: {e}")
        bad.append(t)

print(f"\nDone. {len(ok)} OK, {len(bad)} failed: {bad}")
