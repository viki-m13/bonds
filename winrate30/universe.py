"""Stock universe for the winrate30 tool.

~300 liquid US large/mid caps across all sectors. This is a *current* list,
so backtests on it carry survivorship bias (companies that failed or were
acquired are missing). Mitigations: the universe is restricted to large caps
(which rarely vanish outright), and validation emphasizes recent walk-forward
windows. See README for the full caveat.
"""

TECH = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ORCL", "CRM",
    "ADBE", "AMD", "INTC", "QCOM", "TXN", "AMAT", "MU", "LRCX", "KLAC", "ADI",
    "NXPI", "INTU", "NOW", "PANW", "SNPS", "CDNS", "CSCO", "IBM", "ACN",
    "ADSK", "ANET", "FTNT", "MSI", "APH", "TEL", "GLW", "HPQ", "ON", "MCHP",
    "SWKS", "QRVO", "ZBRA", "KEYS", "IT", "CTSH",
]

COMMUNICATION = [
    "T", "VZ", "TMUS", "CMCSA", "DIS", "NFLX", "CHTR", "EA", "TTWO", "OMC",
    "IPG",
]

FINANCIALS = [
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA",
    "PYPL", "SPGI", "MCO", "ICE", "CME", "COF", "USB", "PNC", "TFC", "BK",
    "STT", "AIG", "MET", "PRU", "AFL", "ALL", "TRV", "CB", "PGR", "MMC",
    "AON", "AJG", "CINF", "TROW", "NTRS", "FITB", "KEY", "RF", "HBAN", "MTB",
]

HEALTHCARE = [
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "ABT", "TMO", "DHR", "BMY",
    "AMGN", "GILD", "CVS", "CI", "ELV", "HUM", "CNC", "MCK", "CAH", "BAX",
    "BDX", "BSX", "MDT", "SYK", "EW", "ISRG", "ZBH", "RMD", "IDXX", "A",
    "IQV", "MTD", "WAT", "STE", "HOLX", "DGX", "LH", "VRTX", "REGN", "BIIB",
    "ZTS", "HCA", "UHS", "DVA",
]

STAPLES = [
    "PG", "KO", "PEP", "COST", "WMT", "TGT", "MDLZ", "CL", "KMB", "GIS",
    "HSY", "SJM", "CAG", "CPB", "HRL", "MKC", "TSN", "KR", "SYY", "STZ",
    "TAP", "BF-B", "EL", "CHD", "CLX", "MO", "PM", "MNST", "DG", "DLTR",
]

ENERGY = [
    "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY", "HAL",
    "BKR", "DVN", "FANG", "HES", "KMI", "WMB", "OKE", "TRGP", "APA",
]

INDUSTRIALS = [
    "GE", "BA", "CAT", "DE", "HON", "UNP", "UPS", "FDX", "LMT", "RTX", "NOC",
    "GD", "LHX", "MMM", "EMR", "ETN", "ITW", "PH", "CMI", "PCAR", "ROK",
    "DOV", "XYL", "AME", "ROP", "IR", "CSX", "NSC", "ODFL", "JBHT", "CHRW",
    "EXPD", "WM", "RSG", "WAB", "TDG", "MAS", "PNR", "SWK", "FAST", "GWW",
    "URI", "PWR", "EME", "LII", "AOS", "NDSN",
]

MATERIALS = [
    "LIN", "APD", "SHW", "ECL", "PPG", "NUE", "STLD", "FCX", "NEM", "VMC",
    "MLM", "ALB", "IFF", "MOS", "CF", "LYB",
]

REITS = [
    "AMT", "PLD", "CCI", "EQIX", "PSA", "O", "SPG", "WELL", "VTR", "AVB",
    "EQR", "ESS", "MAA", "UDR", "DLR", "ARE", "BXP", "KIM", "REG", "FRT",
    "HST", "EXR", "CPT", "IRM", "WY",
]

UTILITIES = [
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "WEC", "ES",
    "PEG", "EIX", "DTE", "PPL", "AEE", "CMS", "CNP", "ATO", "NI", "LNT",
    "FE", "PNW",
]

DISCRETIONARY = [
    "HD", "LOW", "MCD", "SBUX", "NKE", "TJX", "ROST", "YUM", "CMG", "ORLY",
    "AZO", "BBY", "EBAY", "MAR", "HLT", "RCL", "CCL", "LVS", "WYNN", "MGM",
    "DRI", "DPZ", "POOL", "TSCO", "ULTA", "LULU", "DECK", "GRMN", "HAS",
    "WHR", "GPC", "LEN", "DHI", "PHM", "NVR", "TOL", "F", "GM", "TSLA",
    "APTV", "BWA", "RL", "PVH",
]

UNIVERSE = sorted(set(
    TECH + COMMUNICATION + FINANCIALS + HEALTHCARE + STAPLES + ENERGY
    + INDUSTRIALS + MATERIALS + REITS + UTILITIES + DISCRETIONARY
))

# Context series (not tradeable members of the universe)
MARKET_TICKER = "SPY"
VIX_TICKER = "^VIX"
