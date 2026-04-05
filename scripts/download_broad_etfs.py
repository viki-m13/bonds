"""
Download massive ETF universe across ALL asset classes.
This gives us 100+ instruments for cross-asset diversification.
"""
import yfinance as yf
import pandas as pd
import os
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'etfs')

BROAD_ETFS = {
    # === EQUITY SECTORS ===
    "XLF": "Financial Select Sector",
    "XLK": "Technology Select Sector",
    "XLE": "Energy Select Sector",
    "XLV": "Health Care Select Sector",
    "XLI": "Industrial Select Sector",
    "XLP": "Consumer Staples Select Sector",
    "XLY": "Consumer Discretionary Select Sector",
    "XLU": "Utilities Select Sector",
    "XLB": "Materials Select Sector",
    "XLRE": "Real Estate Select Sector",
    "XLC": "Communication Services Select Sector",
    # === BROAD EQUITY ===
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000 Small Cap",
    "MDY": "S&P MidCap 400",
    "DIA": "Dow Jones Industrial",
    # === INTERNATIONAL EQUITY ===
    "EFA": "MSCI EAFE (Developed ex-US)",
    "EEM": "MSCI Emerging Markets Equity",
    "VWO": "Vanguard FTSE Emerging Markets",
    "VEA": "Vanguard FTSE Developed Markets",
    "FXI": "China Large Cap",
    "EWJ": "Japan",
    "EWG": "Germany",
    "EWU": "United Kingdom",
    "EWZ": "Brazil",
    "EWY": "South Korea",
    "EWT": "Taiwan",
    "INDA": "India",
    # === COMMODITIES ===
    "GLD": "Gold",
    "SLV": "Silver",
    "USO": "United States Oil Fund",
    "UNG": "United States Natural Gas",
    "DBA": "Agriculture",
    "DBC": "Commodities Index",
    "PDBC": "Optimum Yield Diversified Commodity",
    "CPER": "Copper",
    "WEAT": "Wheat",
    "CORN": "Corn",
    # === REITS ===
    "VNQ": "Vanguard Real Estate",
    "IYR": "iShares US Real Estate",
    "VNQI": "Vanguard Global ex-US Real Estate",
    "REM": "iShares Mortgage Real Estate",
    "MORT": "VanEck Mortgage REIT Income",
    # === ALTERNATIVES / VOLATILITY ===
    "VIXY": "ProShares VIX Short-Term Futures",
    "BTAL": "AGFiQ US Market Neutral Anti-Beta",
    "QAI": "IQ Hedge Multi-Strategy Tracker",
    "MNA": "IQ Merger Arbitrage",
    # === CURRENCY ===
    "UUP": "Invesco DB US Dollar Index Bullish",
    "FXE": "CurrencyShares Euro Trust",
    "FXY": "CurrencyShares Japanese Yen Trust",
    "FXB": "CurrencyShares British Pound Trust",
    "FXA": "CurrencyShares Australian Dollar",
    "FXC": "CurrencyShares Canadian Dollar",
    "CYB": "WisdomTree Chinese Yuan Strategy",
    "CEW": "WisdomTree Emerging Currency Strategy",
    # === LEVERAGED / INVERSE (for hedging signals) ===
    "TBT": "ProShares UltraShort 20+ Year Treasury",
    "TMF": "Direxion Daily 20+ Year Treasury Bull 3X",
    # === DIVIDEND / INCOME ===
    "DVY": "iShares Select Dividend",
    "VIG": "Vanguard Dividend Appreciation",
    "SCHD": "Schwab US Dividend Equity",
    "HDV": "iShares Core High Dividend",
    # === PREFERRED STOCK (bond-like equity) ===
    "PFF": "iShares Preferred & Income Securities",
    "PGX": "Invesco Preferred",
    # === CONVERTIBLE BONDS ===
    "CWB": "SPDR Bloomberg Convertible Securities",
    # === INTERNATIONAL BONDS ===
    "BNDX": "Vanguard Total International Bond",
    "IGOV": "iShares International Treasury Bond",
    "EMLC": "VanEck JP Morgan EM Local Currency Bond",
    "PCY": "Invesco Emerging Markets Sovereign Debt",
    # === INFLATION / REAL ASSETS ===
    "RINF": "ProShares Inflation Expectations",
    "IVOL": "Quadratic Interest Rate Volatility",
    # === BANK LOANS ===
    "BKLN": "Invesco Senior Loan",
    "SRLN": "SPDR Blackstone Senior Loan",
}


def download_etf(ticker):
    print(f"  {ticker}...", end=" ", flush=True)
    try:
        data = yf.download(ticker, start="2005-01-01", progress=False)
        if data.empty:
            print("NO DATA")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.index.name = "Date"
        print(f"{len(data)} rows")
        return data
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading {len(BROAD_ETFS)} broad ETFs...")
    
    success = 0
    for ticker, name in BROAD_ETFS.items():
        # Skip if we already have it
        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        data = download_etf(ticker)
        if data is not None:
            data.to_csv(path)
            success += 1

    print(f"\nDownloaded {success}/{len(BROAD_ETFS)} ETFs")


if __name__ == "__main__":
    main()
