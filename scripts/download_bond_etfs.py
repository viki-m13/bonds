"""
Download historical price data for bond ETFs that trade like stocks.

These ETFs represent different segments of the bond market and are
liquid enough to actively trade. This is the core dataset for building
a bond trading model.

ETF Categories:
- Government/Treasury: TLT, IEF, SHY, VGLT, GOVT, SPTL
- Corporate Investment Grade: LQD, VCIT, VCSH, IGIB
- High Yield: HYG, JNK, USHY
- Aggregate/Total Bond: AGG, BND
- TIPS (Inflation Protected): TIP, SCHP
- Municipal: MUB, VTEB
- Emerging Market: EMB, VWOB
- Floating Rate: FLOT, FLRN
- Mortgage-Backed: MBB, VMBS
"""

import yfinance as yf
import pandas as pd
import os
import sys
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'etfs')

BOND_ETFS = {
    # US Treasury - different durations
    'SHY':  {'name': 'iShares 1-3 Year Treasury Bond',       'category': 'treasury_short'},
    'IEI':  {'name': 'iShares 3-7 Year Treasury Bond',       'category': 'treasury_medium'},
    'IEF':  {'name': 'iShares 7-10 Year Treasury Bond',      'category': 'treasury_intermediate'},
    'TLH':  {'name': 'iShares 10-20 Year Treasury Bond',     'category': 'treasury_long'},
    'TLT':  {'name': 'iShares 20+ Year Treasury Bond',       'category': 'treasury_ultra_long'},
    'GOVT': {'name': 'iShares US Treasury Bond',              'category': 'treasury_broad'},
    'SPTL': {'name': 'SPDR Portfolio Long Term Treasury',     'category': 'treasury_long'},
    'VGLT': {'name': 'Vanguard Long-Term Treasury',           'category': 'treasury_long'},

    # Corporate Investment Grade
    'LQD':  {'name': 'iShares Investment Grade Corporate',    'category': 'corp_ig'},
    'VCIT': {'name': 'Vanguard Intermediate-Term Corp Bond',  'category': 'corp_ig'},
    'VCSH': {'name': 'Vanguard Short-Term Corp Bond',         'category': 'corp_ig_short'},
    'IGIB': {'name': 'iShares 5-10 Year IG Corp Bond',       'category': 'corp_ig'},

    # High Yield (Junk Bonds)
    'HYG':  {'name': 'iShares High Yield Corporate Bond',    'category': 'high_yield'},
    'JNK':  {'name': 'SPDR Bloomberg High Yield Bond',       'category': 'high_yield'},
    'USHY': {'name': 'iShares Broad USD High Yield Corp',    'category': 'high_yield'},

    # Aggregate / Total Bond Market
    'AGG':  {'name': 'iShares Core US Aggregate Bond',       'category': 'aggregate'},
    'BND':  {'name': 'Vanguard Total Bond Market',           'category': 'aggregate'},

    # TIPS - Inflation Protected
    'TIP':  {'name': 'iShares TIPS Bond',                    'category': 'tips'},
    'SCHP': {'name': 'Schwab US TIPS',                       'category': 'tips'},

    # Municipal Bonds
    'MUB':  {'name': 'iShares National Muni Bond',           'category': 'municipal'},
    'VTEB': {'name': 'Vanguard Tax-Exempt Bond',             'category': 'municipal'},

    # Emerging Market Bonds
    'EMB':  {'name': 'iShares JP Morgan USD EM Bond',        'category': 'emerging_market'},
    'VWOB': {'name': 'Vanguard Emerging Markets Govt Bond',  'category': 'emerging_market'},

    # Floating Rate
    'FLOT': {'name': 'iShares Floating Rate Bond',           'category': 'floating_rate'},

    # Mortgage-Backed Securities
    'MBB':  {'name': 'iShares MBS',                          'category': 'mbs'},
    'VMBS': {'name': 'Vanguard Mortgage-Backed Securities',  'category': 'mbs'},
}


def download_etf_data(ticker, start='2005-01-01', end=None):
    """Download historical OHLCV data for a single ETF."""
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')

    print(f"  Downloading {ticker}...", end=' ', flush=True)
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            print("NO DATA")
            return None

        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.index.name = 'Date'
        print(f"{len(data)} rows ({data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')})")
        return data
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 70)
    print("DOWNLOADING BOND ETF HISTORICAL DATA")
    print("=" * 70)

    summary = []

    for ticker, info in BOND_ETFS.items():
        data = download_etf_data(ticker)
        if data is not None:
            filepath = os.path.join(DATA_DIR, f'{ticker}.csv')
            data.to_csv(filepath)
            summary.append({
                'ticker': ticker,
                'name': info['name'],
                'category': info['category'],
                'rows': len(data),
                'start': data.index[0].strftime('%Y-%m-%d'),
                'end': data.index[-1].strftime('%Y-%m-%d'),
            })

    # Save summary/metadata
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(DATA_DIR, '_metadata.csv'), index=False)
        print(f"\n{'=' * 70}")
        print(f"Downloaded {len(summary)} ETFs successfully")
        print(f"Data saved to: {os.path.abspath(DATA_DIR)}")
        print(f"{'=' * 70}")
    else:
        print("\nNo data was downloaded!")
        sys.exit(1)


if __name__ == '__main__':
    main()
