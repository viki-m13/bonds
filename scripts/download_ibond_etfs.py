"""
Download historical price data for iBonds (target maturity) ETFs.

iBonds ETFs hold bonds maturing in a specific year, making them behave
more like individual bonds than traditional bond ETFs. They roll down
the yield curve over time and mature at par value.

This gives us the closest thing to individual bond trading with
stock-like liquidity.
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'etfs')

# iShares iBonds - Corporate target maturity
IBOND_ETFS = {
    # iBonds Dec Corporate (investment grade, mature in December of given year)
    'IBDQ': {'name': 'iBonds Dec 2025 Term Corporate', 'maturity': 2025, 'type': 'corp'},
    'IBDR': {'name': 'iBonds Dec 2026 Term Corporate', 'maturity': 2026, 'type': 'corp'},
    'IBDS': {'name': 'iBonds Dec 2027 Term Corporate', 'maturity': 2027, 'type': 'corp'},
    'IBDT': {'name': 'iBonds Dec 2028 Term Corporate', 'maturity': 2028, 'type': 'corp'},
    'IBDU': {'name': 'iBonds Dec 2029 Term Corporate', 'maturity': 2029, 'type': 'corp'},
    'IBDV': {'name': 'iBonds Dec 2030 Term Corporate', 'maturity': 2030, 'type': 'corp'},
    'IBDW': {'name': 'iBonds Dec 2031 Term Corporate', 'maturity': 2031, 'type': 'corp'},
    'IBDX': {'name': 'iBonds Dec 2032 Term Corporate', 'maturity': 2032, 'type': 'corp'},
    'IBDY': {'name': 'iBonds Dec 2033 Term Corporate', 'maturity': 2033, 'type': 'corp'},
    'IBHF': {'name': 'iBonds Dec 2026 Term HY & Income', 'maturity': 2026, 'type': 'hy'},
    'IBHG': {'name': 'iBonds Dec 2027 Term HY & Income', 'maturity': 2027, 'type': 'hy'},
    'IBHH': {'name': 'iBonds Dec 2028 Term HY & Income', 'maturity': 2028, 'type': 'hy'},
    'IBHI': {'name': 'iBonds Dec 2029 Term HY & Income', 'maturity': 2029, 'type': 'hy'},

    # iBonds Dec Treasury (target maturity treasuries)
    'IBTF': {'name': 'iBonds Dec 2025 Term Treasury', 'maturity': 2025, 'type': 'treasury'},
    'IBTG': {'name': 'iBonds Dec 2026 Term Treasury', 'maturity': 2026, 'type': 'treasury'},
    'IBTH': {'name': 'iBonds Dec 2027 Term Treasury', 'maturity': 2027, 'type': 'treasury'},
    'IBTI': {'name': 'iBonds Dec 2028 Term Treasury', 'maturity': 2028, 'type': 'treasury'},
    'IBTJ': {'name': 'iBonds Dec 2029 Term Treasury', 'maturity': 2029, 'type': 'treasury'},
    'IBTK': {'name': 'iBonds Dec 2030 Term Treasury', 'maturity': 2030, 'type': 'treasury'},
    'IBTL': {'name': 'iBonds Dec 2031 Term Treasury', 'maturity': 2031, 'type': 'treasury'},
    'IBTM': {'name': 'iBonds Dec 2032 Term Treasury', 'maturity': 2032, 'type': 'treasury'},

    # Invesco BulletShares Corporate (similar to iBonds)
    'BSCQ': {'name': 'BulletShares 2026 Corp Bond',   'maturity': 2026, 'type': 'corp'},
    'BSCR': {'name': 'BulletShares 2027 Corp Bond',   'maturity': 2027, 'type': 'corp'},
    'BSCS': {'name': 'BulletShares 2028 Corp Bond',   'maturity': 2028, 'type': 'corp'},
    'BSCT': {'name': 'BulletShares 2029 Corp Bond',   'maturity': 2029, 'type': 'corp'},
    'BSCU': {'name': 'BulletShares 2030 Corp Bond',   'maturity': 2030, 'type': 'corp'},
    'BSCV': {'name': 'BulletShares 2031 Corp Bond',   'maturity': 2031, 'type': 'corp'},
    'BSCW': {'name': 'BulletShares 2032 Corp Bond',   'maturity': 2032, 'type': 'corp'},
    'BSCX': {'name': 'BulletShares 2033 Corp Bond',   'maturity': 2033, 'type': 'corp'},
    'BSCY': {'name': 'BulletShares 2034 Corp Bond',   'maturity': 2034, 'type': 'corp'},

    # Invesco BulletShares High Yield
    'BSJQ': {'name': 'BulletShares 2026 HY Corp Bond', 'maturity': 2026, 'type': 'hy'},
    'BSJR': {'name': 'BulletShares 2027 HY Corp Bond', 'maturity': 2027, 'type': 'hy'},
    'BSJS': {'name': 'BulletShares 2028 HY Corp Bond', 'maturity': 2028, 'type': 'hy'},
    'BSJT': {'name': 'BulletShares 2029 HY Corp Bond', 'maturity': 2029, 'type': 'hy'},
    'BSJU': {'name': 'BulletShares 2030 HY Corp Bond', 'maturity': 2030, 'type': 'hy'},
    'BSJV': {'name': 'BulletShares 2031 HY Corp Bond', 'maturity': 2031, 'type': 'hy'},
}


def download_etf_data(ticker):
    """Download historical OHLCV data for a single ETF."""
    print(f"  Downloading {ticker}...", end=' ', flush=True)
    try:
        data = yf.download(ticker, start='2010-01-01', progress=False)
        if data.empty:
            print("NO DATA")
            return None

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
    print("DOWNLOADING iBOND / TARGET MATURITY ETF DATA")
    print("=" * 70)

    summary = []

    for ticker, info in IBOND_ETFS.items():
        data = download_etf_data(ticker)
        if data is not None:
            filepath = os.path.join(DATA_DIR, f'{ticker}.csv')
            data.to_csv(filepath)
            summary.append({
                'ticker': ticker,
                'name': info['name'],
                'category': f'ibond_{info["type"]}',
                'maturity_year': info['maturity'],
                'rows': len(data),
                'start': data.index[0].strftime('%Y-%m-%d'),
                'end': data.index[-1].strftime('%Y-%m-%d'),
            })

    # Append to existing metadata
    if summary:
        new_meta = pd.DataFrame(summary)
        existing_meta_path = os.path.join(DATA_DIR, '_metadata.csv')
        if os.path.exists(existing_meta_path):
            existing = pd.read_csv(existing_meta_path)
            # Remove any old ibond entries
            existing = existing[~existing['ticker'].isin(new_meta['ticker'])]
            combined = pd.concat([existing, new_meta], ignore_index=True)
        else:
            combined = new_meta
        combined.to_csv(existing_meta_path, index=False)

        print(f"\n{'=' * 70}")
        print(f"Downloaded {len(summary)} iBond/target maturity ETFs")
        print(f"{'=' * 70}")
    else:
        print("\nNo iBond data was downloaded!")


if __name__ == '__main__':
    main()
