"""
Download key bond market indicators from FRED (Federal Reserve Economic Data).

Uses the FRED API directly via HTTP (no API key needed for basic CSV downloads).
These series provide context for bond trading: spreads, rates, economic indicators.
"""

import pandas as pd
import requests
import os
import sys
from io import StringIO

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'fred')

# Key FRED series for bond trading
FRED_SERIES = {
    # Treasury Yields (constant maturity)
    'DGS1MO':  'Treasury Yield 1-Month',
    'DGS3MO':  'Treasury Yield 3-Month',
    'DGS6MO':  'Treasury Yield 6-Month',
    'DGS1':    'Treasury Yield 1-Year',
    'DGS2':    'Treasury Yield 2-Year',
    'DGS5':    'Treasury Yield 5-Year',
    'DGS7':    'Treasury Yield 7-Year',
    'DGS10':   'Treasury Yield 10-Year',
    'DGS20':   'Treasury Yield 20-Year',
    'DGS30':   'Treasury Yield 30-Year',

    # Key Spreads
    'T10Y2Y':   '10-Year minus 2-Year Treasury Spread',
    'T10Y3M':   '10-Year minus 3-Month Treasury Spread',
    'T10YFF':   '10-Year minus Fed Funds Rate',

    # Credit Spreads
    'BAMLC0A0CM':    'ICE BofA US Corporate Index OAS',
    'BAMLH0A0HYM2':  'ICE BofA US High Yield Index OAS',
    'BAMLC0A4CBBB':  'ICE BofA BBB US Corporate Index OAS',
    'BAMLC0A1CAAA':  'ICE BofA AAA US Corporate Index OAS',

    # Fed Funds & Policy Rates
    'FEDFUNDS':  'Federal Funds Effective Rate',
    'DFEDTARU':  'Fed Funds Target Rate Upper',
    'DFEDTARL':  'Fed Funds Target Rate Lower',

    # Inflation
    'T5YIE':   '5-Year Breakeven Inflation Rate',
    'T10YIE':  '10-Year Breakeven Inflation Rate',
    'CPIAUCSL': 'CPI All Urban Consumers',

    # Economic Indicators (useful for bond trading signals)
    'UNRATE':   'Unemployment Rate',
    'VIXCLS':   'CBOE Volatility Index (VIX)',
    'DTWEXBGS': 'Trade Weighted US Dollar Index',
}


def download_fred_series(series_id, start='2000-01-01'):
    """Download a single FRED series via direct CSV URL."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and ('DATE' in resp.text[:200].upper() or 'observation_date' in resp.text[:200]):
            df = pd.read_csv(StringIO(resp.text))
            # Normalize column names
            df.columns = ['Date', series_id]
            df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.dropna()
            return df
    except Exception as e:
        print(f"    Error: {e}")
    return None


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 70)
    print("DOWNLOADING FRED BOND MARKET DATA")
    print("=" * 70)

    all_series = {}
    failed = []

    for series_id, description in FRED_SERIES.items():
        print(f"  {series_id}: {description}...", end=' ', flush=True)
        df = download_fred_series(series_id)
        if df is not None:
            print(f"{len(df)} observations")
            all_series[series_id] = df
            # Save individual series
            df.to_csv(os.path.join(DATA_DIR, f'{series_id}.csv'), index=False)
        else:
            print("FAILED")
            failed.append(series_id)

    # Create a combined wide-format dataset (aligned by date)
    if all_series:
        print("\nCreating combined dataset...")
        combined = None
        for series_id, df in all_series.items():
            df_renamed = df.set_index('Date')
            if combined is None:
                combined = df_renamed
            else:
                combined = combined.join(df_renamed, how='outer')

        combined = combined.sort_index()
        combined.to_csv(os.path.join(DATA_DIR, '_combined_fred.csv'))
        print(f"Combined dataset: {combined.shape[0]} dates x {combined.shape[1]} series")

    # Save metadata
    metadata = pd.DataFrame([
        {'series_id': k, 'description': v, 'downloaded': k in all_series}
        for k, v in FRED_SERIES.items()
    ])
    metadata.to_csv(os.path.join(DATA_DIR, '_metadata.csv'), index=False)

    print(f"\n{'=' * 70}")
    print(f"Downloaded {len(all_series)}/{len(FRED_SERIES)} FRED series")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"Data saved to: {os.path.abspath(DATA_DIR)}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
