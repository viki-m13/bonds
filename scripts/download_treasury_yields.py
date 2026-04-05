"""
Download US Treasury yield curve data from the US Treasury website.

This provides daily yield curve rates for various maturities, which is
essential for understanding rate movements, yield curve shape, and
relative value in bond trading.

Source: https://home.treasury.gov/resource-center/data-chart-center/interest-rates
"""

import pandas as pd
import requests
import os
import sys
from datetime import datetime
import xml.etree.ElementTree as ET

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'treasury')


def download_treasury_yields_csv():
    """
    Download Treasury yield curve data from Treasury.gov XML feed.
    Returns daily yields for 1mo, 2mo, 3mo, 4mo, 6mo, 1yr, 2yr, 3yr,
    5yr, 7yr, 10yr, 20yr, 30yr maturities.
    """
    print("Downloading Treasury yield curve data...")

    all_data = []

    # Treasury.gov provides XML data by year
    current_year = datetime.now().year
    for year in range(2000, current_year + 1):
        url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
        print(f"  Fetching {year}...", end=' ', flush=True)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 100:
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))
                all_data.append(df)
                print(f"{len(df)} rows")
            else:
                print(f"status {resp.status_code}, skipping")
        except Exception as e:
            print(f"error: {e}")

    if not all_data:
        print("No Treasury yield data downloaded!")
        return None

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal: {len(combined)} rows of yield curve data")
    return combined


def download_treasury_par_yields():
    """Download Treasury par yield curve rates (used for bond pricing)."""
    print("\nDownloading Treasury par yield curve data...")

    all_data = []
    current_year = datetime.now().year

    for year in range(2000, current_year + 1):
        url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_par_yield_curve&field_tdr_date_value={year}&page&_format=csv"
        print(f"  Fetching {year}...", end=' ', flush=True)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 100:
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))
                all_data.append(df)
                print(f"{len(df)} rows")
            else:
                print(f"status {resp.status_code}, skipping")
        except Exception as e:
            print(f"error: {e}")

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Total: {len(combined)} rows of par yield data")
    return combined


def download_real_yields():
    """Download Treasury real yield curve (TIPS yields)."""
    print("\nDownloading Treasury real yield curve (TIPS) data...")

    all_data = []
    current_year = datetime.now().year

    for year in range(2003, current_year + 1):  # Real yields available from 2003
        url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_real_yield_curve&field_tdr_date_value={year}&page&_format=csv"
        print(f"  Fetching {year}...", end=' ', flush=True)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 100:
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))
                all_data.append(df)
                print(f"{len(df)} rows")
            else:
                print(f"status {resp.status_code}, skipping")
        except Exception as e:
            print(f"error: {e}")

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Total: {len(combined)} rows of real yield data")
    return combined


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 70)
    print("DOWNLOADING US TREASURY YIELD DATA")
    print("=" * 70)

    # Nominal yields
    yields = download_treasury_yields_csv()
    if yields is not None:
        filepath = os.path.join(DATA_DIR, 'daily_treasury_yields.csv')
        yields.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")

    # Real yields (TIPS)
    real = download_real_yields()
    if real is not None:
        filepath = os.path.join(DATA_DIR, 'daily_treasury_real_yields.csv')
        real.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")

    print(f"\n{'=' * 70}")
    print("Treasury yield data download complete")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
