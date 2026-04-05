"""Master script to download all bond market data."""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

scripts = [
    'download_bond_etfs.py',
    'download_ibond_etfs.py',
    'download_treasury_yields.py',
    'download_fred_data.py',
]

for script in scripts:
    print(f"\n{'#' * 70}")
    print(f"# Running {script}")
    print(f"{'#' * 70}\n")
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, script)],
        cwd=SCRIPT_DIR,
    )
    if result.returncode != 0:
        print(f"\nWARNING: {script} exited with code {result.returncode}")

print("\n\nAll downloads complete!")
