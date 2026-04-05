#!/usr/bin/env python3
"""
Daily dashboard update script.
Downloads fresh data, re-runs strategy, and regenerates the HTML page.
Designed to run via GitHub Actions cron or manually.
"""

import subprocess
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent

def run(cmd, desc=""):
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / cmd)],
        cwd=str(ROOT_DIR),
        capture_output=True, text=True, timeout=600,
    )
    if result.stdout:
        # Print last 20 lines
        lines = result.stdout.strip().split("\n")
        for line in lines[-20:]:
            print(f"  {line}")
    if result.returncode != 0:
        print(f"  WARNING: {cmd} exited with code {result.returncode}")
        if result.stderr:
            print(f"  STDERR: {result.stderr[-500:]}")
    return result.returncode


def main():
    print("DICHS Dashboard Daily Update")
    print(f"Working directory: {ROOT_DIR}")

    # Step 1: Download fresh ETF data
    run("download_bond_etfs.py", "Updating bond ETF prices")

    # Step 2: Download fresh FRED data
    run("download_fred_data.py", "Updating FRED data")

    # Step 3: Re-run the strategy
    run("strategy_final.py", "Re-running DICHS strategy")

    # Step 4: Regenerate dashboard HTML
    print(f"\n{'='*60}\nRegenerating dashboard HTML\n{'='*60}")

    # Run the dashboard generation inline
    subprocess.run(
        [sys.executable, "-c", f"""
import json, sys
sys.path.insert(0, "{SCRIPT_DIR}")

# Re-generate dashboard data
exec(open("{SCRIPT_DIR}/generate_dashboard_data.py").read())
"""],
        cwd=str(ROOT_DIR), timeout=300,
    )

    print("\nUpdate complete!")


if __name__ == "__main__":
    main()
