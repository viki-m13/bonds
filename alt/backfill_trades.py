"""Backfill missed market days in the trade log.

When the cron hasn't run for several weekdays, the trade log has gaps. This
script fills them by replaying live_signal.py for each missed market day,
generating the trade entries that would have been recorded if the cron had
run on each of those days.

Usage:
    python3 alt/backfill_trades.py           # backfill from last log entry to today
    python3 alt/backfill_trades.py --since 2026-04-13   # explicit start date

Each backfilled day runs live_signal.py --as-of YYYY-MM-DD --skip-fetch.
The signal computes weights based on data up to that date's close, compares
to the previous day's logged positions, and appends any trades above the
0.5% threshold. Positions and trades CSVs grow naturally — no existing
history is overwritten.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
R = ROOT / "data/results"
ALT = ROOT / "alt"


def latest_market_date() -> date | None:
    """Latest common market close across SPY/QQQ/IBIT."""
    ETF = ROOT / "data/etfs"
    dates = []
    for t in ["SPY", "QQQ"]:
        p = ETF / f"{t}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["Date"])
        dates.append(df["Date"].iloc[-1].date())
    return min(dates) if dates else None


def market_days_between(start: date, end: date, universe_p: Path) -> list[date]:
    """List of market-close dates from start (exclusive) to end (inclusive) using SPY's calendar."""
    spy_p = ROOT / "data/etfs/SPY.csv"
    if not spy_p.exists():
        return []
    df = pd.read_csv(spy_p, parse_dates=["Date"])
    dates = df["Date"].dt.date.tolist()
    return [d for d in dates if start < d <= end]


def last_trade_date() -> date | None:
    p = R / "live_trades.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"])
    if len(df) == 0:
        return None
    return df["Date"].max().date()


def last_position_date() -> date | None:
    p = R / "live_positions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"])
    if len(df) == 0:
        return None
    return df["Date"].max().date()


def run_signal_asof(d: date) -> bool:
    """Run live_signal --as-of d --skip-fetch. Returns True on success."""
    proc = subprocess.run(
        [sys.executable, str(ALT / "live_signal.py"),
         "--as-of", d.isoformat(),
         "--skip-fetch"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"  FAIL on {d}: {proc.stderr[:400]}")
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--since", type=str, default=None,
                   help="Start date (exclusive). Defaults to latest date already in live_positions.csv.")
    p.add_argument("--until", type=str, default=None,
                   help="End date (inclusive). Defaults to latest market date.")
    args = p.parse_args()

    start = (date.fromisoformat(args.since) if args.since
             else last_position_date())
    end = (date.fromisoformat(args.until) if args.until
           else latest_market_date())
    if start is None:
        start = date(2026, 4, 1)  # default
    if end is None:
        print("No latest market date available; aborting.")
        return 1

    days = market_days_between(start, end, ROOT / "data/etfs/SPY.csv")
    if not days:
        print(f"No market days to backfill (start={start}, end={end}).")
        return 0

    print(f"Backfilling {len(days)} market day(s): {days[0]} → {days[-1]}")
    print("(Each replay appends to live_positions.csv / live_trades.csv)")
    print()

    n_ok = 0
    for d in days:
        print(f"  Replaying signal as-of {d}...", end=" ", flush=True)
        if run_signal_asof(d):
            print("OK")
            n_ok += 1
        else:
            print("failed (see stderr)")

    print(f"\nCompleted: {n_ok}/{len(days)} days backfilled.")
    if n_ok > 0:
        # Dedupe the trade log on (Date, ticker, side) in case of re-runs
        tp = R / "live_trades.csv"
        if tp.exists():
            df = pd.read_csv(tp, parse_dates=["Date"])
            before = len(df)
            df = df.drop_duplicates(subset=["Date", "ticker", "side"], keep="last").sort_values(["Date", "ticker"])
            df.to_csv(tp, index=False)
            print(f"Trade log: deduped {before} → {len(df)} rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
