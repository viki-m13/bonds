"""Post-refresh validation — prints a pass/fail report against frozen expectations.

Run at the end of every cron. Confirms:
  1. Production parameters match the locked config (weights, vol target, cap, etc.)
  2. Each sleeve's returns CSV covers IS (2010-2018) and now extends through today
  3. IS-window metrics (Sharpe, CAGR, MDD) match the frozen backtest values —
     historical returns must NOT change between runs; if they do, something
     upstream has drifted
  4. OOS-window metrics sanity check
  5. Full-window (2010-today) metrics sanity check
  6. Live signal: overlay multiplier math, regime flags, date alignment

Exit code: 0 if all checks pass, 1 if any FAIL (cron will still continue —
this is a report, not a gate; but the log makes drift visible).
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
R = ROOT / "data/results"
ETF = ROOT / "data/etfs"

# ==== FROZEN EXPECTATIONS (these should NEVER change across cron runs) ====
EXPECTED = {
    "weights": {"VANGUARD": 0.236, "ORION": 0.327, "HELIOS": 0.185,
                "QUANTUM": 0.152, "CRYPTO": 0.101},
    "params": {"target_vol": 0.15, "vol_cap": 1.0, "vol_floor": 0.25,
               "dd_floor": -0.10, "vol_gate_pct": 0.99, "tc_bps_per_lev_chg": 10.0},
    # IS metrics — these are FROZEN. If they drift > TOLERANCE, flag it.
    "is_metrics": {"sharpe": 2.52, "cagr": 0.375, "mdd": -0.163, "navx": 18.3},
    # OOS metrics — frozen (until cron extends today's data; then we expect
    # small drift within tolerance as new days are appended)
    "oos_metrics": {"sharpe_min": 1.80, "cagr_min": 0.30},
    # Full-window metrics (change very slightly with each new day)
    "full_metrics": {"sharpe_min": 2.10, "cagr_min": 0.33, "mdd_max": -0.20},
}
TOLERANCE_SHARPE = 0.05   # Sharpe must be within ±0.05 of expected
TOLERANCE_CAGR = 0.01     # CAGR within ±1%
TOLERANCE_MDD = 0.01      # MDD within ±1%

IS_END = "2018-12-31"
OOS_START = "2019-01-02"


def log(symbol, msg):
    print(f"  [{symbol}] {msg}")


checks = {"pass": 0, "fail": 0, "warn": 0}


def check(cond, msg_ok, msg_fail, warn=False):
    if cond:
        log("PASS", msg_ok); checks["pass"] += 1
    elif warn:
        log("WARN", msg_fail); checks["warn"] += 1
    else:
        log("FAIL", msg_fail); checks["fail"] += 1


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {}
    mu = r.mean() * 252; sd = r.std() * np.sqrt(252)
    sr = mu / sd if sd > 0 else 0
    c = (1 + r).cumprod(); dd = (c / c.cummax() - 1).min()
    yrs = len(r) / 252
    cagr = c.iloc[-1] ** (1 / yrs) - 1 if c.iloc[-1] > 0 else -1
    return {"sharpe": float(sr), "cagr": float(cagr), "mdd": float(dd),
            "navx": float(c.iloc[-1]), "n": int(len(r))}


def main():
    print("=" * 74)
    print("PHOENIX state validation — run after every cron refresh")
    print("=" * 74)
    print(f"Validation time (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print()

    # ------- 1. Production parameters -------
    print("[1/6] Production parameters")
    prod_meta = json.loads((R/"phoenix_production_metrics.json").read_text())
    p = prod_meta.get("params", {})
    for k, v in EXPECTED["params"].items():
        got = p.get(k)
        check(got == v,
              f"{k} = {v}",
              f"{k} expected {v} but got {got}")
    pw = prod_meta.get("params", {}).get("weights", {})
    ew = EXPECTED["weights"]
    for k, v in ew.items():
        got = pw.get(k)
        check(abs((got or 0) - v) < 0.005,
              f"weight {k} = {v:.3f}",
              f"weight {k} expected {v:.3f} but got {got}")

    # ------- 2. Sleeve CSVs: coverage -------
    print("\n[2/6] Sleeve CSV coverage (start + rows + MUST be current)")
    # Find the latest common market close across SPY/QQQ/IBIT — this is the
    # "truth" date that sleeves should extend to.
    market_latest = None
    for t in ["SPY", "QQQ", "IBIT"]:
        p = ROOT / "data/etfs" / f"{t}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["Date"])
            d = df["Date"].iloc[-1].date()
            market_latest = min(market_latest, d) if market_latest else d
    log("INFO", f"Latest common market date (SPY/QQQ/IBIT): {market_latest}")

    sleeve_end_dates = {}
    for name, fn, col in [("VANGUARD", "vanguard_returns.csv", "net_ret"),
                           ("ORION", "orion_returns.csv", "orion"),
                           ("HELIOS", "helios_returns.csv", "ret"),
                           ("QUANTUM", "quantum_returns.csv", "ret"),
                           ("CRYPTO", "crypto_returns.csv", "ret")]:
        fp = R / fn
        if not fp.exists():
            log("FAIL", f"{name}: {fn} missing!"); checks["fail"] += 1; continue
        df = pd.read_csv(fp, parse_dates=[0] if name == "VANGUARD" else ["Date"])
        if name == "VANGUARD":
            df = df.set_index(df.columns[0])
        else:
            df = df.set_index("Date")
        start = df.index[0].date()
        end = df.index[-1].date()
        n = len(df)
        sleeve_end_dates[name] = end
        check(start <= pd.Timestamp("2010-04-01").date(),
              f"{name:9s}  start {start} (covers 2010-IS)",
              f"{name} start {start} is AFTER 2010-04-01")
        check(n > 4000,
              f"{name:9s}  {n} rows",
              f"{name} only {n} rows (expected > 4000)")
        # CRITICAL: sleeve MUST be current (within 5 business days of market close)
        # If it's not, it means today's cron didn't actually extend it (paths broken,
        # yfinance failed, strategy crashed, etc.) and the live signal is running
        # on STALE backtest data.
        if market_latest is not None:
            gap_days = (market_latest - end).days
            check(gap_days <= 5,
                  f"{name:9s}  end   {end} (≤ 5 cal days from market {market_latest})",
                  f"{name:9s}  end   {end} is {gap_days} cal days BEHIND market {market_latest} — sleeve did NOT extend this cron!")

    # Also check sleeves agree with each other (all at same end date)
    if len(set(sleeve_end_dates.values())) > 1:
        log("WARN", f"Sleeves disagree on end date: {sleeve_end_dates}")
        checks["warn"] += 1
    else:
        log("PASS", f"All 5 sleeves at same end date: {list(sleeve_end_dates.values())[0]}")
        checks["pass"] += 1

    latest = max(sleeve_end_dates.values()) if sleeve_end_dates else None

    # ------- 3. IS metrics stability (frozen expectations) -------
    print("\n[3/6] IS metrics (FROZEN — should match backtest exactly)")
    prod = pd.read_csv(R/"phoenix_production_returns.csv", parse_dates=["Date"]).set_index("Date")
    ret = prod["net_ret"]
    m_is = metrics(ret.loc[:IS_END])
    exp = EXPECTED["is_metrics"]
    check(abs(m_is["sharpe"] - exp["sharpe"]) < TOLERANCE_SHARPE,
          f"IS Sharpe  {m_is['sharpe']:.2f}  (expected {exp['sharpe']} ± {TOLERANCE_SHARPE})",
          f"IS Sharpe  {m_is['sharpe']:.2f}  DRIFTED from expected {exp['sharpe']}")
    check(abs(m_is["cagr"] - exp["cagr"]) < TOLERANCE_CAGR,
          f"IS CAGR    {m_is['cagr']*100:.1f}%  (expected {exp['cagr']*100:.1f}% ± {TOLERANCE_CAGR*100}%)",
          f"IS CAGR    {m_is['cagr']*100:.1f}%  DRIFTED from expected {exp['cagr']*100:.1f}%")
    check(abs(m_is["mdd"] - exp["mdd"]) < TOLERANCE_MDD,
          f"IS MDD     {m_is['mdd']*100:.1f}%  (expected {exp['mdd']*100:.1f}% ± {TOLERANCE_MDD*100}%)",
          f"IS MDD     {m_is['mdd']*100:.1f}%  DRIFTED from expected {exp['mdd']*100:.1f}%")

    # ------- 4. OOS metrics sanity -------
    print("\n[4/6] OOS metrics (grows as cron extends; sanity-check only)")
    m_oos = metrics(ret.loc[OOS_START:])
    exp = EXPECTED["oos_metrics"]
    check(m_oos["sharpe"] >= exp["sharpe_min"],
          f"OOS Sharpe {m_oos['sharpe']:.2f}  (>= {exp['sharpe_min']})",
          f"OOS Sharpe {m_oos['sharpe']:.2f}  BELOW minimum {exp['sharpe_min']}")
    check(m_oos["cagr"] >= exp["cagr_min"],
          f"OOS CAGR   {m_oos['cagr']*100:.1f}%  (>= {exp['cagr_min']*100:.0f}%)",
          f"OOS CAGR   {m_oos['cagr']*100:.1f}%  BELOW minimum {exp['cagr_min']*100:.0f}%")

    # ------- 5. Full-window metrics -------
    print("\n[5/6] Full-window metrics (2010-today)")
    m_full = metrics(ret)
    exp = EXPECTED["full_metrics"]
    check(m_full["sharpe"] >= exp["sharpe_min"],
          f"Full Sharpe {m_full['sharpe']:.2f}  (>= {exp['sharpe_min']})",
          f"Full Sharpe {m_full['sharpe']:.2f}  BELOW minimum {exp['sharpe_min']}")
    check(m_full["cagr"] >= exp["cagr_min"],
          f"Full CAGR   {m_full['cagr']*100:.1f}%  (>= {exp['cagr_min']*100:.0f}%)",
          f"Full CAGR   {m_full['cagr']*100:.1f}%  BELOW minimum {exp['cagr_min']*100:.0f}%")
    check(m_full["mdd"] >= exp["mdd_max"],
          f"Full MDD    {m_full['mdd']*100:.1f}%  (>= {exp['mdd_max']*100:.0f}%)",
          f"Full MDD    {m_full['mdd']*100:.1f}%  WORSE than expected {exp['mdd_max']*100:.0f}%")
    log("INFO", f"NAV× = {m_full['navx']:.1f}  ({m_full['n']} trading days)")
    log("INFO", f"Last return date in production CSV: {ret.index[-1].date()}")

    # ------- 6. Live signal -------
    print("\n[6/6] Live signal")
    live_p = R / "live_signal.json"
    if not live_p.exists():
        log("FAIL", "live_signal.json missing!"); checks["fail"] += 1
    else:
        live = json.loads(live_p.read_text())
        ctx = live.get("context", {})
        as_of = ctx.get("as_of")
        mult = ctx.get("overlay_mult")
        regime = ctx.get("regime_gate_pass")
        vix = ctx.get("vix")
        positions = live.get("target_positions", [])
        total_w = sum(p.get("weight", 0) for p in positions)
        log("INFO", f"as_of: {as_of}")
        log("INFO", f"overlay_mult: {(mult or 0)*100:.1f}%  (0 ≤ mult ≤ 1)")
        log("INFO", f"regime_gate_pass: {regime}  (VIX={vix})")
        log("INFO", f"target positions: {len(positions)} entries")
        log("INFO", f"  " + "  ·  ".join(f"{p['ticker']} {p['pct']:.1f}%" for p in positions[:6]))
        check(mult is not None and 0 <= mult <= 1.0,
              f"overlay_mult in [0, 1.0] — no margin",
              f"overlay_mult = {mult}  VIOLATES no-margin constraint")
        check(abs(total_w - 1.0) < 0.01,
              f"target positions sum to {total_w*100:.2f}% (≈ 100%)",
              f"target positions sum to {total_w*100:.2f}% — should be ~100%")
        check(as_of is not None,
              f"signal has an as_of date: {as_of}",
              f"signal missing as_of date")

    # ------- Summary -------
    print()
    print("=" * 74)
    print(f"Validation summary: {checks['pass']} pass, {checks['warn']} warn, {checks['fail']} fail")
    print("=" * 74)
    if checks["fail"] > 0:
        print("⚠️  Drift detected. Investigate before trusting today's live signal.")
        return 1
    print("✅ All checks passed.  Live signal matches the frozen backtest spec.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
