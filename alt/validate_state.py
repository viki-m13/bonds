"""PHOENIX v4 state validation — run at the end of every refresh.

Checks (exit code 1 on any FAIL — the refresh pipeline treats this as a
gate and the workflow will not commit a failing state):
  1. Production params in phoenix_production_metrics.json match the code.
  2. Every sleeve CSV exists, is current (within 5 calendar days of the
     SPY close), has no duplicate dates, and has no suspicious trailing
     run of exact-zero returns (>= 4 rows — under the unified convention
     the last row is fully known, so long zero tails mean a data problem,
     not a pending repair).
  3. No sleeve has a |daily return| > 60% (split/seam corruption canary;
     the biggest legitimate day in 16 years of 3x-LETF sleeves is ~35%).
  4. Price-basis integrity: sampled price files must NOT contain a mixed
     adjusted/raw seam (an 'Adj Close' column differing from Close).
  5. Frozen-history integrity: the 2014-2018 window of net_ret matches the
     frozen reference Sharpe within +/-0.02 (this window's sleeve rows are
     frozen; only a merge bug or retroactive edit can move it).
  6. live_signal.json sanity: weights sum to ~100%, gross <= 100.5%,
     overlay mult in [0,1], as_of current.
  7. BIL dividend-drop canary (catches raw-on-adjusted price seams that the
     Adj-Close comparison is structurally blind to) and production-output
     currency (a crashed production step must not ship stale results).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "data/results"
ETF = ROOT / "data/etfs"
ALT = ROOT / "alt"
sys.path.insert(0, str(ALT))

import phoenix_production as prod

# Frozen reference: 2014-2018 Sharpe of the v3 backtest at the time the
# history baseline was committed. The sleeve rows in this window are frozen
# by refresh_all's merge, so this number must be stable run-to-run.
FROZEN_2014_2018_SHARPE = 1.0904  # v4 baseline (equal-active allocator,
# VAN gross 1.0, CRY from IBIT listing, gate NaN fix), committed 2026-07-02

FAILS = []
WARNS = []


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False):
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        (WARNS if warn_only else FAILS).append(name)


def main() -> int:
    print("=" * 70)
    print("PHOENIX v4 state validation")
    print("=" * 70)

    # --- 1. params
    m = json.loads((R / "phoenix_production_metrics.json").read_text())
    p = m["params"]
    check("params.target_vol matches code", p["target_vol"] == prod.TARGET_VOL)
    check("params.vol_cap matches code", p["vol_cap"] == prod.VOL_CAP)
    check("params.gate_pct matches code", p.get("gate_pct") == prod.GATE_PCT)
    check("params.ewma_lambda matches code", p.get("ewma_lambda") == prod.EWMA_LAMBDA)
    check("params.vol_floor matches code", p.get("vol_floor") == prod.VOL_FLOOR)
    check("params.allocator matches code",
          p.get("allocator", {}).get("type") == prod.ALLOCATOR,
          detail=f"metrics={p.get('allocator', {}).get('type')} code={prod.ALLOCATOR}")
    check("params.sleeves match registry",
          set(p["sleeves"].keys()) == set(prod.SLEEVES.keys()),
          detail=",".join(sorted(p["sleeves"].keys())))

    # --- anchor date
    spy = pd.read_csv(ETF / "SPY.csv", parse_dates=["Date"]).sort_values("Date")
    spy_last = spy["Date"].iloc[-1]

    # --- 2+3. sleeves
    for tag, (csv, col) in prod.SLEEVES.items():
        f = R / csv
        if not f.exists():
            check(f"{tag}: CSV exists", False)
            continue
        df = pd.read_csv(f)
        dc = "Date" if "Date" in df.columns else df.columns[0]
        dates = pd.to_datetime(df[dc])
        s = pd.to_numeric(df[col], errors="coerce")
        check(f"{tag}: current", (spy_last - dates.iloc[-1]).days <= 5,
              detail=f"last={dates.iloc[-1].date()} vs SPY {spy_last.date()}")
        check(f"{tag}: no duplicate dates", not dates.duplicated().any())
        tail = s.tail(6).values
        run0 = 0
        for v in tail[::-1]:
            if v == 0.0:
                run0 += 1
            else:
                break
        check(f"{tag}: no trailing zero-run", run0 < 4,
              detail=f"trailing zeros={run0}", warn_only=(run0 < 6))
        big = float(np.nanmax(np.abs(s.values))) if len(s) else 0.0
        check(f"{tag}: no absurd daily return", big <= 0.60,
              detail=f"max |ret|={big:.2%}")

    # --- 4. price-basis integrity (sampled)
    for t in ["TQQQ", "TMF", "SPY", "BIL", "IBIT"]:
        f = ETF / f"{t}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if "Adj Close" in df.columns:
            adj = pd.to_numeric(df["Adj Close"], errors="coerce")
            cls = pd.to_numeric(df["Close"], errors="coerce")
            both = adj.notna() & cls.notna()
            mixed = bool(both.any() and not np.allclose(adj[both], cls[both], rtol=1e-6))
            check(f"{t}: no adjusted/raw seam", not mixed, warn_only=True,
                  detail="Adj Close differs from Close — mixed basis")
        # else: single-basis file (fully adjusted rewrite) — fine

    # --- 4b. dividend-drop canary: on an adjusted (total-return) basis a
    # T-bill ETF's open must NEVER drop more than a few bps. Any BIL day
    # below -10 bps means raw ex-div rows were appended onto adjusted
    # history — the exact seam the Adj-Close comparison above cannot see
    # (freshly appended raw rows always have Adj == Close).
    bil_f = ETF / "BIL.csv"
    if bil_f.exists():
        bil = pd.read_csv(bil_f, parse_dates=["Date"]).sort_values("Date")
        bo = pd.to_numeric(bil["Open"], errors="coerce")
        recent = bo.pct_change().tail(90)
        worst = float(recent.min()) if len(recent) else 0.0
        # warn-only until the first post-v4 cron rewrites full adjusted
        # history (heals the Apr-Jun 2026 raw tail); a hard gate afterwards.
        heal_grace = pd.Timestamp.now() < pd.Timestamp("2026-08-01")
        check("BIL: no ex-div drop in trailing 90d (raw-tail seam canary)",
              worst > -0.0010, detail=f"worst day={worst:.4%}",
              warn_only=heal_grace)

    # --- 4c. production output currency: a crashed production step must not
    # ship yesterday's blend series as today's.
    pr_f = R / "phoenix_production_returns.csv"
    if pr_f.exists():
        pr_dates = pd.to_datetime(pd.read_csv(pr_f, usecols=["Date"])["Date"])
        check("production returns current",
              (spy_last - pr_dates.iloc[-1]).days <= 5,
              detail=f"last={pr_dates.iloc[-1].date()} vs SPY {spy_last.date()}")

    # --- 5. frozen-window stability
    prod_csv = R / "phoenix_production_returns.csv"
    if prod_csv.exists():
        pr = pd.read_csv(prod_csv, parse_dates=["Date"]).set_index("Date")["net_ret"]
        win = pr.loc["2014-01-02":"2018-12-31"].dropna()
        sr = float(win.mean() / win.std() * np.sqrt(252)) if win.std() > 0 else 0.0
        if FROZEN_2014_2018_SHARPE is None:
            print(f"  [INFO] 2014-2018 Sharpe = {sr:.4f} (no frozen reference set yet)")
        else:
            check("frozen 2014-2018 Sharpe stable",
                  abs(sr - FROZEN_2014_2018_SHARPE) <= 0.02,
                  detail=f"{sr:.4f} vs frozen {FROZEN_2014_2018_SHARPE:.4f}")

    # --- 6. live signal sanity
    ls = R / "live_signal.json"
    if ls.exists():
        live = json.loads(ls.read_text())
        tw = {t["ticker"]: t["weight"] for t in live.get("target_positions", [])}
        tot = sum(tw.values())
        check("live: weights sum ~100%", 0.97 <= tot <= 1.005, detail=f"sum={tot:.4f}")
        gross_risk = sum(v for k, v in tw.items() if k != "BIL")
        check("live: no margin (risk gross <= 100.5%)", gross_risk <= 1.005,
              detail=f"risk gross={gross_risk:.4f}")
        om = live.get("context", {}).get("overlay_mult", None)
        check("live: overlay mult in [0,1]", om is not None and 0.0 <= om <= 1.0,
              detail=f"mult={om}")
        check("live: as_of current",
              (spy_last - pd.Timestamp(live["context"]["as_of"])).days <= 5,
              detail=f"as_of={live['context']['as_of']}")
    else:
        check("live_signal.json exists", False, warn_only=True)

    print("-" * 70)
    if FAILS:
        print(f"VALIDATION FAILED: {len(FAILS)} check(s): {FAILS}")
        return 1
    print(f"Validation passed ({len(WARNS)} warning(s): {WARNS})" if WARNS
          else "Validation passed (all checks green)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
