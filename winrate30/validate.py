"""Walk-forward validation of the 30-day positive-return rule ensemble.

For each test year Y (2016..present):
  1. Rules are searched and selected using ONLY data ending 21 trading days
     before Jan 1 of Y (so no forward window leaks into training).
  2. The selected ensemble (union of rules, deduplicated per stock) is applied
     to year Y and every signal's actual 21-trading-day forward return is
     recorded.

Nothing from a test year ever influences the rules used in that year, so the
pooled hit rate is a genuine out-of-sample estimate.

Outputs:
  reports/validation_report.md   human-readable report
  reports/oos_signals.csv        every out-of-sample signal with its outcome
  selected_rules.json            final ensemble (trained on all data) used by
                                 recommend.py
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from config import (FIRST_TEST_YEAR, HORIZON, N_SELECT, REPORTS_DIR,
                    RULES_FILE)
from data import load_prices
from features import compute_panel, describe_rule, Panel
from rules import RuleStats, select_rules, union_mask, wilson_lb


def row_region(panel: Panel, start_pos: int, end_pos: int) -> np.ndarray:
    region = np.zeros(panel.close.shape, dtype=bool)
    region[start_pos:end_pos, :] = True
    return region


def signals_dataframe(panel: Panel, mask: np.ndarray) -> pd.DataFrame:
    rows, cols = np.nonzero(mask)
    df = pd.DataFrame({
        "date": panel.index[rows],
        "ticker": [panel.tickers[c] for c in cols],
        "fwd_ret": panel.fwd[rows, cols],
    })
    return df.dropna(subset=["fwd_ret"]).reset_index(drop=True)


def run_validation() -> None:
    prices = load_prices()
    panel = compute_panel(prices)
    idx = panel.index
    last_year = idx[-1].year

    folds = []
    all_signals = []
    for year in range(FIRST_TEST_YEAR, last_year + 1):
        cutoff = idx.searchsorted(pd.Timestamp(year, 1, 1))
        year_end = idx.searchsorted(pd.Timestamp(year + 1, 1, 1))
        train = row_region(panel, 0, max(cutoff - HORIZON, 0))
        test = row_region(panel, cutoff, year_end)

        print(f"\n=== Fold {year}: training on {idx[0].date()} .. "
              f"{idx[max(cutoff - HORIZON - 1, 0)].date()} ===")
        chosen = select_rules(panel, train, verbose=True)
        rules = [s.rule for s in chosen]

        u = union_mask(panel, rules, test)
        sig = signals_dataframe(panel, u)
        n, h = len(sig), int((sig["fwd_ret"] > 0).sum())

        base_mask = panel.valid & test
        bn = int(np.count_nonzero(base_mask))
        bh = int(np.count_nonzero(base_mask & (panel.fwd > 0)))

        folds.append({
            "year": year, "rules": rules, "n": n, "hits": h,
            "rate": h / n if n else float("nan"),
            "avg_ret": float(sig["fwd_ret"].mean()) if n else float("nan"),
            "min_ret": float(sig["fwd_ret"].min()) if n else float("nan"),
            "baseline_rate": bh / bn if bn else float("nan"),
        })
        all_signals.append(sig.assign(fold=year))
        print(f"  OOS {year}: signals={n} hit_rate={h / n if n else float('nan'):.3f} "
              f"baseline={bh / bn:.3f}")

    oos = pd.concat(all_signals, ignore_index=True)
    n, h = len(oos), int((oos["fwd_ret"] > 0).sum())
    lb = wilson_lb(h, n)

    # Monthly clustering analysis (signals are cross-sectionally correlated)
    oos["month"] = oos["date"].dt.to_period("M")
    monthly = oos.groupby("month").agg(
        n=("fwd_ret", "size"), hit_rate=("fwd_ret", lambda s: (s > 0).mean()),
        avg_ret=("fwd_ret", "mean"),
    )
    n_months_total = oos["month"].nunique()
    span_months = pd.period_range(oos["date"].min(), oos["date"].max(), freq="M")
    worst_months = monthly.sort_values("hit_rate").head(10)

    # Final selection on ALL available history -> production rules
    full_train = row_region(panel, 0, len(idx))
    print("\n=== Final selection on full history ===")
    final = select_rules(panel, full_train, verbose=True)

    _write_outputs(panel, folds, oos, monthly, span_months, n, h, lb,
                   worst_months, final)


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%" if np.isfinite(x) else "n/a"


def _write_outputs(panel, folds, oos, monthly, span_months, n, h, lb,
                   worst_months, final: list[RuleStats]) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    oos.to_csv(REPORTS_DIR / "oos_signals.csv", index=False)

    lines = []
    a = lines.append
    a("# winrate30 — Walk-Forward Validation Report")
    a("")
    a(f"Generated: {datetime.now():%Y-%m-%d}  |  Data: "
      f"{panel.index[0].date()} .. {panel.index[-1].date()}  |  "
      f"Universe: {len(panel.tickers)} stocks  |  Horizon: {HORIZON} "
      f"trading days (~30 calendar days)")
    a("")
    a("## Headline out-of-sample result")
    a("")
    a(f"- **Signals (non-overlapping, fully out-of-sample): {n}**")
    a(f"- **Positive after {HORIZON} trading days: {h}  ->  hit rate "
      f"{_fmt_pct(h / n)}**")
    a(f"- **95% Wilson lower bound: {_fmt_pct(lb)}**")
    a(f"- Average forward return per signal: {_fmt_pct(oos['fwd_ret'].mean())}")
    a(f"- Median: {_fmt_pct(oos['fwd_ret'].median())}  |  5th percentile: "
      f"{_fmt_pct(oos['fwd_ret'].quantile(0.05))}  |  worst: "
      f"{_fmt_pct(oos['fwd_ret'].min())}")
    a(f"- Signal frequency: {n / max(len(span_months), 1):.1f}/month on average; "
      f"signals occurred in {monthly.shape[0]} of {len(span_months)} months")
    a("")
    bm_n = monthly.shape[0]
    bm_h = int((monthly["avg_ret"] > 0).sum())
    a("## Basket-level result (recommended way to use the tool)")
    a("")
    a("Buying every signal of a calendar month as one equal-weight basket "
      "and holding each position ~30 days diversifies away single-stock "
      "misses:")
    a("")
    a(f"- **{bm_h} of {bm_n} signal-months had a positive basket return = "
      f"{_fmt_pct(bm_h / bm_n)}** (Wilson 95% lower bound "
      f"{_fmt_pct(wilson_lb(bm_h, bm_n))})")
    a(f"- Average basket month return: {_fmt_pct(monthly['avg_ret'].mean())}; "
      f"worst basket month: {_fmt_pct(monthly['avg_ret'].min())}")
    a("")
    a("## Per-year walk-forward results")
    a("")
    a("Rules are re-selected each year using only prior data, then tested on "
      "the following year.")
    a("")
    a("| Test year | Signals | Hit rate | Avg 30d ret | Worst signal | "
      "Baseline (any stock) |")
    a("|---|---|---|---|---|---|")
    for f in folds:
        a(f"| {f['year']} | {f['n']} | {_fmt_pct(f['rate'])} | "
          f"{_fmt_pct(f['avg_ret'])} | {_fmt_pct(f['min_ret'])} | "
          f"{_fmt_pct(f['baseline_rate'])} |")
    a("")
    a("## Worst months (cross-sectional risk)")
    a("")
    a("Signals cluster in time and stocks move together, so the binomial "
      "confidence interval understates tail risk. The worst signal-months:")
    a("")
    a("| Month | Signals | Hit rate | Avg ret |")
    a("|---|---|---|---|")
    for m, row in worst_months.iterrows():
        a(f"| {m} | {int(row['n'])} | {_fmt_pct(row['hit_rate'])} | "
          f"{_fmt_pct(row['avg_ret'])} |")
    a("")
    a("## Production rules (selected on full history)")
    a("")
    for s in final:
        a(f"- **{' + '.join(s.rule)}** — n={s.n} non-overlapping signals, "
          f"hit rate {_fmt_pct(s.rate)}, Wilson LB {_fmt_pct(s.lb)}")
        a(f"  - i.e. buy when: {describe_rule(s.rule)}")
    a("")
    a("## Methodology & caveats")
    a("")
    a(f"- A 'hit' = adjusted close is higher {HORIZON} trading days after "
      "the signal.")
    a("- Signals are deduplicated per stock (a stock cannot re-signal within "
      f"{HORIZON} trading days), so outcomes do not double count "
      "overlapping windows.")
    a("- Rule selection per fold never sees the test year (training data even "
      f"ends {HORIZON} trading days before it, so no forward window leaks).")
    a("- **Survivorship bias**: the universe is today's large caps; failed "
      "companies are absent. Large-cap restriction limits but does not "
      "eliminate this; true forward-looking hit rates are likely somewhat "
      "lower than backtested ones.")
    a("- **Regime risk**: most of the validation window is a structural bull "
      "market. In a 2008-style year the realized hit rate would be far below "
      "the average — see the worst-months table.")
    a("- Returns ignore transaction costs, slippage and taxes (small at a "
      "30-day horizon for liquid large caps, but not zero).")
    a("- This is research tooling, not investment advice.")
    report = "\n".join(lines)
    (REPORTS_DIR / "validation_report.md").write_text(report)
    print(f"\nReport written to {REPORTS_DIR / 'validation_report.md'}")
    print(f"\nPOOLED OOS: n={n} hits={h} rate={h / n:.4f} wilson_lb={lb:.4f}")

    RULES_FILE.write_text(json.dumps({
        "generated": datetime.now().strftime("%Y-%m-%d"),
        "horizon_trading_days": HORIZON,
        "rules": [s.rule for s in final],
        "rule_train_stats": [
            {"rule": s.rule, "n": s.n, "hits": s.hits, "rate": s.rate,
             "wilson_lb": s.lb, "description": describe_rule(s.rule)}
            for s in final
        ],
        "oos_pooled": {"n": n, "hits": h, "rate": h / n, "wilson_lb": lb},
    }, indent=2))
    print(f"Production rules written to {RULES_FILE}")


if __name__ == "__main__":
    run_validation()
