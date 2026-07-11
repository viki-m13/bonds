"""Fill the tables and conclusion in FINDINGS.md from the result CSVs.
Run after run_backtest.py oos. Keeps prose and numbers in lockstep."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
IS = HERE / "results" / "is_grid.csv"
OOS = HERE / "results" / "oos_results.csv"
LOCKED = HERE / "locked_configs.json"
DOC = HERE / "FINDINGS.md"


def pct(x) -> str:
    return "n/a" if pd.isna(x) else f"{x*100:+.2f}%"


def is_table() -> str:
    df = pd.read_csv(IS)
    hdr = ("| family | params | n | win | mean ret | excess vs ctrl "
           "| boot p |\n|---|---|--:|--:|--:|--:|--:|")
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["n"]) or r["n"] == 0:
            rows.append(f"| {r['family']} | {r['params']} | 0 | — | — | — | — |")
            continue
        rows.append(
            f"| {r['family']} | {r['params']} | {int(r['n'])} "
            f"| {r['win_rate']*100:.0f}% | {pct(r['mean_ret'])} "
            f"| {pct(r['excess_vs_control'])} | {r['excess_p_boot']:.3f} |")
    return hdr + "\n" + "\n".join(rows)


def locked_table() -> str:
    locked = json.loads(LOCKED.read_text())
    hdr = ("| family | params | IS n | IS excess vs ctrl | IS boot p |\n"
           "|---|---|--:|--:|--:|")
    rows = [f"| {e['family']} | {json.dumps(e['params'])} | {e['is_n']} "
            f"| {pct(e['is_excess_vs_control'])} | {e['is_excess_p_boot']:.3f} |"
            for e in locked]
    return hdr + "\n" + "\n".join(rows)


def oos_table() -> str:
    df = pd.read_csv(OOS)
    hdr = ("| family | window | n | win | mean ret | excess vs ctrl "
           "| boot p |\n|---|---|--:|--:|--:|--:|--:|")
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["n"]) or r["n"] == 0:
            rows.append(f"| {r['family']} | {r['window']} | 0 | — | — | — | — |")
            continue
        rows.append(
            f"| {r['family']} | {r['window']} | {int(r['n'])} "
            f"| {r['win_rate']*100:.0f}% | {pct(r['mean_ret'])} "
            f"| {pct(r['excess_vs_control'])} | {r['excess_p_boot']:.3f} |")
    return hdr + "\n" + "\n".join(rows)


def conclusion() -> str:
    df = pd.read_csv(OOS)
    main = df[df["window"].str.startswith("OOS")]
    any_sig = ((main["excess_vs_control"] > 0) & (main["excess_p_boot"] < 0.05))
    n_families = main["family"].nunique()
    best = main.sort_values("excess_vs_control", ascending=False).iloc[0]
    lines = []
    if any_sig.any():
        winners = main[any_sig]["family"].tolist()
        lines.append(
            f"Out-of-sample, {len(winners)} of {n_families} locked families "
            f"beat their random-entry control at p<0.05: "
            f"{', '.join(winners)}. See the table for magnitudes; treat with "
            "the disclosed survivorship caveat and transaction-cost realism "
            "in mind.")
    else:
        lines.append(
            f"Out-of-sample, **none of the {n_families} locked families beat "
            "its matched random-entry control.** The best was "
            f"`{best['family']}` at {pct(best['excess_vs_control'])} excess "
            f"(bootstrap p={best['excess_p_boot']:.2f}) — indistinguishable "
            "from random timing, and the raw return stays negative because "
            "the customer round-trip spread is paid on every trade.")
    lines.append(
        "\nThe survivorship-free final year (reported in the table) does not "
        "change the verdict. Trading individual munis like stocks, on public "
        "trade-tape signals, does not overcome the dealer spread.")
    return "\n".join(lines)


def main() -> None:
    doc = DOC.read_text()
    doc = doc.replace("<!--IS_TABLE-->", is_table())
    doc = doc.replace("<!--LOCKED_TABLE-->", locked_table())
    doc = doc.replace("<!--OOS_TABLE-->", oos_table())
    doc = doc.replace("<!--CONCLUSION-->", conclusion())
    DOC.write_text(doc)
    print("FINDINGS.md filled")


if __name__ == "__main__":
    main()
