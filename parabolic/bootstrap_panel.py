"""Materialize the PIT S&P 500 panel cache from the committed summit_panel.parquet.

A fresh clone ships `data/pit/summit_panel.parquet` (open/close/volume/member for
720 PIT tickers, 2004->present) but not the per-field `panel_*.parquet` cache that
`dca/data.build_panel()` expects (those are git-ignored, normally rebuilt from the
git-ignored `data/pit/prices/` CSVs which aren't shipped either).

This script reconstructs the per-field cache so the entire existing harness
(`dca/data.py`, `dca/fast.py`, `dca/protocol.py`) works unchanged. High/Low are
not in summit_panel and are sparse across the full universe, so we fill them with
Close (a documented limitation). The parabolic strategy is built on close/open/
volume only and never relies on High/Low, so this is harmless here; the canonical
EDA's few high/low features are out of scope for this strategy.

Run once from the repo root or anywhere:  python parabolic/bootstrap_panel.py
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIT = os.path.join(ROOT, "data", "pit")
FIELDS = ["open", "high", "low", "close", "volume", "member"]


def materialize(force: bool = False) -> dict:
    paths = {f: os.path.join(PIT, f"panel_{f}.parquet") for f in FIELDS}
    if not force and all(os.path.exists(p) for p in paths.values()):
        return {f: pd.read_parquet(p) for f, p in paths.items()}

    wide = pd.read_parquet(os.path.join(PIT, "summit_panel.parquet"))
    have = set(wide.columns.get_level_values(0))
    out = {}
    for f in ("open", "close", "volume", "member"):
        out[f] = wide[f].copy()
    # High/Low are absent from summit_panel -> fall back to Close (documented).
    out["high"] = out["close"].copy()
    out["low"] = out["close"].copy()
    out["member"] = out["member"].astype(bool)

    for f, p in paths.items():
        out[f].to_parquet(p)
    return out


if __name__ == "__main__":
    P = materialize(force=True)
    for f in FIELDS:
        v = P[f]
        print(f"{f:7s} {v.shape}  {v.index[0].date()} -> {v.index[-1].date()}")
    print("members on last day:", int(P["member"].iloc[-1].sum()))
