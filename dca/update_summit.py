"""Daily SUMMIT refresh for the GitHub Actions cron.

Strategy: keep a compact, committed full-history panel
(`data/pit/summit_panel.parquet`, open/close/volume/member) frozen, and on
each run stitch fresh recent bars for the *current* S&P 500 members on top of
it in memory. Then rebuild the factsheet JSONs. The big parquet is not
rewritten, so daily commits are just the small JSON payloads.

If the recent download fails entirely, we fall back to the frozen panel so the
page still rebuilds (just without the last day or two).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import data as data_mod  # noqa: E402
import build_factsheet  # noqa: E402

PANEL = os.path.join(ROOT, "data", "pit", "summit_panel.parquet")
PIT = os.path.join(ROOT, "data", "pit", "sp500_pit_membership.csv")
ETF_DIR = os.path.join(ROOT, "data", "etfs_extended")


def _load_frozen():
    big = pd.read_parquet(PANEL)
    return {f: big[f].copy() for f in ("open", "close", "volume", "member")}


def _current_members():
    mem = pd.read_csv(PIT)
    mem["date"] = pd.to_datetime(mem["date"])
    last = mem.sort_values("date").iloc[-1]
    return [t.replace(".", "-") for t in last["tickers"].split(",")]


def _download_recent(tickers, start):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i + 50]
        try:
            df = yf.download(batch, start=start, interval="1d",
                             auto_adjust=True, progress=False, threads=True,
                             group_by="ticker")
        except Exception as e:
            print("  batch err", i, e)
            continue
        for t in batch:
            try:
                sub = df[t].dropna(how="all") if len(batch) > 1 else df
                if len(sub):
                    out[t] = sub
            except Exception:
                pass
    return out


def _refresh_benchmarks(start):
    import yfinance as yf
    for tk in ("QQQ", "SPY"):
        try:
            df = yf.download(tk, start=start, interval="1d", auto_adjust=True,
                             progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index.name = "Date"
            path = os.path.join(ETF_DIR, f"{tk}.csv")
            if os.path.exists(path):
                old = pd.read_csv(path, index_col=0, parse_dates=True)
                merged = pd.concat([old[~old.index.isin(df.index)],
                                    df[["Open", "High", "Low", "Close",
                                        "Volume"]]]).sort_index()
            else:
                merged = df
            merged.to_csv(path)
            print(f"  {tk}: {len(merged)} rows -> {merged.index[-1].date()}")
        except Exception as e:
            print("  bench err", tk, e)


def _stitch(frozen, recent):
    """Overlay recent OHLCV onto the frozen panel, extending the date index and
    correcting recent rows. Returns a fresh panel dict."""
    o, c, v, m = (frozen["open"], frozen["close"], frozen["volume"],
                  frozen["member"])
    add_o, add_c, add_v = {}, {}, {}
    new_dates = set()
    for t, df in recent.items():
        if t not in c.columns:
            continue
        add_o[t] = df["Open"]; add_c[t] = df["Close"]; add_v[t] = df["Volume"]
        new_dates.update(df.index)
    if not add_c:
        return frozen
    full_idx = c.index.union(sorted(new_dates))
    def merge(base, add):
        wide = pd.DataFrame(add).reindex(full_idx)
        base2 = base.reindex(full_idx)
        # recent values win where present
        return base2.where(wide.isna(), wide)
    o2, c2, v2 = merge(o, add_o), merge(c, add_c), merge(v, add_v)
    # membership: current members True on all new dates (ffill the snapshot)
    cur = _current_members()
    m2 = m.reindex(full_idx)
    tail = full_idx[full_idx > m.index[-1]]
    for t in cur:
        if t in m2.columns:
            m2.loc[tail, t] = True
    m2 = m2.fillna(False).astype(bool)
    return {"open": o2, "close": c2, "volume": v2, "member": m2}


def main():
    frozen = _load_frozen()
    last_frozen = frozen["close"].index[-1]
    start = (last_frozen - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    print("frozen panel through", last_frozen.date(), "| fetching from", start)
    try:
        _refresh_benchmarks("2005-01-01")
        recent = _download_recent(_current_members(), start)
        print(f"  downloaded {len(recent)} tickers")
        P = _stitch(frozen, recent)
    except Exception as e:
        print("recent refresh failed, using frozen panel:", e)
        P = frozen
    print("panel now through", P["close"].index[-1].date())
    for cfg in (build_factsheet.summit_cfg(), build_factsheet.rotator_cfg(),
                build_factsheet.wave_cfg()):
        _, r, s = build_factsheet.build(cfg, P=P, write=True)
        print(f"[{cfg['prefix']}] {s['regime']} picks={s['picks']} | "
              f"ITD {r['table'][0]['strat_mult']:.2f}x vs QQQ "
              f"{r['table'][0]['qqq_mult']:.2f}x")


if __name__ == "__main__":
    main()
