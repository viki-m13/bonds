"""APEX — Phase 1: Build extended LETF universe with synthetic pre-inception history.

Extends the existing synthetic LETF builder with a broader universe and longer
history (back to 1999 where underlying allows) for pre-2008 crisis testing.

Universe: 19 LETFs across 6 asset classes (equity index, sector, international,
fixed income, commodity, REIT). All prices are Open + Close. Pre-inception
history is built from the underlying index using the standard daily-reset formula:

    r_letf[t] = L * r_under[t] - (L-1) * rf[t] / 252 - expense / 252

where rf is FEDFUNDS (daily decimal) and expense = 90 bps/yr.

Calibration: on the overlap window with real LETF data (first 3y), we compute
correlation of synthetic vs real daily returns. Typical corr > 0.98.

Output:
    data/apex/prices.parquet   — wide Open/Close for the universe
    data/apex/metadata.json    — per-ticker info (L, under, start, corr)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
OUT = ROOT / "data/apex"
OUT.mkdir(parents=True, exist_ok=True)

EXPENSE = 0.0090
DPY = 252
START = pd.Timestamp("1999-03-10")   # QQQ inception

# (ticker, leverage, underlying) — universe spans 6 asset classes
UNIVERSE = [
    # --- US equity index / large cap ---
    ("UPRO",  3.0, "SPY"),
    ("SSO",   2.0, "SPY"),
    ("TQQQ",  3.0, "QQQ"),
    ("QLD",   2.0, "QQQ"),
    # --- US sector ---
    ("TECL",  3.0, "XLK"),
    ("FAS",   3.0, "XLF"),
    ("ERX",   2.0, "XLE"),
    ("SOXL",  3.0, "SMH"),
    # --- REIT ---
    ("DRN",   3.0, "VNQ"),
    # --- International equity ---
    ("EDC",   3.0, "EEM"),
    ("YINN",  3.0, "FXI"),
    # --- Treasuries ---
    ("TMF",   3.0, "TLT"),
    ("UBT",   2.0, "TLT"),
    ("TYD",   3.0, "IEF"),
    # --- Commodities / metals ---
    ("UGL",   2.0, "GLD"),
    ("UCO",   2.0, "USO"),
    # --- Unleveraged benchmarks (for signal + cash replacement) ---
    ("SPY",   1.0, None),
    ("QQQ",   1.0, None),
    ("TLT",   1.0, None),
    ("GLD",   1.0, None),
    ("BIL",   1.0, None),   # cash
    ("SHY",   1.0, None),   # cash-ish
]


def load(ticker: str) -> pd.DataFrame | None:
    fp = ETF / f"{ticker}.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp, parse_dates=["Date"]).sort_values("Date").drop_duplicates("Date").set_index("Date")
    return df[["Open", "Close"]].astype(float)


def fedfunds(idx: pd.DatetimeIndex) -> pd.Series:
    fp = FRED / "FEDFUNDS.csv"
    df = pd.read_csv(fp, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    return (df["FEDFUNDS"].astype(float) / 100).reindex(idx).ffill().bfill().fillna(0.02)


def synth_letf(L: float, under: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    rc = under["Close"].pct_change()
    ro = under["Open"].pct_change()
    dr = rf / DPY
    de = EXPENSE / DPY
    rlc = L * rc - (L - 1.0) * dr - de
    rlo = L * ro - (L - 1.0) * dr - de
    rlc = rlc.fillna(0.0)
    rlo = rlo.fillna(0.0)
    sc = (1.0 + rlc).cumprod() * 100.0
    so = (1.0 + rlo).cumprod() * 100.0
    return pd.DataFrame({"Open": so, "Close": sc}, index=under.index)


def main():
    meta = []
    all_open = {}
    all_close = {}

    for (tic, L, under) in UNIVERSE:
        real = load(tic)
        info = {"ticker": tic, "L": L, "under": under}
        if real is None:
            info["error"] = "missing"
            meta.append(info)
            print(f"{tic:5s}  MISSING")
            continue
        if under is None:
            # Plain ETF
            merged = real
            info["real_start"] = str(real.index.min().date())
            info["synth_start"] = None
        else:
            udf = load(under)
            if udf is None:
                merged = real
                info["error"] = f"underlying {under} missing"
            else:
                rf = fedfunds(udf.index)
                synth = synth_letf(L, udf, rf)
                # Scale synth so its Close on the day before real inception == Open on inception day
                real_start = real.index.min()
                synth_pre = synth[synth.index < real_start]
                if len(synth_pre) > 0:
                    scale = float(real.loc[real_start, "Open"]) / float(synth_pre["Close"].iloc[-1])
                    synth_pre_scaled = synth_pre * scale
                    merged = pd.concat([synth_pre_scaled, real])
                else:
                    merged = real
                # calibration
                ov = udf.index.intersection(real.index)
                if len(ov) >= 60:
                    end3y = ov.min() + pd.Timedelta(days=3 * 365)
                    ov = ov[ov <= end3y]
                    s_ret = synth["Close"].reindex(ov).pct_change()
                    r_ret = real["Close"].reindex(ov).pct_change()
                    info["corr_close_overlap3y"] = round(float(s_ret.corr(r_ret)), 4)
                    info["overlap_days"] = len(ov)
                info["real_start"] = str(real.index.min().date())
                info["synth_start"] = str(synth.index.min().date())
                info["synth_days_pre"] = int(len(synth_pre))

        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        all_open[tic] = merged["Open"]
        all_close[tic] = merged["Close"]
        info["start"] = str(merged.index.min().date())
        info["end"] = str(merged.index.max().date())
        info["n"] = int(len(merged))
        meta.append(info)
        cc = info.get("corr_close_overlap3y", "-")
        print(f"{tic:5s} L={L} under={str(under):5s} start={info['start']} n={info['n']:>5d} corr={cc}")

    po = pd.DataFrame(all_open).sort_index()
    pc = pd.DataFrame(all_close).sort_index()
    po = po[po.index >= START]
    pc = pc[pc.index >= START]

    # Save as parquet (wide format with multi-index column)
    prices = pd.concat({"Open": po, "Close": pc}, axis=1)
    prices.to_parquet(OUT / "prices.parquet")
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nSaved prices: {prices.shape} → {OUT/'prices.parquet'}")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")


if __name__ == "__main__":
    main()
