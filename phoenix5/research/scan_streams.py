"""Scan every *_returns.csv in data/results: IS/OOS Sharpe + corr to PHOENIX raw blend.

Goal: find return streams that are (a) decent OOS, (b) low-corr to the current
PHOENIX 5-sleeve raw blend, as candidates for additional ensemble sleeves.
"""
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "data/results"

PHOENIX_FILES = {
    "VANGUARD": ("vanguard_returns.csv", "net_ret"),
    "ORION": ("orion_returns.csv", "orion"),
    "HELIOS": ("helios_returns.csv", "ret"),
    "QUANTUM": ("quantum_returns.csv", "ret"),
    "CRYPTO": ("crypto_returns.csv", "ret"),
}
W = {"VANGUARD": 0.236, "ORION": 0.327, "HELIOS": 0.185, "QUANTUM": 0.152, "CRYPTO": 0.101}


def sr(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 50 and r.std() > 0 else np.nan


def load_ret(path):
    """Best-effort: find date col + a daily-return column."""
    df = pd.read_csv(path)
    dcol = None
    for c in df.columns:
        if c.lower() in ("date", "ts", "index", "unnamed: 0"):
            dcol = c
            break
    if dcol is None:
        dcol = df.columns[0]
    try:
        df[dcol] = pd.to_datetime(df[dcol])
    except Exception:
        return None
    df = df.set_index(dcol).sort_index()
    pref = ["net_ret", "net", "ret", "return", "daily_ret", "strategy", "port_ret", "r"]
    cols = {c.lower(): c for c in df.columns}
    col = None
    for p in pref:
        if p in cols:
            col = cols[p]
            break
    if col is None:
        # first numeric col that looks like daily returns (|mean| < 0.01, std < 0.2)
        for c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            if v.notna().sum() > 100 and abs(v.mean()) < 0.01 and 0 < v.std() < 0.2:
                col = c
                break
    if col is None:
        return None
    r = pd.to_numeric(df[col], errors="coerce").dropna()
    if r.std() == 0 or len(r) < 300:
        return None
    return r, col


def main():
    rets = {}
    for n, (f, c) in PHOENIX_FILES.items():
        df = pd.read_csv(R / f)
        dcol = "Date" if "Date" in df.columns else df.columns[0]
        df[dcol] = pd.to_datetime(df[dcol])
        rets[n] = pd.to_numeric(df.set_index(dcol)[c], errors="coerce")
    ph = pd.concat(rets, axis=1).fillna(0)
    raw = ph @ pd.Series(W)

    rows = []
    for f in sorted(R.glob("*returns*.csv")):
        out = load_ret(f)
        if out is None:
            continue
        r, col = out
        idx = r.index.intersection(raw.index)
        if len(idx) < 300:
            continue
        c_ph = float(np.corrcoef(r.loc[idx], raw.loc[idx])[0, 1])
        rows.append({
            "file": f.name, "col": col,
            "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
            "sr_full": round(sr(r), 2),
            "sr_is": round(sr(r.loc[:"2018"]), 2),
            "sr_oos": round(sr(r.loc["2019":]), 2),
            "corr_phx": round(c_ph, 2),
        })
    t = pd.DataFrame(rows).sort_values("sr_oos", ascending=False)
    pd.set_option("display.width", 200)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
