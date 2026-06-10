"""PHOENIX-5 — meta-ensemble built on top of PHOENIX with new orthogonal sleeves.

Components (each a daily return stream):
  PHXCORE — the production 5-sleeve PHOENIX raw blend (VAN/ORI/HEL/QUA/CRY,
            static IS inverse-vol weights, as in alt/phoenix_production.py)
  MOSAIC  — 66 small carry/hedge/TSMOM streams across credit, FX, commodities,
            sectors, REITs (strategy_v10 framework minus un-investable
            LETF-short engines), trailing-Sharpe adaptive, causal
  CREDLO  — low-vol floating-rate/short-duration credit carry with HY-OAS +
            rate-spike gate (BKLN/FLOT/MINT/HYG/GLD, 2011+)
  MFUT    — managed-futures ETF basket (DBMF/KMLM/CTA, equal-weight as
            available, 2019+) — pure diversifier, no backtest of our own

Construction (all causal):
  1. Each sleeve vol-targeted to 10% ann (trailing 63d, mult capped 0.25-4).
  2. Sleeve weights: walk-forward, monthly refresh = inverse-vol base tilted
     by trailing 252d Sharpe (softmax-lite tilt, capped 3x relative).
     A sleeve enters once it has >=189d of history.
  3. Portfolio overlay (as production): 15% vol target (cap 1.0 = no
     leverage), DD throttle (-10% floor, 252d HWM), 99th-pct vol gate,
     multiplier smoothed 3d, 10bp TC per unit multiplier change.
  4. Idle capital (1 - total multiplier, when < 1) earns BIL total return.

Reports IS (2010-2018) / OOS (2019+) / full metrics, bootstrap CI and
deflated-Sharpe context, plus each sleeve's stats and correlations.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data/results"
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
OUT = ROOT / "phoenix5/results"
OUT.mkdir(parents=True, exist_ok=True)

IS_END = "2018-12-31"
OOS_START = "2019-01-02"
W_PROD = {"VANGUARD": 0.236, "ORION": 0.327, "HELIOS": 0.185,
          "QUANTUM": 0.152, "CRYPTO": 0.101}

SLEEVE_VT = 0.10        # per-sleeve vol target
PORT_VT = 0.15          # portfolio vol target
VOL_CAP = 1.0           # no portfolio leverage
MIN_HIST = 189          # days before a sleeve is eligible
WREFRESH = 21           # weight refresh cadence
TILT_CAP = 3.0          # max relative Sharpe-tilt
TC_BPS_MULT = 10.0      # TC per unit multiplier change


# ------------------------------------------------------------------ utils
def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 60:
        return {}
    mu, sd = r.mean() * 252, r.std() * np.sqrt(252)
    c = (1 + r).cumprod()
    mdd = (c / c.cummax() - 1).min()
    yrs = len(r) / 252
    neg = r[r < 0]
    return {
        "sharpe": round(float(mu / sd), 3),
        "sortino": round(float(mu / (neg.std() * np.sqrt(252))), 3) if len(neg) else None,
        "cagr": round(float(c.iloc[-1] ** (1 / yrs) - 1), 4),
        "vol": round(float(sd), 4),
        "mdd": round(float(mdd), 4),
        "calmar": round(float((c.iloc[-1] ** (1 / yrs) - 1) / abs(mdd)), 3) if mdd < 0 else None,
        "navx": round(float(c.iloc[-1]), 2),
        "n": int(len(r)),
    }


def px(t):
    s = pd.read_csv(ETF / f"{t}.csv", parse_dates=["Date"], index_col="Date")["Close"]
    return s[~s.index.duplicated()].sort_index()


def fred_series(name):
    s = pd.read_csv(FRED / f"{name}.csv", parse_dates=["Date"], index_col="Date")[name]
    return pd.to_numeric(s, errors="coerce")


# ------------------------------------------------------------------ sleeves
def build_phxcore() -> pd.Series:
    van = pd.read_csv(R / "vanguard_returns.csv", parse_dates=[0], index_col=0)["net_ret"]
    ori = pd.read_csv(R / "orion_returns.csv", parse_dates=["Date"]).set_index("Date")["orion"]
    hel = pd.read_csv(R / "helios_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    qua = pd.read_csv(R / "quantum_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    cry = pd.read_csv(R / "crypto_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    df = pd.concat({"VANGUARD": van, "ORION": ori, "HELIOS": hel,
                    "QUANTUM": qua, "CRYPTO": cry}, axis=1, sort=True).fillna(0.0)
    df = df.loc["2010-03-11":]
    return (df @ pd.Series(W_PROD)).rename("PHXCORE")


def build_mosaic() -> pd.Series:
    """Strategy_v10 stream framework minus LETF-short engines (cached)."""
    cache = OUT / "mosaic_adaptive.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=[0], index_col=0)["ret"].rename("MOSAIC")
    import importlib.util
    spec = importlib.util.spec_from_file_location("sv10", ROOT / "scripts/strategy_v10.py")
    sv10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sv10)
    prices, fred = sv10.load_all_data()
    streams = sv10.generate_all_streams(prices.pct_change(), fred)
    for k in [k for k in streams if k.startswith(("vdecay", "bbpair"))]:
        del streams[k]
    port, _ = sv10.adaptive_portfolio(streams, fred)
    port = port.dropna().rename("ret")
    port.to_csv(cache)
    return port.rename("MOSAIC")


def build_credlo() -> pd.Series:
    basket = {"BKLN": 0.25, "FLOT": 0.25, "MINT": 0.30, "HYG": 0.10, "GLD": 0.10}
    r = pd.concat({t: px(t).pct_change() for t in list(basket) + ["BIL"]}, axis=1, sort=True)
    core = r[list(basket)] @ pd.Series(basket)
    oas = fred_series("BAMLH0A0HYM2").reindex(core.index).ffill()
    r10 = fred_series("DGS10").reindex(core.index).ffill()
    gate = (((8.0 - oas) / 3.0).clip(0, 1) * (r10.diff(63) < 0.7)).shift(1).fillna(1.0)
    port = gate * core + (1 - gate) * r["BIL"].fillna(0) - 0.0003 * gate.diff().abs().fillna(0)
    return port.dropna().rename("CREDLO")


def build_mfut() -> pd.Series:
    rs = []
    for t in ["DBMF", "KMLM", "CTA"]:
        try:
            rs.append(px(t).pct_change())
        except FileNotFoundError:
            continue
    df = pd.concat(rs, axis=1, sort=True)
    return df.mean(axis=1).dropna().rename("MFUT")


def vol_target(r: pd.Series, tgt=SLEEVE_VT) -> pd.Series:
    rv = r.rolling(63, min_periods=30).std() * np.sqrt(252)
    mult = (tgt / rv.clip(lower=0.005)).clip(0.25, 4.0).shift(1)
    return r * mult


# ------------------------------------------------------------------ ensemble
def wf_sharpe_tilt_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly walk-forward weights: inverse-vol base * trailing-Sharpe tilt."""
    W = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    for i in range(MIN_HIST, len(df), WREFRESH):
        hist = df.iloc[max(0, i - 252):i]
        avail = [c for c in df.columns if hist[c].notna().sum() >= MIN_HIST]
        if not avail:
            continue
        h = hist[avail]
        iv = 1.0 / h.std().clip(lower=1e-8)
        base = iv / iv.sum()
        srs = (h.mean() / h.std().clip(lower=1e-8)) * np.sqrt(252)
        tilt = np.exp((srs - srs.mean()).clip(-2, 2) * 0.5).clip(1 / TILT_CAP, TILT_CAP)
        w = base * tilt
        w = w / w.sum()
        W.iloc[i, [df.columns.get_loc(c) for c in avail]] = w.values
    return W.ffill()


def overlay(raw: pd.Series, bil: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    rv = raw.rolling(60).std() * np.sqrt(252)
    vol_mult = (PORT_VT / rv).clip(0.25, VOL_CAP).shift(1).fillna(1.0)
    scaled = raw * vol_mult
    cum = (1 + scaled).cumprod()
    hwm = cum.rolling(252, min_periods=30).max()
    dd_mult = (1.0 + (cum / hwm - 1) / -0.10).clip(0, 1).shift(1).fillna(1.0)
    sv = scaled.rolling(60).std()
    thr = sv.rolling(252, min_periods=60).quantile(0.99)
    ok = (sv <= thr).shift(1).fillna(True).astype(float)
    gate_mult = ok + (1 - ok) * 0.5
    total = (vol_mult * dd_mult * gate_mult).ewm(span=3).mean().clip(0, VOL_CAP)
    idle = (1.0 - total).clip(lower=0.0)
    tc = total.diff().abs().fillna(0) * (TC_BPS_MULT / 1e4)
    net = raw * total + idle * bil.reindex(raw.index).fillna(0) - tc
    state = pd.DataFrame({"raw_ret": raw, "total_mult": total, "idle": idle,
                          "tc_drag": tc, "net_ret": net})
    return net, state


def block_bootstrap_sr_ci(r: pd.Series, n_boot=2000, block=21, seed=7):
    rng = np.random.default_rng(seed)
    x = r.dropna().values
    n = len(x)
    srs = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n - block)
            idx.extend(range(s, s + block))
        b = x[idx[:n]]
        srs.append(b.mean() / b.std() * np.sqrt(252))
    return float(np.percentile(srs, 2.5)), float(np.percentile(srs, 97.5))


def main():
    print("Building sleeves...")
    sleeves = {}
    for f in (build_phxcore, build_mosaic, build_credlo, build_mfut):
        s = f()
        sleeves[s.name] = s
        print(f"  {s.name:8s} {s.index[0].date()} -> {s.index[-1].date()}  "
              f"IS={metrics(s.loc[:IS_END]).get('sharpe')}  "
              f"OOS={metrics(s.loc[OOS_START:]).get('sharpe')}")
    df = pd.concat(sleeves, axis=1, sort=True).loc["2010-03-11":]

    print("\nSleeve correlations (full sample):")
    print(df.corr().round(2).to_string())

    vt = df.apply(vol_target)
    W = wf_sharpe_tilt_weights(vt)
    raw = (vt.fillna(0) * W).sum(axis=1)
    raw = raw[W.notna().any(axis=1)].dropna()

    bil = px("BIL").pct_change()
    net, state = overlay(raw, bil)

    m_full, m_is, m_oos = metrics(net), metrics(net.loc[:IS_END]), metrics(net.loc[OOS_START:])
    print("\n=== PHOENIX-5 ===")
    for nm, m in [("FULL", m_full), ("IS", m_is), ("OOS", m_oos)]:
        print(f"  {nm:5s} SR={m['sharpe']:5.2f}  Sortino={m['sortino']:5.2f}  "
              f"CAGR={m['cagr']*100:5.1f}%  Vol={m['vol']*100:4.1f}%  "
              f"MDD={m['mdd']*100:5.1f}%  NAVx={m['navx']}")
    lo, hi = block_bootstrap_sr_ci(net.loc[OOS_START:])
    print(f"  OOS Sharpe 95% CI (block bootstrap): [{lo:.2f}, {hi:.2f}]")

    # benchmark: production PHOENIX
    phx = pd.read_csv(R / "phoenix_production_returns.csv",
                      parse_dates=["Date"]).set_index("Date")["net_ret"]
    print(f"\n  production PHOENIX OOS SR = {metrics(phx.loc[OOS_START:])['sharpe']:.2f} (benchmark)")

    out = {
        "sleeves": {k: {"is": metrics(v.loc[:IS_END]), "oos": metrics(v.loc[OOS_START:]),
                        "full": metrics(v)} for k, v in sleeves.items()},
        "correlations": {a: {b: round(float(df.corr().loc[a, b]), 3) for b in df.columns}
                         for a in df.columns},
        "phoenix5": {"full": m_full, "is": m_is, "oos": m_oos,
                     "oos_sr_ci95": [round(lo, 3), round(hi, 3)],
                     "avg_mult": round(float(state["total_mult"].mean()), 3)},
        "benchmark_phoenix_oos": metrics(phx.loc[OOS_START:]),
        "params": {"sleeve_vt": SLEEVE_VT, "port_vt": PORT_VT, "vol_cap": VOL_CAP,
                   "min_hist": MIN_HIST, "tilt_cap": TILT_CAP,
                   "weight_refresh_days": WREFRESH},
    }
    (OUT / "phoenix5_metrics.json").write_text(json.dumps(out, indent=2))
    state.reset_index().rename(columns={"index": "Date"}).to_csv(
        OUT / "phoenix5_returns.csv", index=False)
    W.dropna(how="all").to_csv(OUT / "phoenix5_weights.csv")
    print(f"\nSaved phoenix5_metrics.json / _returns.csv / _weights.csv in {OUT}")


if __name__ == "__main__":
    main()
