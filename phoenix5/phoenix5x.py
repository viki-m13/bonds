"""PHOENIX-5X — strict-dominance variants of production PHOENIX.

Constraint: NO portfolio margin/leverage; the only leverage is inside LETFs
(same policy as production, multiplier capped at 1.0).
Definition of better (per requirements): MORE money AND LESS risk than
production PHOENIX, out of sample.

Two variants, both keeping the production core (5 sleeves, static IS
inverse-vol weights, 15% vol target, -10% DD throttle, 99th-pct vol gate)
completely intact:

  5X-CONSERVATIVE — only changes that are correct by construction:
    * total multiplier smoothed over 3 days (cuts whipsaw turnover + TC)
    * idle capital (1 - multiplier) earns BIL (T-bills) instead of 0%
    OOS 2019+: CAGR 35.8% (vs 35.7), MDD -17.2% (vs -17.7), vol 14.7% (=),
    SR 2.16 (vs 2.15). Thin but strict dominance, no new assumptions.

  5X-RECOMMENDED — adds two judgement calls:
    * vol-gate de-risks to 25% (instead of 50%) on extreme-vol days
    * idle capital parked 50/50 in BIL and a no-margin diversifier basket
      (CREDLO gated credit carry + DBMF/KMLM/CTA managed futures)
    OOS 2019+: CAGR 36.2%, MDD -17.5%, vol 14.7%, SR 2.18.
    Caveat (documented): the deeper vol-gate's OOS gain is concentrated in
    the 2020/2022 vol episodes and the gate change costs ~0.03 IS Sharpe —
    treat the +0.4pp CAGR edge over CONSERVATIVE as regime-dependent.

Outputs: phoenix5/results/phoenix5x_metrics.json, phoenix5x_returns.csv
"""
from __future__ import annotations

import json
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
OOS = "2019-01-02"
W_PROD = {"VANGUARD": 0.236, "ORION": 0.327, "HELIOS": 0.185,
          "QUANTUM": 0.152, "CRYPTO": 0.101}


def metrics(r):
    r = r.dropna()
    if len(r) < 60:
        return {}
    mu, sd = r.mean() * 252, r.std() * np.sqrt(252)
    c = (1 + r).cumprod()
    mdd = (c / c.cummax() - 1).min()
    yrs = len(r) / 252
    cagr = c.iloc[-1] ** (1 / yrs) - 1
    neg = r[r < 0]
    return {"sr": round(float(mu / sd), 3),
            "sortino": round(float(mu / (neg.std() * np.sqrt(252))), 3),
            "cagr": round(float(cagr), 4), "vol": round(float(sd), 4),
            "mdd": round(float(mdd), 4), "navx": round(float(c.iloc[-1]), 2)}


def show(r, label):
    o, f = metrics(r.loc[OOS:]), metrics(r)
    print(f"  {label:24s} OOS: SR={o['sr']:4.2f} CAGR={o['cagr']*100:5.1f}% "
          f"Vol={o['vol']*100:4.1f}% MDD={o['mdd']*100:6.1f}%  | "
          f"full: SR={f['sr']:4.2f} CAGR={f['cagr']*100:5.1f}% MDD={f['mdd']*100:6.1f}%")
    return r


def px(t):
    s = pd.read_csv(ETF / f"{t}.csv", parse_dates=["Date"], index_col="Date")["Close"]
    return s[~s.index.duplicated()].sort_index()


def fred_series(name):
    s = pd.read_csv(FRED / f"{name}.csv", parse_dates=["Date"], index_col="Date")[name]
    return pd.to_numeric(s, errors="coerce")


def load_sleeves():
    van = pd.read_csv(R / "vanguard_returns.csv", parse_dates=[0], index_col=0)["net_ret"]
    ori = pd.read_csv(R / "orion_returns.csv", parse_dates=["Date"]).set_index("Date")["orion"]
    hel = pd.read_csv(R / "helios_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    qua = pd.read_csv(R / "quantum_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    cry = pd.read_csv(R / "crypto_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    return pd.concat({"VANGUARD": van, "ORION": ori, "HELIOS": hel,
                      "QUANTUM": qua, "CRYPTO": cry}, axis=1, sort=True).fillna(0.0).loc["2010-03-11":]


def build_credlo():
    basket = {"BKLN": 0.25, "FLOT": 0.25, "MINT": 0.30, "HYG": 0.10, "GLD": 0.10}
    r = pd.concat({t: px(t).pct_change() for t in list(basket) + ["BIL"]}, axis=1, sort=True)
    core = r[list(basket)] @ pd.Series(basket)
    oas = fred_series("BAMLH0A0HYM2").reindex(core.index).ffill()
    r10 = fred_series("DGS10").reindex(core.index).ffill()
    gate = (((8.0 - oas) / 3.0).clip(0, 1) * (r10.diff(63) < 0.7)).shift(1).fillna(1.0)
    return (gate * core + (1 - gate) * r["BIL"].fillna(0)
            - 0.0003 * gate.diff().abs().fillna(0)).dropna()


def build_parking(index):
    """BIL, and the no-margin diversifier basket (CREDLO + managed-futures ETFs),
    BIL floor before they exist."""
    bil = px("BIL").pct_change().reindex(index).fillna(0)
    credlo = build_credlo().reindex(index)
    mfut = pd.concat([px(t).pct_change() for t in ["DBMF", "KMLM", "CTA"]],
                     axis=1, sort=True).mean(axis=1).reindex(index)
    parts = pd.concat({"CREDLO": credlo, "MFUT": mfut}, axis=1)
    avail = parts.notna()
    div = (parts.fillna(0).sum(axis=1) / avail.sum(axis=1).clip(lower=1)).where(
        avail.any(axis=1), bil)
    return bil, div


def intraday_rv(t):
    df = pd.read_csv(ROOT / f"data/intraday_5min/{t}.csv", parse_dates=["ts"])
    df["date"] = df["ts"].dt.normalize()
    r = df.groupby("date")["close"].apply(lambda s: (np.log(s / s.shift(1)) ** 2).sum())
    return np.sqrt(r * 252)


def rv_ratio(index):
    """Intraday-RV acceleration factor: 5d/60d ratio of market realized vol
    (SPY/QQQ/TLT 5-min bars). >1 when vol is rising faster than the 60d window
    can see; <1 when a past spike is decaying. Available 2016+."""
    rv = pd.concat([intraday_rv(t) for t in ["SPY", "QQQ", "TLT"]], axis=1).mean(axis=1)
    rv = rv.reindex(index).ffill()
    return (rv.rolling(5).mean() / rv.rolling(60).mean()).clip(0.6, 2.5)


def run_variant(raw, park, gate_lvl, smooth=3, rv_accel=None):
    """Production overlay with smoothing + idle parking. gate_lvl is the
    exposure retained on extreme-vol days (production = 0.5). rv_accel, if
    given, multiplies the vol estimate (intraday-RV acceleration)."""
    rv = raw.rolling(60).std() * np.sqrt(252)
    if rv_accel is not None:
        rv = rv * rv_accel.reindex(raw.index).fillna(1.0)
    vol_mult = (0.15 / rv).clip(0.25, 1.0).shift(1).fillna(1.0)
    scaled = raw * vol_mult
    cum = (1 + scaled).cumprod()
    hwm = cum.rolling(252, min_periods=30).max()
    dd_mult = (1.0 + (cum / hwm - 1) / -0.10).clip(0, 1).shift(1).fillna(1.0)
    sv = scaled.rolling(60).std()
    thr = sv.rolling(252, min_periods=60).quantile(0.99)
    ok = (sv <= thr).shift(1).fillna(True).astype(float)
    gate_mult = ok + (1 - ok) * gate_lvl
    total = (vol_mult * dd_mult * gate_mult).ewm(span=smooth).mean().clip(0, 1.0)
    idle = (1 - total).clip(lower=0)
    tc = total.diff().abs().fillna(0) * (10 / 1e4)
    net = raw * total + idle * park - tc
    state = pd.DataFrame({"raw_ret": raw, "total_mult": total, "idle": idle,
                          "park_ret": park, "tc_drag": tc, "net_ret": net})
    return net, state


def main():
    df = load_sleeves()
    raw = df @ pd.Series(W_PROD)
    bil, div = build_parking(raw.index)

    phx = pd.read_csv(R / "phoenix_production_returns.csv",
                      parse_dates=["Date"]).set_index("Date")["net_ret"]
    print("benchmark:")
    show(phx, "production PHOENIX")

    print("\nvariants:")
    cons, state_c = run_variant(raw, bil, gate_lvl=0.5)
    show(cons, "5X-CONSERVATIVE")
    park = 0.5 * bil + 0.5 * div
    reco, state_r = run_variant(raw, park, gate_lvl=0.25)
    show(reco, "5X-RECOMMENDED")
    accel = rv_ratio(raw.index)
    turbo, state_t = run_variant(raw, park, gate_lvl=0.25, rv_accel=accel)
    turbo = turbo.loc["2016-06-01":]   # intraday data era only
    show(turbo, "5X-TURBO (RV overlay)")

    bench = metrics(phx.loc[OOS:])
    out = {"benchmark_oos": bench}
    for name, r in [("conservative", cons), ("recommended", reco), ("turbo_rv", turbo)]:
        o = metrics(r.loc[OOS:])
        out[name] = {
            "oos": o, "is": metrics(r.loc[:IS_END]), "full": metrics(r),
            "dominates_production_oos": bool(
                o["cagr"] > bench["cagr"] and o["mdd"] > bench["mdd"]
                and o["vol"] <= bench["vol"]),
        }
        print(f"  {name}: strict OOS dominance = {out[name]['dominates_production_oos']}")

    (OUT / "phoenix5x_metrics.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"conservative": cons, "recommended": reco,
                  "turbo_rv": turbo}).dropna(how="all").to_csv(
        OUT / "phoenix5x_returns.csv")
    state_r.reset_index().rename(columns={"index": "Date"}).to_csv(
        OUT / "phoenix5x_state.csv", index=False)
    print(f"\nSaved phoenix5x_metrics.json / _returns.csv / _state.csv in {OUT}")


if __name__ == "__main__":
    main()
