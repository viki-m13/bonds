"""Candidate new sleeves for PHOENIX-5, rebuilt from raw prices.

All: signals lagged >= 1 day, IS <= 2018-12-31 / OOS >= 2019-01-02, honest costs.

S6 CARRY  — hedged credit-carry basket (no selection, equal-risk, monthly rebal).
S7 DECAY  — short bull+bear LETF pairs (vol-decay capture), borrow-cost sensitivity.
S8 CREDLO — low-vol floating/short-duration credit with HY-OAS macro gate.
S9 SVOL   — gated short-vol via SVXY with trend + VIX percentile filter.
"""
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"


def sr(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 60 and r.std() > 0 else np.nan


def stats(r, label):
    r = r.dropna()
    m = {}
    for w, s in [("IS", r.loc[:"2018"]), ("OOS", r.loc["2019":]), ("full", r)]:
        m[w] = sr(s)
    c = (1 + r).cumprod()
    mdd = (c / c.cummax() - 1).min()
    vol = r.std() * np.sqrt(252)
    print(f"  {label:42s} IS={m['IS']:5.2f} OOS={m['OOS']:5.2f} full={m['full']:5.2f} vol={vol*100:4.1f}% mdd={mdd*100:5.1f}%")
    return r


def px(t):
    d = pd.read_csv(ETF / f"{t}.csv", parse_dates=["Date"], index_col="Date")["Close"]
    return d[~d.index.duplicated()].sort_index()


def rets(*ts):
    return pd.concat({t: px(t).pct_change() for t in ts}, axis=1, sort=True)


def fred_series(name):
    d = pd.read_csv(FRED / f"{name}.csv", parse_dates=["Date"], index_col="Date")[name]
    return pd.to_numeric(d, errors="coerce")


# ---------------------------------------------------------------- S6 CARRY
def sleeve_carry():
    print("=" * 90)
    print("S6 CARRY — hedged credit carry, equal-risk, monthly rebal, 5bp/side")
    print("=" * 90)
    pairs = [("HYG", "IEF", 1.0), ("LQD", "IEF", 1.0), ("EMB", "IEF", 1.0),
             ("MUB", "IEI", 1.0), ("PFF", "IEF", 1.0), ("BKLN", "SHY", 1.0),
             ("CWB", "SPY", 0.5), ("VNQ", "IEF", 1.0), ("SRLN", "SHY", 1.0)]
    tickers = sorted({x for p in pairs for x in p[:2]})
    r = rets(*tickers)
    streams = {}
    for lng, hdg, beta_cap in pairs:
        cov = r[lng].rolling(252, min_periods=126).cov(r[hdg])
        var = r[hdg].rolling(252, min_periods=126).var()
        beta = (cov / var.clip(lower=1e-10)).clip(-beta_cap * 3, beta_cap * 3)
        streams[f"{lng}-{hdg}"] = (r[lng] - beta.shift(1) * r[hdg]).dropna()
    df = pd.concat(streams, axis=1, sort=True)
    for c in df.columns:
        stats(df[c], f"raw {c}")
    # equal-risk: scale each to 4% vol w/ trailing 63d, then mean across available
    scaled = pd.DataFrame(index=df.index)
    for c in df.columns:
        rv = df[c].rolling(63, min_periods=30).std() * np.sqrt(252)
        scaled[c] = df[c] * (0.04 / rv.clip(lower=0.005)).clip(0.1, 5).shift(1)
    port = scaled.mean(axis=1)
    # turnover cost approx: monthly re-true of ~9 hedged pairs, gross lev ~2-3x port
    # charge 5bp on 25% of gross monthly => ~ 5bp*0.25*12/252 daily? keep simple: 0.6%/yr drag
    port = port - 0.006 / 252
    return stats(port.dropna(), "S6 CARRY portfolio (net)")


# ---------------------------------------------------------------- S7 DECAY
def sleeve_decay(borrow_pa=0.03):
    print("=" * 90)
    print(f"S7 DECAY — short bull+bear LETF pairs, borrow {borrow_pa*100:.0f}%/yr/leg, 15bp TC")
    print("=" * 90)
    bb = [("TQQQ", "SQQQ"), ("UPRO", "SPXU"), ("FAS", "FAZ"), ("SOXL", "SOXS"),
          ("TMF", "TMV"), ("ERX", "ERY"), ("DRN", "DRV"), ("YINN", "YANG"),
          ("NUGT", "DUST"), ("LABU", "LABD")]
    streams = {}
    for bull, bear in bb:
        try:
            r = rets(bull, bear).dropna()
        except FileNotFoundError:
            continue
        if len(r) < 252:
            continue
        # short 50/50 both legs, daily reset; borrow charged per leg on notional
        gross = -(0.5 * r[bull] + 0.5 * r[bear])
        turn = 0.5 * ((r[bull] - gross).abs() + (r[bear] - gross).abs())
        net = gross - 2 * 0.5 * borrow_pa / 252 - turn * 0.0015
        streams[f"{bull}/{bear}"] = net
        stats(net, f"pair {bull}/{bear}")
    df = pd.concat(streams, axis=1, sort=True)
    scaled = pd.DataFrame(index=df.index)
    for c in df.columns:
        rv = df[c].rolling(63, min_periods=30).std() * np.sqrt(252)
        scaled[c] = df[c] * (0.04 / rv.clip(lower=0.005)).clip(0.1, 5).shift(1)
    port = scaled.mean(axis=1)
    return stats(port.dropna(), f"S7 DECAY portfolio (borrow {borrow_pa*100:.0f}%)")


# ---------------------------------------------------------------- S8 CREDLO
def sleeve_credlo():
    print("=" * 90)
    print("S8 CREDLO — low-vol credit carry + HY OAS gate (investable 2011+)")
    print("=" * 90)
    basket = {"BKLN": 0.25, "FLOT": 0.25, "MINT": 0.30, "HYG": 0.10, "GLD": 0.10}
    r = rets(*basket.keys(), "BIL")
    oas = fred_series("BAMLH0A0HYM2")
    dgs10 = fred_series("DGS10")
    w = pd.Series(basket)
    core = (r[list(basket)] @ w)
    # gate: scale to BIL when HY OAS high or rates spiking (data through t-1)
    oas_d = oas.reindex(core.index).ffill()
    gate_oas = ((8.0 - oas_d) / 3.0).clip(0, 1)        # 1 below 5.0, 0 above 8.0
    r10_chg = dgs10.reindex(core.index).ffill().diff(63)
    gate_rate = (r10_chg < 0.7).astype(float)
    gate = (gate_oas * gate_rate).shift(1).fillna(1.0)
    bil = r["BIL"].fillna(0)
    port = gate * core + (1 - gate) * bil - 0.0003 * gate.diff().abs().fillna(0)
    return stats(port.dropna(), "S8 CREDLO net")


# ---------------------------------------------------------------- S9 SVOL
def sleeve_svol():
    print("=" * 90)
    print("S9 SVOL — long SVXY gated by VIX percentile + trend (2011+), 5bp/side")
    print("=" * 90)
    sv = px("SVXY")
    r = sv.pct_change()
    vix = fred_series("VIXCLS").reindex(sv.index).ffill()
    vix_pct = vix.rolling(252, min_periods=126).rank(pct=True)
    trend = sv > sv.rolling(50).mean()
    sig = ((vix_pct < 0.6) & trend).astype(float).shift(1).fillna(0)
    ret = sig * r - sig.diff().abs().fillna(0) * 0.0005
    stats(ret, "SVXY gated raw")
    # half-size after Feb-2018 style: cap weight 0.5
    ret2 = 0.5 * ret
    return stats(ret2.dropna(), "S9 SVOL half-size")


if __name__ == "__main__":
    s6 = sleeve_carry()
    s7a = sleeve_decay(0.00)
    s7 = sleeve_decay(0.03)
    s7b = sleeve_decay(0.06)
    s8 = sleeve_credlo()
    s9 = sleeve_svol()
    # cross-correlations of candidates
    cand = pd.concat({"CARRY": s6, "DECAY": s7, "CREDLO": s8, "SVOL": s9}, axis=1, sort=True)
    print()
    print("Candidate cross-corr:")
    print(cand.corr().round(2).to_string())
    cand.to_csv(ROOT / "phoenix5/results/candidate_sleeves.csv")
