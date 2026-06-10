"""Ensemble-machinery experiments for PHOENIX-5.

Baseline = production PHOENIX (static IS inv-vol weights + vol target + DD throttle
+ vol gate). Experiments change ONE thing at a time:

  A  baseline replication
  B  + per-sleeve vol targeting (each sleeve scaled to 15% ann vol, trailing 63d)
  C  walk-forward inverse-vol weights (rolling 252d, monthly refresh) instead of static
  D  walk-forward ERC (equal risk contribution w/ shrunk corr) weights
  E  overlay ablation: no DD throttle / no vol gate / neither
  F  + CREDLO sleeve (low-vol credit carry, 2011+)
  G  best combo

Metrics reported IS (2010-2018) / OOS (2019+). All estimates use trailing data only.
"""
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "data/results"
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"

IS_END = "2018-12-31"
W_PROD = {"VANGUARD": 0.236, "ORION": 0.327, "HELIOS": 0.185, "QUANTUM": 0.152, "CRYPTO": 0.101}


def metr(r):
    r = r.dropna()
    mu, sd = r.mean() * 252, r.std() * np.sqrt(252)
    c = (1 + r).cumprod()
    mdd = (c / c.cummax() - 1).min()
    yrs = len(r) / 252
    cagr = c.iloc[-1] ** (1 / yrs) - 1
    return {"sr": mu / sd, "cagr": cagr, "vol": sd, "mdd": mdd}


def show(r, label):
    f, i, o = metr(r), metr(r.loc[:IS_END]), metr(r.loc["2019":])
    print(f"  {label:44s} IS={i['sr']:5.2f} OOS={o['sr']:5.2f} full={f['sr']:5.2f} "
          f"cagr={f['cagr']*100:5.1f}% mdd={f['mdd']*100:5.1f}%")
    return r


def load_sleeves(with_credlo=True):
    van = pd.read_csv(R / "vanguard_returns.csv", parse_dates=[0], index_col=0)["net_ret"]
    ori = pd.read_csv(R / "orion_returns.csv", parse_dates=["Date"]).set_index("Date")["orion"]
    hel = pd.read_csv(R / "helios_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    qua = pd.read_csv(R / "quantum_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    cry = pd.read_csv(R / "crypto_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    d = {"VANGUARD": van, "ORION": ori, "HELIOS": hel, "QUANTUM": qua, "CRYPTO": cry}
    if with_credlo:
        d["CREDLO"] = build_credlo()
    df = pd.concat(d, axis=1, sort=True)
    return df.loc["2010-03-11":]


def build_credlo():
    basket = {"BKLN": 0.25, "FLOT": 0.25, "MINT": 0.30, "HYG": 0.10, "GLD": 0.10}
    px = {}
    for t in list(basket) + ["BIL"]:
        s = pd.read_csv(ETF / f"{t}.csv", parse_dates=["Date"], index_col="Date")["Close"]
        px[t] = s[~s.index.duplicated()].sort_index()
    r = pd.concat({t: s.pct_change() for t, s in px.items()}, axis=1, sort=True)
    oas = pd.read_csv(FRED / "BAMLH0A0HYM2.csv", parse_dates=["Date"], index_col="Date").iloc[:, 0]
    dgs10 = pd.read_csv(FRED / "DGS10.csv", parse_dates=["Date"], index_col="Date").iloc[:, 0]
    core = r[list(basket)] @ pd.Series(basket)
    oas_d = oas.reindex(core.index).ffill()
    gate = (((8.0 - oas_d) / 3.0).clip(0, 1)
            * (dgs10.reindex(core.index).ffill().diff(63) < 0.7).astype(float)).shift(1).fillna(1.0)
    port = gate * core + (1 - gate) * r["BIL"].fillna(0) - 0.0003 * gate.diff().abs().fillna(0)
    return port.dropna()


def vol_target_sleeve(r, tgt=0.15, win=63, cap=2.0, floor=0.25):
    rv = r.rolling(win, min_periods=30).std() * np.sqrt(252)
    mult = (tgt / rv.clip(lower=0.01)).clip(floor, cap).shift(1).fillna(1.0)
    return r * mult


def overlay(raw, dd_on=True, gate_on=True, tgt=0.15, smooth=1):
    rv = raw.rolling(60).std() * np.sqrt(252)
    vol_mult = (tgt / rv).clip(0.25, 1.0).shift(1).fillna(1.0)
    scaled = raw * vol_mult
    if dd_on:
        cum = (1 + scaled).cumprod()
        hwm = cum.rolling(252, min_periods=30).max()
        dd_mult = (1.0 + (cum / hwm - 1) / -0.10).clip(0, 1).shift(1).fillna(1.0)
    else:
        dd_mult = pd.Series(1.0, index=raw.index)
    if gate_on:
        sv = scaled.rolling(60).std()
        thr = sv.rolling(252, min_periods=60).quantile(0.99)
        ok = (sv <= thr).shift(1).fillna(True).astype(float)
        gate_mult = ok + (1 - ok) * 0.5
    else:
        gate_mult = pd.Series(1.0, index=raw.index)
    total = (vol_mult * dd_mult * gate_mult)
    if smooth > 1:
        total = total.ewm(span=smooth).mean()
    net = raw * total - total.diff().abs().fillna(0) * (10 / 1e4)
    return net


def wf_weights(df, mode="invvol", win=252, freq=21, shrink=0.5):
    """Walk-forward weights, refreshed every `freq` days using trailing `win` days."""
    W = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    for i in range(win, len(df), freq):
        hist = df.iloc[i - win:i]
        avail = [c for c in df.columns if hist[c].notna().sum() > win * 0.6]
        h = hist[avail].fillna(0)
        if mode == "invvol":
            iv = 1.0 / h.std().clip(lower=1e-6)
            w = iv / iv.sum()
        elif mode == "erc":
            cov = h.cov().values
            corr = h.corr().values
            n = len(avail)
            corr_s = shrink * np.eye(n) + (1 - shrink) * corr
            sd = np.sqrt(np.diag(cov))
            cov_s = corr_s * np.outer(sd, sd)
            w = np.ones(n) / n
            for _ in range(200):
                mrc = cov_s @ w
                rc = w * mrc
                w = w * (rc.mean() / rc.clip(1e-12))
                w = np.clip(w, 0, None)
                w = w / w.sum()
            w = pd.Series(w, index=avail)
        W.iloc[i, [df.columns.get_loc(c) for c in avail]] = w.values
    return W.ffill()


def main():
    df5 = load_sleeves(with_credlo=False)
    df6 = load_sleeves(with_credlo=True)

    print("== A) baseline: static IS inv-vol weights + full overlay ==")
    raw = (df5.fillna(0) @ pd.Series(W_PROD))
    show(overlay(raw), "A prod replication")

    print("== B) + per-sleeve vol targeting (15%) ==")
    vt = df5.apply(vol_target_sleeve)
    raw_b = vt.fillna(0) @ pd.Series(W_PROD)
    show(overlay(raw_b), "B sleeve-VT + static weights")

    print("== C) walk-forward inverse-vol weights ==")
    for src, tag in [(df5, "raw sleeves"), (vt, "VT sleeves")]:
        W = wf_weights(src, "invvol")
        raw_c = (src.fillna(0) * W).sum(axis=1)
        raw_c = raw_c[W.notna().any(axis=1)]
        show(overlay(raw_c), f"C wf-invvol on {tag}")

    print("== D) walk-forward ERC weights ==")
    for shrink in [0.0, 0.5, 1.0]:
        W = wf_weights(vt, "erc", shrink=shrink)
        raw_d = (vt.fillna(0) * W).sum(axis=1)
        raw_d = raw_d[W.notna().any(axis=1)]
        show(overlay(raw_d), f"D wf-ERC shrink={shrink} on VT sleeves")

    print("== E) overlay ablation (on B config) ==")
    show(overlay(raw_b, dd_on=False), "E no DD throttle")
    show(overlay(raw_b, gate_on=False), "E no vol gate")
    show(overlay(raw_b, dd_on=False, gate_on=False), "E vol target only")
    show(overlay(raw_b, smooth=5), "E full overlay, 5d smoothed mult")

    print("== F) + CREDLO sleeve ==")
    vt6 = df6.apply(vol_target_sleeve)
    W6 = wf_weights(vt6, "invvol")
    raw_f = (vt6.fillna(0) * W6).sum(axis=1)
    raw_f = raw_f[W6.notna().any(axis=1)]
    show(overlay(raw_f), "F wf-invvol, 6 sleeves (VT)")
    W6e = wf_weights(vt6, "erc", shrink=0.5)
    raw_fe = (vt6.fillna(0) * W6e).sum(axis=1)
    raw_fe = raw_fe[W6e.notna().any(axis=1)]
    show(overlay(raw_fe), "F wf-ERC 0.5, 6 sleeves (VT)")
    show(overlay(raw_fe, dd_on=False), "F wf-ERC 0.5, 6 sleeves, no DD")

    print("== sleeve diag: VT effect per sleeve ==")
    for c in df6.columns:
        a, b = metr(df6[c].dropna()), metr(vol_target_sleeve(df6[c].dropna()))
        print(f"  {c:9s} raw SR={a['sr']:5.2f} -> VT SR={b['sr']:5.2f}")


if __name__ == "__main__":
    main()
