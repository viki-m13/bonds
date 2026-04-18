"""Step 2: Cross-asset momentum rotation among leveraged ETFs,
gated by SPY>200dma and VIX regime. Monthly rebalance.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("/home/user/bonds/data")
ETF = DATA / "etfs"
FRED = DATA / "fred"


def load(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()


def load_fred(s):
    p = FRED / f"{s}.csv"
    if not p.exists(): return None
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    return pd.to_numeric(d, errors="coerce").sort_index()


def metrics(r):
    r = r.loc[r.ne(0).idxmax():] if (r != 0).any() else r
    if r.std() == 0 or len(r) == 0: return None
    ar, av = r.mean() * 252, r.std() * np.sqrt(252)
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    neg = r[r < 0]
    sor = ar / (neg.std() * np.sqrt(252)) if len(neg) and neg.std() > 0 else 0
    return {"sharpe": ar/av if av > 0 else 0, "ret": ar, "vol": av, "mdd": dd,
            "sortino": sor, "calmar": ar/abs(dd) if dd < 0 else 0,
            "total": cum.iloc[-1] - 1, "n": len(r)/252,
            "start": r.index[0].date(), "end": r.index[-1].date()}


def momo_rotation(universe, lookback_days, top_n, gate_fn, start="2014-01-01",
                  rebalance_days=21, tc_bps=15.0, cash_ticker="BIL"):
    prices = pd.DataFrame({t: load(t) for t in universe}).dropna()
    prices = prices.loc[start:]
    rets = prices.pct_change().fillna(0)
    dates = rets.index
    cash = load(cash_ticker).reindex(dates).ffill().pct_change().fillna(0)
    gate = gate_fn(dates)
    current = pd.Series(0.0, index=universe)
    port = pd.Series(0.0, index=dates)
    last_idx = -rebalance_days
    rebal_count = 0
    for i, d in enumerate(dates):
        if i >= lookback_days and i - last_idx >= rebalance_days:
            momo = prices.iloc[i] / prices.iloc[i - lookback_days] - 1
            ranked = momo.sort_values(ascending=False)
            top = ranked.head(top_n).index.tolist()
            new_target = pd.Series(0.0, index=universe)
            # Equal weight among top_n with positive momentum only
            positive = [t for t in top if momo[t] > 0]
            if positive:
                w = 1.0 / len(positive)
                for t in positive:
                    new_target[t] = w
            tc = (new_target - current).abs().sum() * (tc_bps / 1e4)
            port.iloc[i] -= tc
            current = new_target
            last_idx = i
            rebal_count += 1
        r = (rets.iloc[i] * current).sum()
        g = float(gate.get(d, 0))
        r = g * r + (1 - g) * cash.iloc[i]
        port.iloc[i] += r
    return port, rebal_count


def gate_spy_200(dates):
    spy = load("SPY").reindex(dates).ffill()
    return (spy > spy.rolling(200).mean()).shift(1).fillna(False).astype(float)


def gate_spy_and_vix(dates, vix_cap=25):
    spy = load("SPY").reindex(dates).ffill()
    spy_ok = (spy > spy.rolling(200).mean())
    vix = load_fred("VIXCLS")
    if vix is not None:
        v = vix.reindex(dates).ffill()
        vix_ok = (v < vix_cap)
    else:
        vix_ok = pd.Series(True, index=dates)
    return (spy_ok & vix_ok).shift(1).fillna(False).astype(float)


def gate_always(dates):
    return pd.Series(1.0, index=dates)


# Universes
U_3X_EQUITY = ["TQQQ", "UPRO", "SOXL", "TECL", "FAS"]
U_3X_MIX   = ["TQQQ", "UPRO", "SOXL", "TECL", "FAS", "TMF", "UGL"]
U_2X_MIX   = ["QLD", "SSO", "UGL", "TLT", "UBT"]
U_FULL     = ["TQQQ", "UPRO", "SOXL", "TECL", "FAS", "TMF", "UGL",
              "QLD", "SSO", "BITO", "SVXY"]
U_WITH_ZEPHYR = ["TQQQ", "UPRO", "TECL", "SOXL", "TMF", "UGL", "BITO", "JAAA", "JPST"]

experiments = [
    ("3x_eq_top1_126_spygate",  U_3X_EQUITY, 126, 1, gate_spy_200),
    ("3x_eq_top1_63_spygate",   U_3X_EQUITY, 63,  1, gate_spy_200),
    ("3x_eq_top2_126_spygate",  U_3X_EQUITY, 126, 2, gate_spy_200),
    ("3x_mix_top1_126_spygate", U_3X_MIX,    126, 1, gate_spy_200),
    ("3x_mix_top2_126_spygate", U_3X_MIX,    126, 2, gate_spy_200),
    ("3x_mix_top1_63_spygate",  U_3X_MIX,    63,  1, gate_spy_200),
    ("3x_mix_top2_63_vixgate",  U_3X_MIX,    63,  2, gate_spy_and_vix),
    ("3x_mix_top1_252_spygate", U_3X_MIX,    252, 1, gate_spy_200),
    ("2x_mix_top2_126_spygate", U_2X_MIX,    126, 2, gate_spy_200),
    ("full_top2_126_spygate",   U_FULL,      126, 2, gate_spy_200),
    ("full_top3_126_spygate",   U_FULL,      126, 3, gate_spy_200),
]

print(f"{'Experiment':<32}{'SR':>6}{'Ret':>8}{'Vol':>7}{'MDD':>8}{'Sor':>6}{'Cal':>6}{'Rbls':>5}{'Yrs':>6}")
for name, uni, lb, tn, g in experiments:
    try:
        port, rc = momo_rotation(uni, lb, tn, g)
        m = metrics(port)
        if m is None: continue
        print(f"{name:<32}{m['sharpe']:6.2f}{m['ret']*100:7.1f}%{m['vol']*100:6.1f}%"
              f"{m['mdd']*100:7.1f}%{m['sortino']:6.2f}{m['calmar']:6.2f}{rc:5}{m['n']:6.1f}")
    except Exception as e:
        print(f"{name:<32}  ERROR: {e}")
