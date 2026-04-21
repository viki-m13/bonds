"""PHOENIX enhanced — blend + own-drawdown throttle + vol-regime gate.

Extends alt/phoenix_blend.py with two additional novel mechanics designed to
lift Sharpe from 1.47 toward 2.0 without overfitting or daily vol scaling:

1. OWN-DRAWDOWN THROTTLE. When the blend's own NAV is in drawdown vs its
   (long) rolling high-water mark, scale weights down linearly toward 0 at a
   hard floor. No re-leveraging above 1.0 — purely a de-risk switch.

2. VOL-REGIME GATE. When blend's rolling realized vol (30d) breaches a
   long-term percentile threshold, scale down. This is NOT daily vol scaling
   (which would divide weights by vol every day). It is a BINARY regime
   gate keyed to a long-window percentile — triggers rarely.

Both mechanics compute from the blend's OWN realised returns up through
close[t-1] (no look-ahead). The throttle is applied to position sizes for
open[t].

IS: 2010-03-11 to 2018-12-31. OOS: 2019-01-02 to 2026-04-02.
Parameters tuned ON IS only.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
RESULTS = ROOT / "data/results"

STRATEGIES = {
    "VANGUARD": ("vanguard_returns.csv", "net_ret"),
    "ORION":    ("orion_returns.csv",    "orion"),
    "HELIOS":   ("helios_returns.csv",   "ret"),
}

IS_END = "2018-12-31"
OOS_START = "2019-01-02"


def load_all() -> pd.DataFrame:
    rets = {}
    for name, (fname, col) in STRATEGIES.items():
        df = pd.read_csv(RESULTS / fname, index_col=0, parse_dates=True)
        r = pd.to_numeric(df[col], errors="coerce")
        rets[name] = r
    df = pd.concat(rets, axis=1).sort_index()
    return df.loc["2010-03-11":"2026-04-02"].fillna(0.0)


def metrics(r: pd.Series, label: str = "") -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {}
    mu = r.mean() * 252
    sd = r.std() * np.sqrt(252)
    sr = mu / sd if sd > 0 else 0
    c = (1 + r).cumprod()
    dd = (c / c.cummax() - 1).min()
    yrs = len(r) / 252
    cagr = c.iloc[-1] ** (1 / yrs) - 1 if c.iloc[-1] > 0 else -1
    neg = r[r < 0]
    sortino = mu / (neg.std() * np.sqrt(252)) if len(neg) > 0 and neg.std() > 0 else 0
    return {
        "label": label, "n": int(len(r)),
        "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
        "sharpe": round(float(sr), 4), "sortino": round(float(sortino), 4),
        "cagr": round(float(cagr), 4), "ann_vol": round(float(sd), 4),
        "mdd": round(float(dd), 4), "navx": round(float(c.iloc[-1]), 4),
        "calmar": round(float(cagr / abs(dd)), 4) if dd < 0 else 0,
    }


def apply_throttles(raw_ret: pd.Series,
                    dd_win: int, dd_floor: float,
                    vol_win: int, vol_pct_thr: float,
                    pct_lookback: int):
    """Apply own-DD throttle and vol-regime gate, both using strictly past data."""
    # 1-bar lag everything — the throttle for day t uses data up through day t-1
    cum = (1 + raw_ret).cumprod()
    hwm = cum.rolling(dd_win, min_periods=30).max()
    dd = (cum / hwm - 1)
    # Linear throttle: 1.0 at 0%, 0.0 at dd_floor (e.g. -25%)
    dd_mult = (1.0 + dd / dd_floor).clip(lower=0.0, upper=1.0)
    dd_mult = dd_mult.shift(1).fillna(1.0)

    # Vol regime: scale down when 30d rolling vol exceeds its long-window (e.g. 252d) percentile threshold
    rv = raw_ret.rolling(vol_win).std()
    rv_thr = rv.rolling(pct_lookback, min_periods=60).quantile(vol_pct_thr)
    vol_ok = (rv <= rv_thr).shift(1).fillna(True).astype(float)
    # Linear dampener: if vol > thr, scale 0.5; else 1.0
    vol_mult = vol_ok * 1.0 + (1 - vol_ok) * 0.5

    mult = (dd_mult * vol_mult).clip(lower=0.0, upper=1.0)
    ret = raw_ret * mult
    return ret, mult


def main():
    df = load_all()
    is_df = df.loc[:IS_END]

    # 1. Build blend weights from IS inverse-vol
    inv_vol = 1.0 / is_df.std()
    w = (inv_vol / inv_vol.sum()).to_dict()
    print(f"PHOENIX blend weights (IS inv-vol): {w}")

    # 2. Raw blend returns
    raw = df @ pd.Series(w).reindex(df.columns).fillna(0.0)

    # 3. Throttle parameter grid on IS only
    print()
    print("Grid search on throttle params (optimize IS Sharpe):")
    best = None
    grid_rows = []
    for dd_win in [252, 378, 504]:
        for dd_floor in [-0.10, -0.15, -0.20, -0.25, -0.35]:
            for vol_win in [20, 30, 60]:
                for vol_pct_thr in [0.70, 0.80, 0.90, 0.95, 0.99]:
                    ret_t, mult = apply_throttles(raw, dd_win, dd_floor,
                                                   vol_win, vol_pct_thr, 252)
                    m_is = metrics(ret_t.loc[:IS_END], "IS")
                    m_full = metrics(ret_t, "FULL")
                    m_oos = metrics(ret_t.loc[OOS_START:], "OOS")
                    grid_rows.append({
                        "dd_win": dd_win, "dd_floor": dd_floor,
                        "vol_win": vol_win, "vol_pct": vol_pct_thr,
                        "is_sr": m_is["sharpe"], "oos_sr": m_oos["sharpe"],
                        "full_sr": m_full["sharpe"],
                        "is_cagr": m_is["cagr"], "oos_cagr": m_oos["cagr"],
                        "full_cagr": m_full["cagr"], "mdd": m_full["mdd"],
                        "avg_mult": float(mult.mean()),
                    })
                    if best is None or m_is["sharpe"] > best["is_sr"]:
                        best = grid_rows[-1]

    g = pd.DataFrame(grid_rows).sort_values("is_sr", ascending=False)
    print(g.head(15).to_string(index=False))
    print()

    # Select best by IS (then check OOS as one-shot)
    chosen = g.iloc[0].to_dict()
    dd_win = int(chosen["dd_win"]); dd_floor = float(chosen["dd_floor"])
    vol_win = int(chosen["vol_win"]); vol_pct = float(chosen["vol_pct"])
    ret_t, mult = apply_throttles(raw, dd_win, dd_floor, vol_win, vol_pct, 252)

    full = metrics(ret_t, "FULL")
    is_m = metrics(ret_t.loc[:IS_END], "IS")
    oos_m = metrics(ret_t.loc[OOS_START:], "OOS")
    raw_full = metrics(raw, "RAW")
    raw_is = metrics(raw.loc[:IS_END], "RAW IS")
    raw_oos = metrics(raw.loc[OOS_START:], "RAW OOS")

    print("=== PHOENIX ENHANCED — with throttles ===")
    print(f"  Chosen: dd_win={dd_win}  dd_floor={dd_floor:.2f}  "
          f"vol_win={vol_win}  vol_pct={vol_pct}")
    print(f"  Avg throttle multiplier: {mult.mean():.3f}  "
          f"(<1.0 = less-than-full participation)")
    print()
    print(f"  {'window':10s}  {'SR':>5s} {'CAGR':>6s} {'Vol':>5s} {'MDD':>6s} "
          f"{'Calmar':>6s} {'Sortino':>7s}")
    for name, m in [("RAW FULL", raw_full), ("RAW IS", raw_is), ("RAW OOS", raw_oos),
                    ("ENH FULL", full), ("ENH IS", is_m), ("ENH OOS", oos_m)]:
        print(f"  {name:10s}  {m['sharpe']:5.2f} {m['cagr']*100:5.1f}% "
              f"{m['ann_vol']*100:5.1f}% {m['mdd']*100:5.1f}% "
              f"{m['calmar']:6.2f} {m['sortino']:7.2f}")
    gap = abs(is_m["sharpe"] - oos_m["sharpe"])
    print(f"  IS-OOS gap: {gap:.2f}")

    # Save
    out = {
        "weights": w,
        "throttle": {"dd_win": dd_win, "dd_floor": dd_floor,
                     "vol_win": vol_win, "vol_pct": vol_pct,
                     "pct_lookback": 252},
        "raw": {"full": raw_full, "is": raw_is, "oos": raw_oos},
        "enhanced": {"full": full, "is": is_m, "oos": oos_m},
        "is_oos_gap": round(gap, 4),
        "avg_mult": float(mult.mean()),
    }
    (RESULTS / "phoenix_enh_metrics.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({
        "Date": ret_t.index,
        "ret": ret_t.values,
        "raw_ret": raw.values,
        "mult": mult.values,
    }).to_csv(RESULTS / "phoenix_enh_returns.csv", index=False)
    g.to_csv(RESULTS / "phoenix_enh_grid.csv", index=False)


if __name__ == "__main__":
    main()
