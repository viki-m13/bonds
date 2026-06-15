"""
Parameter bagging of the three rules-based PHOENIX sleeves
==========================================================

Goal
----
Remove specification / overfit luck from each rules-based sleeve (VANGUARD,
ORION, HELIOS) by running it across a small grid of NEARBY parameter values,
collecting each variant's daily NET-return stream, and equal-weight-averaging
those streams into a single "bagged" sleeve.

For each sleeve we compute, for both the CANONICAL single-parameter stream and
the BAGGED stream:
    * IS  (<= 2018-12-31)  Sharpe / CAGR / vol / max-drawdown
    * OOS (>= 2019-01-02)  Sharpe / CAGR / vol / max-drawdown
    * Full-sample          Sharpe / CAGR / vol / max-drawdown
    * IS -> OOS Sharpe gap

Rigor
-----
* Strictly causal: we ONLY vary the listed parameters. No look-ahead property of
  any sleeve is touched. Each sleeve's own panel building, costs and execution
  are reused verbatim -- the strategy logic is never reimplemented, only driven
  with different parameter values.
* Transaction costs are applied exactly as each sleeve already applies them.
* Bagging is equal-weight across variants. A variant contributes to a given day
  only if it has a (non-NaN) value on that day; the bagged value is the mean of
  the variants present that day.

Outputs (all under phoenix5/bagging/)
    vanguard_bagged_returns.csv   (Date, ret)
    orion_bagged_returns.csv      (Date, ret)
    helios_bagged_returns.csv     (Date, ret)
    bagging_metrics.json

Run:  python3 phoenix5/bagging/bag_sleeves.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Make the repo root importable so we can import the alt/* sleeve modules.
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]   # .../bonds
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import alt.vanguard_strategy as vanguard   # noqa: E402
import alt.orion_strategy as orion         # noqa: E402
import alt.helios_strategy as helios       # noqa: E402

OUT_DIR = ROOT / "phoenix5" / "bagging"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Common evaluation windows
IS_END    = pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-02")
EVAL_START = pd.Timestamp("2010-03-11")

ANN = 252


# --------------------------------------------------------------------------- #
# Shared metrics (annualised, mean/std*sqrt(252) Sharpe; geometric CAGR; mdd)
# --------------------------------------------------------------------------- #
def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 5:
        return {"sharpe": float("nan"), "cagr": float("nan"),
                "vol": float("nan"), "mdd": float("nan"), "n": int(len(r))}
    mu = r.mean() * ANN
    sigma = r.std(ddof=0) * np.sqrt(ANN)
    sharpe = float(mu / sigma) if sigma > 0 else float("nan")
    nav = (1.0 + r).cumprod()
    yrs = len(r) / ANN
    cagr = float(nav.iloc[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else float("nan")
    mdd = float((nav / nav.cummax() - 1.0).min())
    return {"sharpe": sharpe, "cagr": cagr, "vol": float(sigma),
            "mdd": mdd, "n": int(len(r))}


def windowed_metrics(r: pd.Series) -> dict:
    r = r.loc[EVAL_START:]
    is_r = r.loc[:IS_END]
    oos_r = r.loc[OOS_START:]
    m_is = metrics(is_r)
    m_oos = metrics(oos_r)
    m_full = metrics(r)
    gap = float(m_is["sharpe"] - m_oos["sharpe"])
    return {"is": m_is, "oos": m_oos, "full": m_full, "is_oos_sharpe_gap": gap}


def bag(streams: list[pd.Series]) -> pd.Series:
    """Equal-weight mean across variant streams.  A variant contributes only on
    dates where it has a non-NaN value (so the per-day mean is over present
    variants only)."""
    df = pd.concat(streams, axis=1)
    return df.mean(axis=1, skipna=True)


# --------------------------------------------------------------------------- #
# VANGUARD
# --------------------------------------------------------------------------- #
def vanguard_stream(mom_lb: int, sma_lb: int, vol_lb: int) -> pd.Series:
    """Daily NET return for one VANGUARD parameter set, reusing the sleeve's own
    panel building, gate, costs and execution (gross fixed at canonical 1.5)."""
    opens, _closes = vanguard.build_panels(vanguard.CORE)
    w = vanguard.build_weights(mom_lb=mom_lb, sma_lb=sma_lb, vol_lb=vol_lb, gross=1.5)
    bt = vanguard.backtest(opens, w).loc[vanguard.IS_START:]
    s = bt["net_ret"].copy()
    s.name = f"v_{mom_lb}_{sma_lb}_{vol_lb}"
    return s


def run_vanguard():
    canon = dict(mom_lb=189, sma_lb=200, vol_lb=60)
    grid_mom = [126, 150, 189, 210, 252]
    grid_sma = [150, 200, 250]
    grid_vol = [40, 60, 90]

    canon_stream = vanguard_stream(**canon)

    streams = []
    for mom, sma, vol in itertools.product(grid_mom, grid_sma, grid_vol):
        streams.append(vanguard_stream(mom, sma, vol))
    bagged = bag(streams)
    bagged.name = "ret"
    return canon_stream, bagged, len(streams)


# --------------------------------------------------------------------------- #
# ORION
# --------------------------------------------------------------------------- #
# ORION's signal helpers bind their lookback/MA as *default arguments* at def
# time (sig_momentum(lookback=MOM_LOOKBACK, skip=MOM_SKIP),
# sig_trend_filter(ma=TREND_MA)), and build_risk_sleeve / build_safe_sleeve call
# them with NO override.  So to vary MOM_LOOKBACK / TREND_MA we must rebind the
# function __defaults__ (mutating the module globals alone would NOT take
# effect).  We restore them afterward.
def _orion_opens():
    opens, _closes = orion.load_prices(orion.UNIVERSE)
    opens = opens.dropna(how="any")
    opens = opens.loc[orion.START_DATE:orion.END_DATE]
    return opens


def orion_stream(mom_lookback: int, trend_ma: int) -> pd.Series:
    """Daily NET return for one ORION parameter set.  Reuses build_weights (and
    therefore the sleeve's sleeve construction, weekly freeze, macro gate, and
    50/50 sleeve mix) plus backtest (costs + open-to-open execution)."""
    sig_mom_def = orion.sig_momentum.__defaults__       # (lookback, skip)
    sig_trend_def = orion.sig_trend_filter.__defaults__  # (ma,)
    try:
        # Keep MOM_SKIP at its canonical value (second default element).
        orion.sig_momentum.__defaults__ = (mom_lookback, sig_mom_def[1])
        orion.sig_trend_filter.__defaults__ = (trend_ma,)
        W = orion.build_weights()
        opens = _orion_opens()
        net, _dw = orion.backtest(W, opens)
        net = net.loc[orion.START_DATE:]
    finally:
        orion.sig_momentum.__defaults__ = sig_mom_def
        orion.sig_trend_filter.__defaults__ = sig_trend_def
    s = net.copy()
    s.name = f"o_{mom_lookback}_{trend_ma}"
    return s


def run_orion():
    canon = dict(mom_lookback=252, trend_ma=200)
    grid_mom = [189, 210, 252]
    grid_trend = [150, 200, 250]

    canon_stream = orion_stream(**canon)

    streams = []
    for mom, trend in itertools.product(grid_mom, grid_trend):
        streams.append(orion_stream(mom, trend))
    bagged = bag(streams)
    bagged.name = "ret"
    return canon_stream, bagged, len(streams)


# --------------------------------------------------------------------------- #
# HELIOS
# --------------------------------------------------------------------------- #
# HELIOS reads MOM_LB / MOM_SKIP / SMA_LB inside build_signals() and TOP_N
# inside build_target_weights() directly from the module globals at call time,
# so assigning the globals before each call takes effect.  We restore afterward.
def helios_stream(mom_lb: int, mom_skip: int, top_n: int) -> pd.Series:
    """Daily NET return for one HELIOS parameter set, reusing the sleeve's own
    panel building, macro gate, costs and next-day-open execution."""
    saved = (helios.MOM_LB, helios.MOM_SKIP, helios.TOP_N)
    try:
        helios.MOM_LB = mom_lb
        helios.MOM_SKIP = mom_skip
        helios.TOP_N = top_n
        close_u, opens = helios.build_panel()
        lev_firsts = []
        for lev in helios.PAIRS.values():
            ser = opens[lev].dropna()
            if len(ser):
                lev_firsts.append(ser.index.min())
        start = max(max(lev_firsts), helios.IS_START)
        close_u = close_u.loc[start:]
        opens = opens.loc[start:]
        W, _rebal = helios.build_target_weights(close_u, opens)
        bt = helios.run_backtest(W, opens).loc[helios.IS_START:]
        net = bt["ret"].copy()
    finally:
        helios.MOM_LB, helios.MOM_SKIP, helios.TOP_N = saved
    net.name = f"h_{mom_lb}_{mom_skip}_{top_n}"
    return net


def run_helios():
    canon = dict(mom_lb=189, mom_skip=42, top_n=2)
    grid_mom = [150, 189, 210]
    grid_skip = [21, 42, 63]
    grid_top = [2, 3]

    canon_stream = helios_stream(**canon)

    streams = []
    for mom, skip, top in itertools.product(grid_mom, grid_skip, grid_top):
        streams.append(helios_stream(mom, skip, top))
    bagged = bag(streams)
    bagged.name = "ret"
    return canon_stream, bagged, len(streams)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def save_returns(name: str, bagged: pd.Series):
    out = bagged.loc[EVAL_START:].rename("ret").to_frame()
    out.index.name = "Date"
    path = OUT_DIR / f"{name}_bagged_returns.csv"
    out.to_csv(path)
    return path


def main():
    sleeves = {}

    print("Running VANGUARD bagging ...")
    v_canon, v_bag, v_n = run_vanguard()
    save_returns("vanguard", v_bag)
    sleeves["vanguard"] = {
        "n_variants": v_n,
        "canonical": windowed_metrics(v_canon),
        "bagged": windowed_metrics(v_bag),
    }

    print("Running ORION bagging ...")
    o_canon, o_bag, o_n = run_orion()
    save_returns("orion", o_bag)
    sleeves["orion"] = {
        "n_variants": o_n,
        "canonical": windowed_metrics(o_canon),
        "bagged": windowed_metrics(o_bag),
    }

    print("Running HELIOS bagging ...")
    h_canon, h_bag, h_n = run_helios()
    save_returns("helios", h_bag)
    sleeves["helios"] = {
        "n_variants": h_n,
        "canonical": windowed_metrics(h_canon),
        "bagged": windowed_metrics(h_bag),
    }

    with open(OUT_DIR / "bagging_metrics.json", "w") as f:
        json.dump(sleeves, f, indent=2, default=float)

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    print()
    print("=" * 92)
    print("PARAMETER BAGGING SUMMARY  (Sharpe annualised; IS<=2018-12-31, OOS>=2019-01-02)")
    print("=" * 92)
    hdr = (f"{'Sleeve':9s} {'Variant':9s} {'#var':>4s} | "
           f"{'IS Sh':>7s} {'OOS Sh':>7s} {'Full Sh':>7s} | "
           f"{'IS->OOS gap':>11s} | {'OOS CAGR':>8s} {'OOS MDD':>8s}")
    print(hdr)
    print("-" * 92)
    for name in ["vanguard", "orion", "helios"]:
        s = sleeves[name]
        for variant in ["canonical", "bagged"]:
            m = s[variant]
            nvar = s["n_variants"] if variant == "bagged" else 1
            print(f"{name:9s} {variant:9s} {nvar:4d} | "
                  f"{m['is']['sharpe']:7.3f} {m['oos']['sharpe']:7.3f} {m['full']['sharpe']:7.3f} | "
                  f"{m['is_oos_sharpe_gap']:11.3f} | "
                  f"{m['oos']['cagr']*100:7.2f}% {m['oos']['mdd']*100:7.2f}%")
        print("-" * 92)

    print()
    print("Bagging effect (positive OOS-Sharpe delta and/or smaller |gap| = helped):")
    for name in ["vanguard", "orion", "helios"]:
        c = sleeves[name]["canonical"]
        b = sleeves[name]["bagged"]
        d_oos = b["oos"]["sharpe"] - c["oos"]["sharpe"]
        d_gap = abs(b["is_oos_sharpe_gap"]) - abs(c["is_oos_sharpe_gap"])
        verdict = "HELPED" if (d_oos > 0 or d_gap < 0) else "no improvement"
        print(f"  {name:9s}  dOOS Sharpe={d_oos:+.3f}  d|gap|={d_gap:+.3f}  -> {verdict}")

    print()
    print("Files written:")
    for fn in ["vanguard_bagged_returns.csv", "orion_bagged_returns.csv",
               "helios_bagged_returns.csv", "bagging_metrics.json"]:
        print(f"  {OUT_DIR / fn}")


if __name__ == "__main__":
    main()
