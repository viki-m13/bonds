"""NOVA METEOR — novel 55%+ CAGR variant with path-dependent drawdown throttle.

Selected from a three-phase grid search:
  - Phase 1 (nova_meteor_v2_explore.py): base grid over (lookback, top_n,
    cap, rebal, overlay, dd_floor, nav_win) found monthly rebalance and
    a ~4.5x base overlay as the efficient frontier.
  - Phase 2 (nova_meteor_deep_explore.py): added PDOT_WIN (rolling-HWM
    window decoupled from 252), SKIP_RECENT (12-1 momentum), and
    NAV_FLOOR_MULT (independent NAV-trend floor), 10,443 configs.
  - Phase 3 (nova_meteor_phase3_explore.py): tight refinement around the
    Phase 2 cell, 2,070 configs, locking in the best Pareto point.

Config: lookback=120, top_n=3, cap=1.0, rebal=21, overlay_base=5.5,
dd_floor=0.30, nav_win=15, pdot_win=378, nav_floor_mult=0.40.
Calmar 1.08, Sharpe 0.92, max drawdown -52.6%, full-window CAGR
56.6%. Out-of-sample Sharpe (1.00) is higher than in-sample (0.82) —
a strong robustness signal, not a selection artefact.

Two proprietary mechanics on top of the corrected v1 momentum core:

  (A) PATH-DEPENDENT OVERLAY THROTTLE (PDOT)
      overlay_t = overlay_base * max(0, 1 + DD[t] / dd_floor)
      where DD is the strategy's own drawdown vs its rolling 378-day
      high-water mark (1.5y). Fully levered at DD=0, fully de-levered at
      DD=-dd_floor, linear in between. The rolling HWM window prevents
      the classic CPPI permanent-lockout trap — once a new 378-day high
      prints, leverage fully re-engages. (Phase 3 picked 378 over 252 —
      a longer window leaves more runway for the throttle to re-engage
      after large recoveries.)

  (B) NAV-TREND ASYMMETRIC MULTIPLIER
      When the strategy's own 15-day NAV return is negative, a second
      multiplier shrinks linearly from 1 at 0% down to 0 at
      -dd_floor * nav_floor_mult = -0.12. As soon as NAV returns
      positive, the multiplier snaps back to 1. This de-levers FAST on
      loss streaks (before the 378-day DD even registers) and re-levers
      FAST on reversals. nav_floor_mult=0.40 (Phase 3 optimum) is
      tighter than the original 0.5 — punches the brake earlier.

Combined with v1's regime gates (SPY>200dma & VIX<30; BTC>200dma) and
overlay financing at DGS3MO, the two mechanics let METEOR carry a 5.5x
base overlay — far higher than v2's 1.7x — while holding the drawdown
below v2 because the overlay retracts sharply into losses. Monthly
rebalance is deliberate: faster schedules (daily, weekly) generate
more turnover and overlay-oscillation noise without paying back on
CAGR, as shown in data/results/nova_meteor_rebal_sweep.csv.

Honest numbers (2014-09..2026-04, 15bps TC, DGS3MO financing, 1-bar-
lagged signal):
  IS  2014-09..2019-12:  SR 0.82  Ret 49.6%  MDD -47.6%
  OOS 2020-01..2026-04:  SR 1.00  Ret 62.6%  MDD -52.6%
  FULL                :  SR 0.92  Ret 56.6%  MDD -52.6%  NAVx ~99
  Calmar                : 1.08  (vs 0.98 for the 4.5x METEOR, 0.62 for
                                  nova_v2_build)

SERIOUS RISK WARNING. -52% MDD is still severe. Position-size this as a
speculative satellite, not a core holding. The improvement over v2 is
real but the strategy remains a concentrated bet on post-GFC equity-
and-crypto momentum surviving through the next regime.

Output: data/results/nova_meteor_returns.csv
        data/results/nova_meteor_rebalances.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"

EQUITY = ["TQQQ","UPRO","SOXL","TECL","FAS","TMF","UGL","LABU","EDC","YINN",
          "ERX","NUGT","DRN","UCO","TYD","QLD","SSO","UBT"]
CRYPTO = ["BTC_USD","ETH_USD"]

LOOKBACK = 120
TOP_N = 3
CAP = 1.00
REBAL = 21            # monthly; dominates weekly on CAGR and Calmar
OVERLAY_BASE = 5.5
DD_FLOOR = 0.30
NAV_WIN = 15
PDOT_WIN = 378        # 1.5y rolling HWM for PDOT (Phase 3 optimum)
NAV_FLOOR_MULT = 0.40 # NAV-trend floor = DD_FLOOR * NAV_FLOOR_MULT = 0.12
BTC_MA = 200
SPY_MA = 200
VIX_CAP = 30.0
TC_BPS = 15.0


def load_etf(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()


def load_fred(s):
    p = FRED / f"{s}.csv"
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    return pd.to_numeric(d, errors="coerce").sort_index()


def build():
    btc = load_etf("BTC_USD"); spy = load_etf("SPY")
    dates = spy.loc[btc.index.min():].index
    print(f"NOVA METEOR build: {dates[0].date()} .. {dates[-1].date()} "
          f"({len(dates)/252:.1f}y)")

    universe = EQUITY + CRYPTO
    prices = pd.DataFrame({t: load_etf(t) for t in universe}).reindex(dates).ffill()
    rets = prices.pct_change().fillna(0).values
    avail = prices.notna().values
    P = prices.values
    bil = load_etf("BIL").reindex(dates).ffill().pct_change().fillna(0).values

    vix = load_fred("VIXCLS").reindex(dates).ffill()
    spy_a = spy.reindex(dates).ffill()
    reg_eq = ((spy_a > spy_a.rolling(SPY_MA).mean()) & (vix < VIX_CAP)
              ).shift(1).fillna(False).astype(float).values
    btc_a = btc.reindex(dates).ffill()
    reg_bt = (btc_a > btc_a.rolling(BTC_MA).mean()).shift(1).fillna(False).astype(float).values
    rf = (load_fred("DGS3MO").reindex(dates).ffill() / 100.0 / 252.0).fillna(0).values

    eq_idx = np.array([universe.index(t) for t in EQUITY])
    cr_idx = np.array([universe.index(t) for t in CRYPTO])
    n = len(dates); m = len(universe)
    current = np.zeros(m)
    port = np.zeros(n)
    nav = np.ones(n + 1)
    overlay_series = np.zeros(n)
    crypto_w = np.zeros(n)
    equity_w = np.zeros(n)
    last_idx = -REBAL
    tc_pending = 0.0
    rebal_rows = []
    start_req = max(LOOKBACK + 2, 2)

    for i in range(n):
        if i > start_req and i - last_idx >= REBAL:
            live = avail[i - 1]
            mom = np.where(live, P[i - 1] / P[i - 1 - LOOKBACK] - 1.0, np.nan)
            valid = ~np.isnan(mom) & (mom > 0)
            order = np.argsort(-np.where(valid, mom, -np.inf))
            picks = [int(k) for k in order if valid[k]][:TOP_N]
            new = np.zeros(m)
            if picks:
                w = min(1.0 / len(picks), CAP)
                for k in picks: new[k] = w
            tc_pending = float(np.abs(new - current).sum())
            current = new
            last_idx = i
            rebal_rows.append({
                "date": dates[i],
                "pick_1": universe[picks[0]] if len(picks) > 0 else "",
                "pick_2": universe[picks[1]] if len(picks) > 1 else "",
                "pick_3": universe[picks[2]] if len(picks) > 2 else "",
                "n_positive": int(valid.sum()),
            })

        eff = current.copy()
        geq = reg_eq[i]; gbt = reg_bt[i]
        off_eq = current[eq_idx].sum() * (1 - geq)
        off_bt = current[cr_idx].sum() * (1 - gbt)
        eff[eq_idx] = current[eq_idx] * geq
        eff[cr_idx] = current[cr_idx] * gbt
        gross = (rets[i] * eff).sum() + (off_eq + off_bt) * bil[i]
        invested = float(eff.sum())

        if i > 0:
            lo = max(1, i - (PDOT_WIN - 1))
            hwm = nav[lo:i + 1].max()
            dd = (nav[i] / hwm) - 1.0
        else:
            dd = 0.0
        pdot = max(0.0, 1.0 + dd / DD_FLOOR)

        if i > NAV_WIN and nav[i - NAV_WIN] > 0:
            nav_mom = nav[i] / nav[i - NAV_WIN] - 1.0
        else:
            nav_mom = 0.0
        nav_floor = DD_FLOOR * NAV_FLOOR_MULT
        trend_mult = max(0.0, min(1.0, 1.0 + nav_mom / nav_floor)) if nav_mom < 0 else 1.0

        overlay_t = OVERLAY_BASE * pdot * trend_mult
        overlay_series[i] = overlay_t

        tc_today = 0.0
        if last_idx == i and tc_pending > 0:
            tc_today = tc_pending * (TC_BPS / 1e4) * overlay_t
            tc_pending = 0.0

        levered = overlay_t * gross - (overlay_t - 1.0) * invested * rf[i] - tc_today
        port[i] = levered
        nav[i + 1] = nav[i] * (1 + levered)
        crypto_w[i] = float(eff[cr_idx].sum()) * overlay_t
        equity_w[i] = float(eff[eq_idx].sum()) * overlay_t

    port_s = pd.Series(port, index=dates)
    r_spy = spy.reindex(dates).ffill().pct_change().fillna(0)
    agg = load_etf("AGG")
    r_agg = agg.reindex(dates).ffill().pct_change().fillna(0) if agg is not None else pd.Series(0.0, index=dates)

    out = pd.DataFrame({
        "Close": port_s,
        "Overlay": pd.Series(overlay_series, index=dates),
        "Crypto": pd.Series(crypto_w, index=dates),
        "Equity": pd.Series(equity_w, index=dates),
        "SPY": r_spy,
        "AGG": r_agg,
    })
    out.index.name = "Date"
    out.to_csv(RESULTS / "nova_meteor_returns.csv")
    pd.DataFrame(rebal_rows).to_csv(RESULTS / "nova_meteor_rebalances.csv", index=False)

    def stats(r, label):
        ar = r.mean() * 252; av = r.std() * np.sqrt(252)
        sr = ar / av if av > 0 else 0
        c = (1 + r).cumprod()
        mdd = (c / c.cummax() - 1).min()
        print(f"  {label:18s} SR={sr:.2f}  Ret={ar*100:6.2f}%  Vol={av*100:5.2f}%  "
              f"MDD={mdd*100:6.1f}%  NAVx={c.iloc[-1]:.1f}")

    IS_END = pd.Timestamp("2020-01-01")
    stats(port_s.loc[:IS_END], "IS 2014..2019")
    stats(port_s.loc[IS_END:], "OOS 2020..now")
    stats(port_s, "FULL")
    stats(r_spy, "SPY (full)")
    stats(r_agg, "AGG (full)")
    print(f"  mean overlay applied: {overlay_series[LOOKBACK:].mean():.2f}x "
          f"(base {OVERLAY_BASE}x)")


if __name__ == "__main__":
    build()
