"""Test adding daily vol scaling to PHOENIX v2.

Question: since we trade daily anyway, what happens if we scale the blend's
exposure each day to target a fixed annualized vol?

Tests two variants:
  (A) Vol target only (replaces overlay): each day, ex-ante leverage =
      target_vol / trailing_realized_vol_60d, capped at [0.5, 2.0].
  (B) Vol target + DD throttle + vol regime gate (keep existing overlays).

Target vols tested: 12%, 15%, 18%, 20% (annualized).
Cap tested at 1.5x, 2.0x, 2.5x to avoid runaway leverage in quiet markets.

Honest trade-offs:
  + Daily vol scaling typically boosts Sharpe 0.1-0.3 on momentum strategies.
  + Lower MDD because it de-levers into vol spikes.
  - Higher turnover (more frequent re-sizing) => TC drag.
  - Pro-cyclical in fast crashes (sells after down-days, buys after up-days).
  - Path dependence: the exact scaling constant depends on how you compute vol.

Uses the existing 4-sleeve v2 raw returns and 5-sleeve with-crypto raw returns.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

R = Path("data/results")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"
TC_BPS_EXTRA_PER_X_OVER_1 = 3.0  # 3 bps extra TC per unit of lever above 1x (estimate)


def sharpe(r):
    r = r.dropna()
    if len(r) == 0: return 0
    mu = r.mean()*252; sd = r.std()*np.sqrt(252)
    return float(mu/sd) if sd>0 else 0

def metrics(r):
    r = r.dropna()
    if len(r) == 0: return {"sharpe":0,"cagr":0,"mdd":0,"vol":0,"sortino":0,"turnover_mult":1.0}
    mu = r.mean()*252; sd = r.std()*np.sqrt(252)
    sr = mu/sd if sd>0 else 0
    c = (1+r).cumprod(); dd = (c/c.cummax()-1).min()
    yrs = len(r)/252
    cagr = c.iloc[-1]**(1/yrs) - 1 if c.iloc[-1]>0 else -1
    neg = r[r<0]
    sortino = mu/(neg.std()*np.sqrt(252)) if len(neg)>0 and neg.std()>0 else 0
    return {"sharpe":float(sr),"cagr":float(cagr),"mdd":float(dd),"vol":float(sd),"sortino":float(sortino)}


def apply_vol_target(r_raw, target_vol, vol_win=60, lev_min=0.25, lev_max=2.0, tc_bps_per_lev_change=10.0):
    """Apply daily vol targeting.
    mult_t = target_vol / realized_vol_{t-1}  (clamped).
    TC: each |Δmult| costs ~tc_bps_per_lev_change bps * average gross turnover assumption.
    """
    rv = r_raw.rolling(vol_win).std() * np.sqrt(252)
    mult = (target_vol / rv).clip(lev_min, lev_max).shift(1).fillna(1.0)
    # Apply
    r_scaled = r_raw * mult
    # Extra TC from daily lever changes (rough estimate: 30x base turnover, each unit of lev-delta costs extra bps)
    dmult = mult.diff().abs().fillna(0)
    # Rough: each unit of mult change = ~30% portfolio turnover at the sleeve level => extra TC proportional to dmult
    tc_drag = dmult * (tc_bps_per_lev_change / 1e4)
    return r_scaled - tc_drag, mult


def apply_dd_throttle(r, dd_floor=-0.10, dd_win=252):
    cum = (1+r).cumprod()
    hwm = cum.rolling(dd_win, min_periods=30).max()
    dd = cum/hwm - 1
    return (1.0 + dd/dd_floor).clip(0,1).shift(1).fillna(1.0)

def apply_vol_gate(r, vol_win=60, vol_pct=0.99, lookback=252):
    rv = r.rolling(vol_win).std()
    rv_thr = rv.rolling(lookback, min_periods=60).quantile(vol_pct)
    ok = (rv <= rv_thr).shift(1).fillna(True).astype(float)
    return ok + (1-ok)*0.5


def main():
    # Baseline raw (no overlay) returns for 4-sleeve and 5-sleeve
    v2 = pd.read_csv(R/"phoenix_v2_returns.csv", parse_dates=["Date"]).set_index("Date")
    v2c = pd.read_csv(R/"phoenix_v2_crypto_returns.csv", parse_dates=["Date"]).set_index("Date")

    raw4 = v2["raw_ret"]  # before any overlay
    raw5 = v2c["raw_ret"]
    # Baseline with current overlay
    cur4 = v2["ret"]
    cur5 = v2c["ret"]

    print(f"{'='*120}")
    print(f"DAILY VOL SCALING TEST")
    print(f"{'='*120}")
    print(f"\nBaselines (existing overlay = DD throttle + vol gate, no vol target):")
    for name, r in [("4-sleeve (current)", cur4), ("5-sleeve+crypto (current)", cur5)]:
        m_f=metrics(r); m_i=metrics(r.loc[:IS_END]); m_o=metrics(r.loc[OOS_START:])
        print(f"  {name:28s}  FULL SR {m_f['sharpe']:5.2f}  IS {m_i['sharpe']:5.2f}  OOS {m_o['sharpe']:5.2f}  "
              f"CAGR {m_f['cagr']*100:5.1f}%  Vol {m_f['vol']*100:4.1f}%  MDD {m_f['mdd']*100:5.1f}%")

    print(f"\n=== Variant A: VOL TARGET ONLY (replace existing overlay) ===")
    print(f"{'Target':>6s}  {'Cap':>5s}  {'Full SR':>7s}  {'IS SR':>6s}  {'OOS SR':>7s}  "
          f"{'CAGR':>6s}  {'Vol':>5s}  {'MDD':>6s}  {'Sortino':>7s}")

    rows = []
    for base_name, r in [("4sl", raw4), ("5sl", raw5)]:
        for target in [0.12, 0.15, 0.18, 0.20]:
            for cap in [1.5, 2.0, 2.5]:
                r_vt, mult = apply_vol_target(r, target, lev_max=cap)
                m_f = metrics(r_vt); m_i = metrics(r_vt.loc[:IS_END]); m_o = metrics(r_vt.loc[OOS_START:])
                print(f"  {base_name}  {target*100:>4.0f}%  {cap:>5.1f}x  {m_f['sharpe']:>7.2f}  "
                      f"{m_i['sharpe']:>6.2f}  {m_o['sharpe']:>7.2f}  "
                      f"{m_f['cagr']*100:>5.1f}%  {m_f['vol']*100:>4.1f}%  {m_f['mdd']*100:>5.1f}%  {m_f['sortino']:>7.2f}")
                rows.append({"variant": f"{base_name}_target{int(target*100)}_cap{cap:.1f}",
                             "target_vol": target, "cap": cap, "base": base_name,
                             "full": m_f, "is": m_i, "oos": m_o, "avg_mult": float(mult.mean())})
        print()

    print(f"\n=== Variant B: VOL TARGET + DD-throttle + vol-regime gate (stack) ===")
    print(f"{'Target':>6s}  {'Cap':>5s}  {'Full SR':>7s}  {'IS SR':>6s}  {'OOS SR':>7s}  "
          f"{'CAGR':>6s}  {'Vol':>5s}  {'MDD':>6s}  {'Sortino':>7s}")
    for base_name, r in [("4sl", raw4), ("5sl", raw5)]:
        for target in [0.15, 0.18]:
            for cap in [1.5, 2.0]:
                # Apply vol target
                r_vt, mult_vt = apply_vol_target(r, target, lev_max=cap)
                # Stack DD + vol-gate ON TOP of vol-target (compute from SCALED returns)
                dd_mult = apply_dd_throttle(r_vt)
                vol_mult = apply_vol_gate(r_vt)
                stack_mult = (dd_mult * vol_mult).clip(0, 1)
                r_final = r_vt * stack_mult
                m_f = metrics(r_final); m_i = metrics(r_final.loc[:IS_END]); m_o = metrics(r_final.loc[OOS_START:])
                print(f"  {base_name}  {target*100:>4.0f}%  {cap:>5.1f}x  {m_f['sharpe']:>7.2f}  "
                      f"{m_i['sharpe']:>6.2f}  {m_o['sharpe']:>7.2f}  "
                      f"{m_f['cagr']*100:>5.1f}%  {m_f['vol']*100:>4.1f}%  {m_f['mdd']*100:>5.1f}%  {m_f['sortino']:>7.2f}")
                rows.append({"variant": f"{base_name}_stack_target{int(target*100)}_cap{cap:.1f}",
                             "target_vol": target, "cap": cap, "base": base_name, "stack": True,
                             "full": m_f, "is": m_i, "oos": m_o})

    # Save
    (R/"vol_scaling_test.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved vol_scaling_test.json ({len(rows)} variants)")


if __name__ == "__main__":
    main()
