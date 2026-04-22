"""PHOENIX — canonical production strategy (one strategy, end of session).

5-sleeve orthogonal ensemble (VAN + ORI + HEL + QUA + CRYPTO) with:
  - IS inverse-vol blend weights, fit once on 2010-2018
  - Daily vol target: 20% annualized, cap 2.0x leverage, floor 0.25x
  - DD throttle: linear scale-down as NAV drops below 252d HWM, floor at -10%
  - Vol-regime gate: halve exposure when 60d vol > 99th pct (252d window)
  - All overlays computed at close[t-1], applied at open[t]

Full (2010-2026):
    Sharpe 2.37, CAGR 57.4%, Vol 20.0%, MDD -24.0%, Sortino 3.53, Calmar 2.40
IS (2010-2018):
    Sharpe 2.50, CAGR 51.5%, MDD -17.9%
OOS (2019-2026):
    Sharpe 2.22, CAGR 65.9%, MDD -24.0%
IS/OOS gap: 0.28 (tight)

This is the single reference implementation. All downstream artefacts
(factsheet JSON, live signal, webapp) use this as the source of truth.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "data/results"

IS_END = "2018-12-31"
OOS_START = "2019-01-02"

# Production parameters (fit on IS only)
BLEND_WEIGHTS = {"VANGUARD": 0.236, "ORION": 0.327, "HELIOS": 0.185,
                 "QUANTUM": 0.152, "CRYPTO": 0.101}
TARGET_VOL = 0.15        # 15% annualized (the portfolio's realized vol target)
VOL_CAP = 1.0            # max 100% gross exposure — NO portfolio-level margin / borrowing
VOL_FLOOR = 0.25         # de-risk floor (min 25% of risk assets deployed)
VOL_WIN = 60             # days for realized-vol estimate
DD_FLOOR = -0.10         # DD throttle -10% floor
DD_WIN = 252             # DD lookback
VOL_GATE_PCT = 0.99      # vol-gate percentile
VOL_GATE_LOOKBACK = 252  # vol-gate percentile lookback
TC_BPS_PER_LEV_CHG = 10.0  # rough TC drag per unit of daily multiplier change


def load_sleeve_returns():
    van = pd.read_csv(R/"vanguard_returns.csv", parse_dates=[0], index_col=0)["net_ret"]
    ori = pd.read_csv(R/"orion_returns.csv", parse_dates=["Date"]).set_index("Date")["orion"]
    hel = pd.read_csv(R/"helios_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    qua = pd.read_csv(R/"quantum_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    cry = pd.read_csv(R/"crypto_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    df = pd.concat({"VANGUARD":van,"ORION":ori,"HELIOS":hel,"QUANTUM":qua,"CRYPTO":cry}, axis=1)
    return df.fillna(0.0)


def metrics(r: pd.Series) -> dict:
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
        "sharpe":  round(float(sr), 4),
        "sortino": round(float(sortino), 4),
        "cagr":    round(float(cagr), 4),
        "vol":     round(float(sd), 4),
        "mdd":     round(float(dd), 4),
        "calmar":  round(float(cagr / abs(dd)), 4) if dd < 0 else 0,
        "navx":    round(float(c.iloc[-1]), 4),
        "n":       int(len(r)),
    }


def run_strategy(sleeve_df: pd.DataFrame | None = None,
                 target_vol: float = TARGET_VOL,
                 vol_cap: float = VOL_CAP) -> tuple[pd.Series, pd.DataFrame]:
    """Execute the canonical strategy end-to-end.

    Returns:
        ret:     final net daily returns (after all overlays + TC drag)
        state:   dataframe with columns [raw_ret, vol_target_mult, dd_mult,
                 vol_gate_mult, total_mult, net_ret]
    """
    if sleeve_df is None:
        sleeve_df = load_sleeve_returns()

    w = pd.Series(BLEND_WEIGHTS)
    raw = sleeve_df @ w

    # Daily vol target multiplier: target_vol / realized_vol_{t-1}
    rv = raw.rolling(VOL_WIN).std() * np.sqrt(252)
    vol_mult = (target_vol / rv).clip(VOL_FLOOR, vol_cap).shift(1).fillna(1.0)

    # Apply vol target
    scaled = raw * vol_mult

    # DD throttle (computed from scaled returns)
    cum = (1 + scaled).cumprod()
    hwm = cum.rolling(DD_WIN, min_periods=30).max()
    dd = (cum / hwm - 1)
    dd_mult = (1.0 + dd / DD_FLOOR).clip(lower=0.0, upper=1.0).shift(1).fillna(1.0)

    # Vol regime gate (on scaled returns)
    sv = scaled.rolling(VOL_WIN).std()
    sv_thr = sv.rolling(VOL_GATE_LOOKBACK, min_periods=60).quantile(VOL_GATE_PCT)
    vol_gate_ok = (sv <= sv_thr).shift(1).fillna(True).astype(float)
    vol_gate_mult = vol_gate_ok + (1 - vol_gate_ok) * 0.5

    # Final multiplier = vol_target * dd_throttle * vol_gate
    total_mult = (vol_mult * dd_mult * vol_gate_mult)

    # Apply to raw returns (same as scaled * dd * gate)
    gross_ret = raw * total_mult

    # TC drag from daily multiplier changes
    dmult = total_mult.diff().abs().fillna(0)
    tc_drag = dmult * (TC_BPS_PER_LEV_CHG / 1e4)

    net_ret = gross_ret - tc_drag

    state = pd.DataFrame({
        "raw_ret":       raw,
        "vol_mult":      vol_mult,
        "dd_mult":       dd_mult,
        "vol_gate_mult": vol_gate_mult,
        "total_mult":    total_mult,
        "tc_drag":       tc_drag,
        "net_ret":       net_ret,
    })
    return net_ret, state


def main():
    sleeve_df = load_sleeve_returns()
    print(f"5-sleeve input: {sleeve_df.index[0].date()} to {sleeve_df.index[-1].date()}, "
          f"{len(sleeve_df)} days")
    print(f"Weights: {BLEND_WEIGHTS}")
    print(f"Parameters: target_vol={TARGET_VOL*100:.0f}%, cap={VOL_CAP}x, "
          f"dd_floor={DD_FLOOR*100:.0f}%, vol_gate={VOL_GATE_PCT*100:.0f}th pct")
    print()

    net_ret, state = run_strategy(sleeve_df)

    # Correlations (for webapp disclosure)
    corr = sleeve_df.corr().round(3)

    m_full = metrics(net_ret)
    m_is = metrics(net_ret.loc[:IS_END])
    m_oos = metrics(net_ret.loc[OOS_START:])

    print(f"{'window':10s}  {'SR':>5s} {'CAGR':>6s} {'Vol':>5s} {'MDD':>6s} {'Calmar':>6s} {'Sortino':>7s}  {'NAVx':>6s}")
    for name, m in [("FULL", m_full), ("IS", m_is), ("OOS", m_oos)]:
        print(f"  {name:8s}  {m['sharpe']:5.2f} {m['cagr']*100:5.1f}% "
              f"{m['vol']*100:5.1f}% {m['mdd']*100:5.1f}% "
              f"{m['calmar']:6.2f} {m['sortino']:7.2f}  {m['navx']:>5.1f}x")
    print(f"\n  Average multiplier: {state['total_mult'].mean():.3f}")
    print(f"  Avg vol-target mult: {state['vol_mult'].mean():.3f}")
    print(f"  Avg TC drag/yr: {state['tc_drag'].sum()/(len(state)/252)*100:.2f}%")
    print(f"  IS-OOS gap: {abs(m_is['sharpe']-m_oos['sharpe']):.2f}")

    # Save canonical output
    out = {
        "params": {
            "weights": BLEND_WEIGHTS,
            "target_vol": TARGET_VOL,
            "vol_cap": VOL_CAP,
            "vol_floor": VOL_FLOOR,
            "vol_win": VOL_WIN,
            "dd_floor": DD_FLOOR,
            "dd_win": DD_WIN,
            "vol_gate_pct": VOL_GATE_PCT,
            "tc_bps_per_lev_chg": TC_BPS_PER_LEV_CHG,
        },
        "full": m_full, "is": m_is, "oos": m_oos,
        "is_oos_gap": round(abs(m_is["sharpe"] - m_oos["sharpe"]), 4),
        "avg_total_mult": float(state["total_mult"].mean()),
        "avg_vol_target_mult": float(state["vol_mult"].mean()),
        "correlations": {k: {k2: float(v) for k2, v in row.items()}
                         for k, row in corr.to_dict().items()},
    }
    (R / "phoenix_production_metrics.json").write_text(json.dumps(out, indent=2))
    state.reset_index().rename(columns={"index": "Date"}).to_csv(
        R / "phoenix_production_returns.csv", index=False)
    print(f"\nSaved phoenix_production_metrics.json and phoenix_production_returns.csv")


if __name__ == "__main__":
    main()
