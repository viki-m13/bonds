"""NOVA8 — Concentrated dual-momentum on LEVERAGED ETFs.

Hypothesis: leveraged ETFs compound fiercely in persistent trends and
decay in chop. Dual-momentum (combining absolute trend filter with
cross-sectional rank) naturally concentrates into the strongest trend
while ejecting to cash in chop.

Design (fixed a priori, no per-parameter tuning):
  Universe: UPRO (3x SPY), TQQQ (3x QQQ), TMF (3x TLT), UGL (2x GLD), BIL
  Signal  : monthly, using last 6-month return (126 trading days).
            1-bar execution lag.
  Rule    : pick the single asset with the highest 6m return.
            If that asset's 6m return < BIL's 6m return (absolute
            momentum filter), hold BIL instead. Binary 100% allocation.
  Rebal   : monthly (first business day).
  TC      : 15 bps per switch (2-leg).

Plus a regime overlay (NOT vol scaling): if VIX > 35 OR SPY 63d vol >
40%, override to BIL regardless of momentum. Binary circuit-breaker.

Evaluate: full-sample SR + pre/post 2018 OOS. Per-year. Per-regime."""
from pathlib import Path
import numpy as np
import pandas as pd

from hydra_core import load_etf, load_fred, stats


TC_BPS = 15.0


def monthly_first_flag(index):
    out = pd.Series(False, index=index)
    out.iloc[0] = True
    for i in range(1, len(index)):
        if index[i].month != index[i - 1].month:
            out.iloc[i] = True
    return out


ASSETS = ["UPRO", "TQQQ", "TMF", "UGL", "BIL"]
LOOKBACK = 126    # 6 months in trading days


def load_all(dates):
    out = {}
    for t in ASSETS:
        p = load_etf(t)
        if p is None:
            raise RuntimeError(f"missing {t}")
        out[t] = p.reindex(dates).ffill()
    return out


def build_nova8(dates, use_circuit=True):
    prices = load_all(dates)
    mom = pd.DataFrame({t: prices[t].pct_change(LOOKBACK) for t in ASSETS})

    # Monthly selection: pick top momentum, with BIL as absolute-mom filter
    first = monthly_first_flag(pd.Index(dates))

    position = pd.Series("BIL", index=dates, dtype=object)
    # Compute at month-start only
    for i, d in enumerate(dates):
        if first.iloc[i]:
            m = mom.iloc[i]
            if m.isna().any():
                position.iloc[i] = "BIL"
                continue
            bil_m = m["BIL"]
            # Consider only risk assets (drop BIL from ranking)
            risk = m.drop("BIL")
            best = risk.idxmax()
            best_m = risk[best]
            if best_m > bil_m:
                position.iloc[i] = best
            else:
                position.iloc[i] = "BIL"
        else:
            position.iloc[i] = position.iloc[i - 1]

    # Circuit-breaker overlay
    if use_circuit:
        vix = load_fred("VIXCLS").reindex(dates).ffill()
        spy_vol = load_etf("SPY").reindex(dates).ffill().pct_change().rolling(63).std() * np.sqrt(252)
        breaker = ((vix > 35) | (spy_vol > 0.40)).fillna(False)
        # Evaluate breaker at month-start, hold for month
        breaker_m = pd.Series(False, index=dates)
        last_b = False
        for i, d in enumerate(dates):
            if first.iloc[i]:
                last_b = breaker.iloc[i]
            breaker_m.iloc[i] = last_b
        position[breaker_m] = "BIL"

    # Shift 1 bar for execution
    position_eff = position.shift(1).fillna("BIL")

    rets = {t: prices[t].pct_change().fillna(0) for t in ASSETS}
    r = pd.Series(0.0, index=dates)
    for t in ASSETS:
        mask = position_eff == t
        r.loc[mask] = rets[t].loc[mask]

    changes = (position_eff != position_eff.shift(1)).astype(int)
    tc = changes * (TC_BPS / 1e4) * 2
    r = r - tc
    return r, position_eff


def main():
    spy = load_etf("SPY")
    dates = spy.index
    print(f"Universe: {dates[0].date()} .. {dates[-1].date()}")
    print("NOVA8 — dual-momentum rotation: UPRO/TQQQ/TMF/UGL/BIL, 6m mom, monthly\n")

    # All assets must be live — earliest is TQQQ 2010-02 but we mask with BIL pre-live
    r, pos = build_nova8(dates, use_circuit=True)
    # Find first date where all assets have 126d momentum
    first_all_live = None
    for d in dates:
        ok = True
        for t in ASSETS:
            p = load_etf(t).reindex(dates).ffill()
            if d < p.dropna().index[0] + pd.Timedelta(days=LOOKBACK * 1.5):
                ok = False
                break
        if ok:
            first_all_live = d
            break
    print(f"All assets warm: {first_all_live.date() if first_all_live else 'N/A'}")

    r_v = r.loc[first_all_live:]
    pos_v = pos.loc[first_all_live:]

    print("\nPosition distribution:")
    print(pos_v.value_counts())

    s = stats(r_v, "NOVA8 (with circuit)")
    print(f"\n{s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    # Same without circuit
    r2, _ = build_nova8(dates, use_circuit=False)
    r2_v = r2.loc[first_all_live:]
    s = stats(r2_v, "NOVA8 (no circuit)")
    print(f"{s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    # IS/OOS split at 2018
    IS = pd.Timestamp("2018-01-01")
    for p, lbl in [(r_v.loc[:IS], "NOVA8 pre-2018"),
                   (r_v.loc[IS:], "NOVA8 post-2018")]:
        s = stats(p, lbl)
        print(f"{s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  MDD={s['mdd']:>7.2f}%")

    # Annual
    ann = r_v.groupby(r_v.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual:")
    print(ann.to_string())

    # 5y rolling windows
    print("\n5y windows:")
    for y0 in range(2011, 2022):
        y1 = y0 + 5
        sub = r_v.loc[pd.Timestamp(f"{y0}-01-01"):pd.Timestamp(f"{y1}-01-01")]
        if len(sub) < 200:
            continue
        s = stats(sub, f"{y0}-{y1-1}")
        print(f"  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  MDD={s['mdd']:>7.2f}%")


if __name__ == "__main__":
    main()
