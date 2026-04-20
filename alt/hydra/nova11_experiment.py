"""NOVA11 — two orthogonal, structurally different sleeves.

After NOVA2-10 all ceilinged ~0.7 OOS SR with naive return-forecasting,
switch framing:

  Sleeve A: CHRONOS DISPERSION GATE
    Reuses NOVA10 monthly forecasts (pred_median, pred_width). Novel
    insight: the WIDTH of Chronos quantiles is a forecast-uncertainty
    signal independent of direction. When the model is CONFIDENT
    (narrow 10-90 width) we lever; when uncertain, we sit in cash.
    This converts epistemic uncertainty from ML into a regime gate,
    which is structurally different from using p(up) as direction.

      Rule (fixed a priori, no tuning):
        w < 0.08  (very narrow)   → UPRO if med>0 else SDS
        w < 0.12  (narrow)        → SPY  if med>0 else SH
        w >= 0.12 (wide)          → BIL
      Monthly rebalance, 1-bar lag, 15 bps TC.

  Sleeve B: DISCIPLINED VIX CONTANGO HARVEST
    Short vol term premium is real and persistent (VIX futures in
    contango ~75% of the time, roll yield avg -10%/yr). 2018 Volmageddon
    killed naive versions. Disciplined version:
      - Long SVXY only when ALL three hold at daily close:
          VIX < 18
          VIX < VIX.rolling(10).mean()
          SPY > SPY.rolling(200).mean()
      - Hard stop: if VIX jumps >15% in a day OR VIX > 22 at any time,
        force flat and LOCK OUT for 20 trading days
      - Size: 50% SVXY, 50% BIL (halved to survive tail events)
      - 15 bps TC each switch

Ensemble: equal-risk-contribution between A and B, based on pre-2018
vol only (no OOS peek). Report IS/OOS separately."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
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


def _tc_on_change(pos_series, dates, two_leg=True):
    changes = (pos_series != pos_series.shift(1)).astype(int)
    return changes * (TC_BPS / 1e4) * (2 if two_leg else 1)


# ===== SLEEVE A: Chronos Dispersion Gate =====

def sleeve_chronos_gate(dates):
    """Map (pred_median, pred_width) from NOVA10 CSV to positions.
    Width-gated: narrow forecast → lever, wide → cash.
    Direction comes from sign of median."""
    df = pd.read_csv("/home/user/bonds/data/results/nova10_returns.csv",
                     parse_dates=["Date"]).set_index("Date")
    med = df["pred_median"].reindex(dates)
    w = df["pred_width"].reindex(dates)

    first = monthly_first_flag(pd.Index(dates))
    position = pd.Series("BIL", index=dates, dtype=object)
    last = "BIL"
    for i, d in enumerate(dates):
        if first.iloc[i] and not pd.isna(w.iloc[i]) and not pd.isna(med.iloc[i]):
            ww = w.iloc[i]
            mm = med.iloc[i]
            if ww < 0.08:
                last = "UPRO" if mm > 0 else "SDS"
            elif ww < 0.12:
                last = "SPY" if mm > 0 else "SH"
            else:
                last = "BIL"
        position.iloc[i] = last
    position_eff = position.shift(1).fillna("BIL")

    tickers = ["UPRO", "SPY", "BIL", "SH", "SDS"]
    rets = {t: (load_etf(t).reindex(dates).ffill().pct_change().fillna(0)
                if load_etf(t) is not None else pd.Series(0.0, index=dates))
            for t in tickers}
    r = pd.Series(0.0, index=dates)
    for t in tickers:
        mask = position_eff == t
        r.loc[mask] = rets[t].loc[mask]
    r = r - _tc_on_change(position_eff, dates, two_leg=True)
    return r.rename("chronos_gate"), position_eff


# ===== SLEEVE B: VIX Contango Harvest =====

def sleeve_vix_harvest(dates):
    """Long SVXY when VIX low & calm & SPY above 200d. Hard circuit:
    any VIX jump >15% day OR VIX>22 forces flat for 20 days."""
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    spy = load_etf("SPY").reindex(dates).ffill()
    vix_ma10 = vix.rolling(10).mean()
    spy_ma200 = spy.rolling(200).mean()

    cond = (vix < 18) & (vix < vix_ma10) & (spy > spy_ma200)

    # VIX jump detection
    vix_ret = vix.pct_change()
    jump = (vix_ret > 0.15) | (vix > 22)

    # State machine: signal_on, cooldown_days_remaining
    signal_on = pd.Series(False, index=dates)
    cooldown = 0
    for i, d in enumerate(dates):
        if cooldown > 0:
            cooldown -= 1
            signal_on.iloc[i] = False
            continue
        if bool(jump.iloc[i]) if not pd.isna(jump.iloc[i]) else False:
            cooldown = 20
            signal_on.iloc[i] = False
            continue
        signal_on.iloc[i] = bool(cond.iloc[i]) if not pd.isna(cond.iloc[i]) else False

    signal_eff = signal_on.shift(1).fillna(False).astype(bool)

    svxy = load_etf("SVXY")
    if svxy is None:
        return pd.Series(0.0, index=dates, name="vix_harvest"), signal_eff
    svxy_ret = svxy.reindex(dates).ffill().pct_change().fillna(0)
    bil = load_etf("BIL").reindex(dates).ffill().pct_change().fillna(0)

    r = pd.Series(0.0, index=dates)
    # 50% SVXY + 50% BIL when on, 100% BIL when off
    on_mask = signal_eff
    r[on_mask] = 0.5 * svxy_ret[on_mask] + 0.5 * bil[on_mask]
    r[~on_mask] = bil[~on_mask]

    # TC on state change (50% of book moves)
    changes = (signal_eff != signal_eff.shift(1)).astype(int).fillna(0)
    r = r - changes * (TC_BPS / 1e4) * 1.0   # 50% turnover × 2 legs = 1x
    return r.rename("vix_harvest"), signal_eff


# ===== Ensemble =====

def main():
    spy = load_etf("SPY")
    dates = spy.index
    print(f"Universe: {dates[0].date()} .. {dates[-1].date()}")
    print("NOVA11 — Chronos dispersion gate × VIX contango harvest\n")

    rA, posA = sleeve_chronos_gate(dates)
    rB, posB = sleeve_vix_harvest(dates)

    # SVXY is live from 2011-10; Chronos has predictions from ~2007
    first_valid = pd.Timestamp("2011-10-05")   # SVXY start
    rA_v = rA.loc[first_valid:]
    rB_v = rB.loc[first_valid:]

    print("Sleeve A (Chronos gate) position distribution:")
    print(posA.loc[first_valid:].value_counts())
    print("\nSleeve B (VIX harvest) on/off:")
    print(posB.loc[first_valid:].astype(int).value_counts().to_dict())

    for r, lbl in [(rA_v, "A: Chronos gate"), (rB_v, "B: VIX harvest")]:
        s = stats(r, lbl)
        print(f"\n{s['label']:28s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    # Correlation
    corr = rA_v.corr(rB_v)
    print(f"\nCorr(A,B) = {corr:+.3f}")

    # Equal-risk-contribution weights from pre-2018 vols only
    CUT = pd.Timestamp("2018-01-01")
    vA = rA.loc[first_valid:CUT].std() * np.sqrt(252)
    vB = rB.loc[first_valid:CUT].std() * np.sqrt(252)
    wA = (1 / vA) / (1 / vA + 1 / vB)
    wB = 1 - wA
    print(f"\nERC weights (pre-2018 vols): A={wA:.3f} B={wB:.3f}")

    port = wA * rA_v + wB * rB_v

    for r, lbl in [(port, "NOVA11 ERC ensemble")]:
        s = stats(r, lbl)
        print(f"\n{s['label']:28s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

        for p, tag in [(r.loc[:CUT], "IS <2018"), (r.loc[CUT:], "OOS >=2018")]:
            ss = stats(p, tag)
            print(f"  {ss['label']:26s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
                  f"Vol={ss['vol']:>5.2f}%  MDD={ss['mdd']:>7.2f}%")

    # Annual
    ann = port.groupby(port.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual:")
    print(ann.to_string())

    # Save
    out = pd.DataFrame({"NOVA11": port, "sleeveA": rA_v, "sleeveB": rB_v,
                       "posA": posA.loc[first_valid:], "posB": posB.loc[first_valid:]})
    out.to_csv("/home/user/bonds/data/results/nova11_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova11_returns.csv")


if __name__ == "__main__":
    main()
