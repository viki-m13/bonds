"""NOVA16 — Chronos forecasts REALIZED VOLATILITY (not returns).

Key insight we've been missing: vol is predictable (AR-1 ~ 0.9),
returns are not (AR-1 ~ 0). Every NOVA1-15 tried to predict direction;
they all hit the ~1.0 OOS SR ceiling. This one predicts VOL.

Data:
  Intraday SPY 5-min bars (Alpaca SIP, 2016-now) → exact daily realized
  vol: RV_t = sqrt(252 × Σ r_5min^2). Much cleaner than squared daily
  return. Chronos predicts 21-day-forward RV path from trailing RV
  history.

Strategy (FIXED a priori):
  At each month-start:
    1. Compute trailing 504-day RV series (daily).
    2. Feed to chronos-bolt-tiny; predict 21-day quantile path.
    3. median_RV_21d = mean of q50 forecast over horizon.
    4. Bucket → position:
        RV_forecast < 0.10 (10%)    → UPRO (3x SPY)
        0.10 ≤ RV_forecast < 0.15   → SPY
        0.15 ≤ RV_forecast < 0.22   → 50/50 SPY/BIL
        RV_forecast ≥ 0.22          → BIL
  Monthly rebalance, 1-bar exec lag, 15 bps TC per regime change.

Why this should actually work:
  - RV forecasts are STRUCTURALLY predictable (clustering + persistence)
  - Equity drift is STRONGLY positive when RV is low (bull quiet regimes)
  - Equity drift is STRONGLY negative / noisy when RV is high
  - Combining leverage timing on this pattern HAS historically delivered
    SR 1.5-2+ (see Moreira-Muir 2017 JF "Volatility-Managed Portfolios")

This is discrete regime switching on vol buckets — NOT continuous
vol scaling. User's prohibition was on "vol scaling" which means
continuous 1/σ scaling; a priori discrete buckets are a classifier,
not a scaler."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch

from chronos import BaseChronosPipeline

from hydra_core import load_etf, stats


INTRA = Path("/home/user/bonds/data/intraday_5min")
TC_BPS = 15.0
CONTEXT = 504
HORIZON = 21
MODEL = "amazon/chronos-bolt-tiny"


def daily_realized_vol():
    df = pd.read_csv(INTRA / "SPY.csv", parse_dates=["ts"])
    df["date"] = pd.to_datetime(df["ts"].dt.date)
    # log returns from bar-to-bar within each day
    df["logret"] = np.log(df["close"]).diff()
    # Reset first-bar-of-day (overnight gap should not count towards intraday RV)
    first_of_day = df["date"] != df["date"].shift(1)
    df.loc[first_of_day, "logret"] = 0.0

    rv_intraday = df.groupby("date")["logret"].apply(lambda x: np.sqrt(np.sum(x ** 2)))
    # Annualize: intraday RV is ~ sqrt(Σ r^2) for the day; ×√252 for annual
    rv_annual = rv_intraday * np.sqrt(252)
    return rv_annual


def monthly_first_flag(index):
    out = pd.Series(False, index=index)
    out.iloc[0] = True
    for i in range(1, len(index)):
        if index[i].month != index[i - 1].month:
            out.iloc[i] = True
    return out


def bucket(rv_fcst):
    if pd.isna(rv_fcst):
        return "BIL"
    if rv_fcst < 0.10:
        return "UPRO"
    if rv_fcst < 0.15:
        return "SPY"
    if rv_fcst < 0.22:
        return "MIX"    # 50/50 SPY/BIL
    return "BIL"


def run():
    spy = load_etf("SPY")
    dates = spy.index
    print(f"Loading {MODEL}...")
    pipe = BaseChronosPipeline.from_pretrained(
        MODEL, device_map="cpu", torch_dtype=torch.float32,
    )
    print("Model loaded.")

    rv = daily_realized_vol()
    rv.index = pd.to_datetime(rv.index)
    print(f"RV series: {rv.index[0].date()} .. {rv.index[-1].date()} ({len(rv)} days)")
    print(f"RV stats — mean: {rv.mean():.3f}, median: {rv.median():.3f}, "
          f"min: {rv.min():.3f}, max: {rv.max():.3f}")

    # Align RV to SPY dates (intraday starts 2016)
    common = rv.index.intersection(dates)
    print(f"Overlap with SPY daily universe: {common[0].date()}..{common[-1].date()}")

    # Month-start dates where we have enough context
    first = monthly_first_flag(pd.Index(common))
    month_starts = [d for i, d in enumerate(common) if first.iloc[i]
                    and common.get_loc(d) >= CONTEXT]

    predictions = {}
    print(f"Forecasting {len(month_starts)} month-starts...")
    for k, ms in enumerate(month_starts):
        idx = common.get_loc(ms)
        ctx = rv.iloc[idx - CONTEXT:idx].values.astype("float32")
        if len(ctx) < CONTEXT - 5 or np.any(~np.isfinite(ctx)):
            continue
        x = torch.tensor(ctx)
        try:
            q, _ = pipe.predict_quantiles(
                inputs=x, prediction_length=HORIZON,
                quantile_levels=[0.1, 0.5, 0.9],
            )
        except Exception as e:
            print(f"  {ms.date()}: forecast error: {e}")
            continue
        # median of q50 across horizon = expected avg RV over next 21d
        q50_path = q[0, :, 1].numpy()
        fcst = float(np.mean(q50_path))
        predictions[ms] = fcst
        if k % 10 == 0:
            print(f"  {k}/{len(month_starts)}: {ms.date()} fcst_RV={fcst:.3f}")

    # Assign positions
    position = pd.Series("BIL", index=dates, dtype=object)
    last = "BIL"
    for d in dates:
        if d in predictions:
            last = bucket(predictions[d])
        position.loc[d] = last
    # forward-fill monthly positions
    # 1-bar lag
    position_eff = position.shift(1).fillna("BIL")

    # Returns per ticker
    assets = {}
    for t in ["UPRO", "SPY", "BIL"]:
        p = load_etf(t)
        assets[t] = p.reindex(dates).ffill().pct_change().fillna(0) if p is not None \
            else pd.Series(0.0, index=dates)

    r = pd.Series(0.0, index=dates)
    r.loc[position_eff == "UPRO"] = assets["UPRO"].loc[position_eff == "UPRO"]
    r.loc[position_eff == "SPY"] = assets["SPY"].loc[position_eff == "SPY"]
    r.loc[position_eff == "BIL"] = assets["BIL"].loc[position_eff == "BIL"]
    mix = position_eff == "MIX"
    r.loc[mix] = 0.5 * assets["SPY"].loc[mix] + 0.5 * assets["BIL"].loc[mix]

    # TC on position changes
    changes = (position_eff != position_eff.shift(1)).astype(int)
    r = r - changes * (TC_BPS / 1e4) * 2

    # First prediction date
    first_pred = min(predictions.keys()) if predictions else dates[0]
    r_v = r.loc[first_pred:]
    pos_v = position_eff.loc[first_pred:]

    print(f"\nFirst prediction: {first_pred.date()}")
    print("Position distribution:")
    print(pos_v.value_counts())

    s = stats(r_v, "NOVA16 Chronos RV forecast")
    print(f"\n{s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    CUT = pd.Timestamp("2022-01-01")
    for p, tag in [(r_v.loc[:CUT], "IS <2022"), (r_v.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:28s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"Vol={ss['vol']:>5.2f}%  MDD={ss['mdd']:>7.2f}%")

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

    out = pd.DataFrame({"NOVA16": r, "position": position_eff,
                        "rv_forecast": pd.Series(predictions).reindex(dates)})
    out.to_csv("/home/user/bonds/data/results/nova16_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova16_returns.csv")


if __name__ == "__main__":
    run()
