"""NOVA10 — Chronos foundation-model forecast-driven strategy.

Amazon's Chronos is a time-series foundation model (transformer
pretrained on a huge corpus). Used here zero-shot to forecast SPY
over a 21-day horizon, every month. Read the 10/50/90 percentile
quantiles, map to concentrated positions.

Design (fixed, no tuning):
  - Model: chronos-bolt-tiny (fast enough for 250+ monthly predictions)
  - Input: last 504 trading days of SPY prices
  - Output: 21-day forecast quantiles (0.1, 0.5, 0.9)
  - Compute: median_return_21d = q50[-1]/current_price - 1
             conf_width = (q90[-1] - q10[-1]) / current_price

  Position buckets (fixed a priori):
    med > 0.02 AND conf_width < 0.12 → UPRO  (high-conviction 3x)
    med > 0.005                       → SPY
    med > -0.005                       → BIL
    med > -0.02                        → SH
    else                               → SDS

  Rebalance: monthly, 1-bar exec lag, 15 bps TC per switch.

Walk-forward: at every month-start, give Chronos only past data.
Pretraining cutoff of Chronos is unclear so OOS can't be strictly
proven; report full-sample plus IS (≤2018) / OOS (>2018) split for
transparency."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch

from chronos import BaseChronosPipeline

from hydra_core import load_etf, stats


TC_BPS = 15.0
CONTEXT = 504          # ~2y of daily data
HORIZON = 21
MODEL = "amazon/chronos-bolt-tiny"


def monthly_first_flag(index):
    out = pd.Series(False, index=index)
    out.iloc[0] = True
    for i in range(1, len(index)):
        if index[i].month != index[i - 1].month:
            out.iloc[i] = True
    return out


def bucket(med_ret, conf_width):
    if pd.isna(med_ret) or pd.isna(conf_width):
        return "BIL"
    if med_ret > 0.02 and conf_width < 0.12:
        return "UPRO"
    if med_ret > 0.005:
        return "SPY"
    if med_ret > -0.005:
        return "BIL"
    if med_ret > -0.02:
        return "SH"
    return "SDS"


def run():
    spy = load_etf("SPY")
    dates = spy.index
    print(f"Universe: {dates[0].date()} .. {dates[-1].date()}")
    print(f"Loading {MODEL}...")

    pipe = BaseChronosPipeline.from_pretrained(
        MODEL, device_map="cpu", torch_dtype=torch.float32,
    )
    print("Model loaded.")

    # Build month-start list
    first = monthly_first_flag(pd.Index(dates))
    month_starts = [d for d in dates if first.loc[d] and dates.get_loc(d) >= CONTEXT]

    # Forecast loop
    predictions = {}
    print(f"Forecasting {len(month_starts)} month-starts...")
    for k, ms in enumerate(month_starts):
        idx = dates.get_loc(ms)
        ctx_start = idx - CONTEXT
        ctx = spy.iloc[ctx_start:idx].values.astype("float32")
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
        cur = float(ctx[-1])
        q10 = float(q[0, -1, 0])
        q50 = float(q[0, -1, 1])
        q90 = float(q[0, -1, 2])
        med = q50 / cur - 1
        width = (q90 - q10) / cur
        predictions[ms] = (med, width)
        if k % 30 == 0:
            print(f"  {k}/{len(month_starts)}: {ms.date()} med={med:+.3f} w={width:.3f}")

    # Assign positions
    position = pd.Series("BIL", index=dates, dtype=object)
    last = "BIL"
    for i, d in enumerate(dates):
        if first.loc[d] and d in predictions:
            med, w = predictions[d]
            last = bucket(med, w)
        position.iloc[i] = last

    # 1-bar lag
    position_eff = position.shift(1).fillna("BIL")

    # Returns
    assets = {}
    for t in ["UPRO", "SPY", "BIL", "SH", "SDS"]:
        p = load_etf(t)
        if p is None:
            assets[t] = pd.Series(0.0, index=dates)
        else:
            assets[t] = p.reindex(dates).ffill().pct_change().fillna(0)

    r = pd.Series(0.0, index=dates)
    for t in ["UPRO", "SPY", "BIL", "SH", "SDS"]:
        mask = position_eff == t
        r.loc[mask] = assets[t].loc[mask]
    changes = (position_eff != position_eff.shift(1)).astype(int)
    r = r - changes * (TC_BPS / 1e4) * 2

    # Strictly evaluate from first date we have a prediction
    first_pred = min(predictions.keys()) if predictions else dates[0]
    r_v = r.loc[first_pred:]
    pos_v = position_eff.loc[first_pred:]

    print(f"\nFirst prediction: {first_pred.date()}")
    print("Position distribution:")
    print(pos_v.value_counts())

    s = stats(r_v, "NOVA10 Chronos")
    print(f"\n{s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    IS = pd.Timestamp("2018-01-01")
    for p, lbl in [(r_v.loc[:IS], "IS ≤2018"), (r_v.loc[IS:], "OOS >2018")]:
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

    # Save
    out_df = pd.DataFrame({
        "NOVA10": r,
        "position": position_eff,
        "pred_median": pd.Series({d: v[0] for d, v in predictions.items()}).reindex(dates),
        "pred_width": pd.Series({d: v[1] for d, v in predictions.items()}).reindex(dates),
    })
    out_df.to_csv("/home/user/bonds/data/results/nova10_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova10_returns.csv")


if __name__ == "__main__":
    run()
