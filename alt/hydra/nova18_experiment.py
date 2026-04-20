"""NOVA18 — Multi-asset time-series momentum (Moskowitz-Ooi-Pedersen 2012).

The ORIGINAL published time-series momentum paper with demonstrated SR
1.4+ across 58 futures contracts 1985-2009. We proxy on 12 liquid ETFs
2016-2026.

Fixed a priori (no tuning):
  Universe: SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, SLV, DBC, VNQ, HYG.
  Signal per ETF: 12-month excess return = r_12m - r_BIL_12m.
  Position per ETF: long (+1) if signal > 0, short (-1) if signal < 0.
  Weight: 1/N across ETFs (equal weight, not vol-scaled — NO continuous
    vol scaling per user constraint). All ETFs weighted equally regardless
    of idiosyncratic vol.
  Rebalance: monthly. 1-bar execution lag. 10 bps TC per rebalance switch.

This is DISCRETE sign-based position — not continuous vol scaling.
Published paper's raw signal has long history of positive OOS SR, so we
expect something in 0.6-1.2 range (less than published because our
universe is smaller and period shorter)."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats


UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD",
            "SLV", "DBC", "VNQ", "HYG"]
LOOKBACK = 252
TC_BPS = 10.0


def monthly_first(index):
    out = pd.Series(False, index=index)
    out.iloc[0] = True
    for i in range(1, len(index)):
        if index[i].month != index[i - 1].month:
            out.iloc[i] = True
    return out


def main():
    px = {}
    for t in UNIVERSE + ["BIL"]:
        s = load_etf(t)
        if s is None:
            print(f"MISSING {t}")
            continue
        px[t] = s
    px = pd.DataFrame(px).sort_index().ffill()
    dates = px.index
    rets = px.pct_change().fillna(0)

    first = monthly_first(dates)
    # 12m cumulative return lagged 1 day (uses info up to t-1)
    cum12 = (1 + rets).rolling(LOOKBACK).apply(lambda x: x.prod() - 1,
                                                raw=True)
    cum12_lag = cum12.shift(1)

    # Position per ETF: sign(ret12 - ret12_BIL)
    sig = cum12_lag[UNIVERSE].sub(cum12_lag["BIL"], axis=0)
    pos_monthly = pd.DataFrame(0.0, index=dates, columns=UNIVERSE)
    last = pd.Series(0.0, index=UNIVERSE)
    for i, d in enumerate(dates):
        if first.iloc[i] and not sig.loc[d].isna().any():
            last = np.sign(sig.loc[d])
        pos_monthly.loc[d] = last.values

    # Equal weight across UNIVERSE: each position is 1/N
    w = pos_monthly / len(UNIVERSE)
    w_eff = w.shift(1).fillna(0)

    port_gross = (w_eff * rets[UNIVERSE]).sum(axis=1)

    # Long + short: residual cash earns BIL
    exposure = w_eff.sum(axis=1)
    cash_w = 1 - exposure.abs().clip(upper=1)   # simplistic: residual to BIL
    # For a long-short portfolio we don't need BIL carry — the notional is
    # assumed 100% gross. Skip BIL to keep it clean (TSMOM papers do this).

    # TC on turnover (L1 change in weights)
    turnover = (w_eff - w_eff.shift(1)).abs().sum(axis=1).fillna(0)
    tc = turnover * (TC_BPS / 1e4)
    port = port_gross - tc

    warm = pd.Timestamp("2017-02-01")
    p_v = port.loc[warm:]

    print(f"NOVA18 — TSMOM on {len(UNIVERSE)} ETFs, lookback 12mo")
    print(f"Avg gross exposure (|w|sum): {w_eff.abs().sum(axis=1).loc[warm:].mean():.2f}")
    print(f"Avg # long: {(w_eff > 0).sum(axis=1).loc[warm:].mean():.1f}, "
          f"Avg # short: {(w_eff < 0).sum(axis=1).loc[warm:].mean():.1f}")

    s = stats(p_v, "NOVA18 TSMOM (12mo)")
    print(f"\n{s['label']:30s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    CUT = pd.Timestamp("2022-01-01")
    for p, tag in [(p_v.loc[:CUT], "IS <2022"), (p_v.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:28s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"Vol={ss['vol']:>5.2f}%  MDD={ss['mdd']:>7.2f}%")

    # Long-only variant (drop shorts — sometimes cleaner OOS)
    w_lo = w_eff.clip(lower=0)
    port_lo = (w_lo * rets[UNIVERSE]).sum(axis=1)
    turnover_lo = (w_lo - w_lo.shift(1)).abs().sum(axis=1).fillna(0)
    port_lo = port_lo - turnover_lo * (TC_BPS / 1e4)
    p_lo = port_lo.loc[warm:]
    s = stats(p_lo, "NOVA18-LO long-only TSMOM")
    print(f"\n{s['label']:30s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    for p, tag in [(p_lo.loc[:CUT], "IS <2022"), (p_lo.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:28s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"MDD={ss['mdd']:>7.2f}%")

    # Annual (long-short)
    ann = p_v.groupby(p_v.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual (long-short):")
    print(ann.to_string())

    out = pd.DataFrame({"NOVA18": port, "NOVA18_LO": port_lo})
    out.to_csv("/home/user/bonds/data/results/nova18_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova18_returns.csv")


if __name__ == "__main__":
    main()
