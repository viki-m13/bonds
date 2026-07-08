"""Cross-sectional strategies over individual Treasury CUSIPs.

Each strategy produces target weights (date, cusip, weight) on a weekly
rebalance grid, using only information available at the rebalance close.

Tradeable universe at each rebalance:
  - nominal coupon notes/bonds (bills excluded from long-short legs),
  - remaining maturity >= 1 year (no near-maturity noise),
  - a valid price and signal on that date.

Portfolio construction for long-short signals:
  - rank signal cross-sectionally; long the top `frac`, short the bottom `frac`
  - within each side, weight proportional to 1/duration (equal duration
    contribution per name), then scale the short side so the portfolio's net
    duration-dollars are zero (duration-neutral: P&L reflects relative
    mispricing, not the direction of rates)
  - each side scaled to gross 1.0 => total gross exposure ~2x NAV
  - per-name weight cap, applied before normalization
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def weekly_rebalance_dates(dates: pd.DatetimeIndex, anchor: str = "W-WED") -> pd.DatetimeIndex:
    """Last available trading date in each anchor-week."""
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby(s.index.to_period(anchor)).max().values)


def zscore_ts(wide: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    mp = min_periods or max(20, window // 3)
    mu = wide.rolling(window, min_periods=mp).mean()
    sd = wide.rolling(window, min_periods=mp).std()
    return (wide - mu) / sd


def build_ls_weights(
    sig: pd.DataFrame,
    dur: pd.DataFrame,
    tradeable: pd.DataFrame,
    reb_dates: pd.DatetimeIndex,
    frac: float = 0.2,
    name_cap: float = 0.10,
) -> pd.DataFrame:
    """Long-short duration-neutral weights from a wide signal matrix.

    sig/dur/tradeable: wide (date x cusip). Higher signal => long.
    """
    out = []
    for t in reb_dates:
        if t not in sig.index:
            continue
        s = sig.loc[t]
        ok = tradeable.loc[t] if t in tradeable.index else None
        if ok is None:
            continue
        s = s[ok.reindex(s.index).fillna(False)].dropna()
        if len(s) < 20:
            continue
        d = dur.loc[t].reindex(s.index)
        n = max(int(len(s) * frac), 5)
        longs = s.nlargest(n).index
        shorts = s.nsmallest(n).index

        def side(names, sign):
            dd = d.reindex(names).clip(lower=0.5)
            w = (1.0 / dd)
            w = np.minimum(w / w.sum(), name_cap)
            w = w / w.sum()
            return sign * w

        wl, ws = side(longs, +1.0), side(shorts, -1.0)
        # duration-neutralize: scale short side so sum(w*dur) nets to zero
        dl = float((wl * d.reindex(wl.index)).sum())
        ds = float(-(ws * d.reindex(ws.index)).sum())
        if ds > 0:
            ws = ws * (dl / ds)
        w = pd.concat([wl, ws])
        w = w.groupby(w.index).sum()
        # rescale to gross 2.0
        gross = w.abs().sum()
        if gross > 0:
            w = w * (2.0 / gross)
        out.append(pd.DataFrame({"date": t, "cusip": w.index, "weight": w.values}))
    return pd.concat(out, ignore_index=True)


class SignalSet:
    """Precomputes wide matrices once; individual strategies index into them."""

    def __init__(self, panel: pd.DataFrame, fits: pd.DataFrame, params: pd.DataFrame):
        p = panel.merge(fits, on=["date", "cusip"], how="left")
        p["resid"] = p["ytm"] - p["fitted"]
        self.panel = p
        self.params = params

        coupon = p[p["sec_type"].isin(["MARKET BASED NOTE", "MARKET BASED BOND"])]
        piv = lambda v: coupon.pivot_table(index="date", columns="cusip", values=v, aggfunc="first")
        self.resid = piv("resid")
        self.dur = piv("mod_dur")
        self.ytm = piv("ytm")
        self.tsy = piv("tsy_years")
        self.fitted = piv("fitted")
        self.ret = piv("ret")
        self.spread = piv("spread_pct")
        self.tradeable = (
            self.tsy.ge(1.0) & self.ytm.notna() & self.resid.notna() & self.dur.notna()
        )

    # ---- signals (wide matrices, higher = more attractive to be long) ----

    def sig_value(self, z_window: int = 60) -> pd.DataFrame:
        """Cheapness vs own history: positive when bond is cheap vs its
        typical relationship to the fitted curve. Mean-reversion trade."""
        return zscore_ts(self.resid, z_window)

    def sig_value_raw(self) -> pd.DataFrame:
        """Raw residual: cheap = positive. Carries persistent liquidity
        premia (off-the-runs always look cheap) - kept as a baseline."""
        return self.resid.copy()

    def sig_carry(self, roll_h: float = 1.0) -> pd.DataFrame:
        """(Carry + rolldown) per unit duration, from the fitted curve.

        annualized expected excess return if the curve is unchanged:
          (ytm - y_short) + dur * (y_fit(tsy) - y_fit(tsy - roll_h))
        divided by duration to remove the pure duration bet.
        """
        from curve import nss_basis
        out = pd.DataFrame(index=self.tsy.index, columns=self.tsy.columns, dtype=float)
        pars = self.params
        y_short_all = {}
        for t in self.tsy.index:
            if t not in pars.index:
                continue
            row = pars.loc[t]
            tsy = self.tsy.loc[t].values.astype(float)
            valid = np.isfinite(tsy)
            tv = tsy[valid]
            X1 = nss_basis(tv, row["tau1"], row["tau2"])
            X0 = nss_basis(np.maximum(tv - roll_h, 0.02), row["tau1"], row["tau2"])
            b = row[["b0", "b1", "b2", "b3"]].values.astype(float)
            roll = (X1 - X0) @ b  # y(tsy) - y(tsy - h): positive when upward-sloping
            y_short = float(nss_basis(np.array([0.25]), row["tau1"], row["tau2"]) @ b)
            vals = np.full(len(tsy), np.nan)
            ytm = self.ytm.loc[t].values.astype(float)[valid]
            dur = self.dur.loc[t].values.astype(float)[valid]
            vals[valid] = ((ytm - y_short) + dur * roll) / np.clip(dur, 0.5, None)
            out.loc[t] = vals
        return out

    def sig_momentum(self, lookback: int = 60) -> pd.DataFrame:
        """Idiosyncratic momentum: negative change of the residual over the
        lookback (residual falling = bond richening = positive momentum)."""
        return -(self.resid - self.resid.shift(lookback))
