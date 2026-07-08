"""Bond math for fixed-coupon nominal Treasuries (and bills).

Conventions used (street convention for USTs):
- Semiannual coupons, actual/actual day count within the coupon period.
- Coupon schedule generated backwards from maturity in 6-month steps.
- Clean price quoted per $100 par; accrued interest added to get dirty price.
- Yield = semiannually-compounded yield to maturity solved from the clean
  price with Newton iterations (vectorized across many bonds at once).

Bills (zero coupon): accrued = 0; yield is bond-equivalent computed from the
same machinery with zero coupon (single cashflow of 100 at maturity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_COUPONS = 61  # 30y bond has 60 coupons remaining at issue


def coupon_schedule(maturity: pd.Timestamp, n: int = MAX_COUPONS) -> pd.DatetimeIndex:
    """Coupon dates: maturity, maturity-6m, ... (descending), length n."""
    return pd.DatetimeIndex([maturity - pd.DateOffset(months=6 * k) for k in range(n)])


def accrued_and_times(
    dates: np.ndarray, maturities: np.ndarray, rates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized accrued interest and cashflow times for many (date, bond) rows.

    Parameters are equal-length arrays: settlement dates (datetime64[D]),
    maturity dates (datetime64[D]), annual coupon rates in percent.

    Returns (accrued, times, cfs):
      accrued : accrued interest per $100 par
      times   : (len, MAX_COUPONS) year-fractions (in half-year units!) to each
                remaining coupon; NaN-padded beyond maturity
      cfs     : (len, MAX_COUPONS) cashflow amounts per $100 par
    """
    n = len(dates)
    dates_m = dates.astype("datetime64[M]")
    mat_m = maturities.astype("datetime64[M]")

    # months from settlement month to maturity month
    months_to_mat = (mat_m - dates_m).astype(int)

    # k-th previous coupon month offset from maturity: maturity - 6k months.
    # Find kf = number of full 6m periods between next coupon and maturity.
    # Next coupon date = maturity - 6*floor((months_to_mat - adj)/6) months, handled
    # exactly below by generating both bracketing coupon dates per row.
    k_next = np.floor_divide(months_to_mat, 6)  # candidate periods remaining after next coupon
    # Generate candidate coupon dates maturity - 6*k months for k = k_next-1..k_next+1
    day = (maturities - maturities.astype("datetime64[M]")).astype(int)  # day-of-month index 0-based

    def coupon_date(k: np.ndarray) -> np.ndarray:
        m = mat_m - (6 * k).astype("timedelta64[M]")
        # clamp day to month length
        next_m = m + np.timedelta64(1, "M")
        mlen = (next_m.astype("datetime64[D]") - m.astype("datetime64[D]")).astype(int)
        d = np.minimum(day, mlen - 1)
        return m.astype("datetime64[D]") + d.astype("timedelta64[D]")

    # next coupon strictly after settlement; previous coupon on/before settlement
    k_hi = k_next.copy()
    cd = coupon_date(k_hi)
    # if candidate next coupon <= settlement, step closer to maturity
    too_early = cd <= dates
    k_hi = np.where(too_early, k_hi - 1, k_hi)
    cd_next = coupon_date(k_hi)
    # if still a candidate further from maturity is > settlement, use it
    cd_try = coupon_date(k_hi + 1)
    use_try = cd_try > dates
    k_hi = np.where(use_try, k_hi + 1, k_hi)
    cd_next = np.where(use_try, cd_try, cd_next)
    cd_prev = coupon_date(k_hi + 1)

    period_days = (cd_next - cd_prev).astype(int).astype(float)
    accr_days = (dates - cd_prev).astype(int).astype(float)
    frac = np.clip(accr_days / np.maximum(period_days, 1.0), 0.0, 1.0)
    accrued = rates / 2.0 * frac

    # remaining coupons: k = k_hi down to 0 (dates maturity-6k)
    ks = np.arange(MAX_COUPONS)
    n_remaining = k_hi + 1  # number of coupons left
    # time to j-th cashflow in half-year units: (1 - frac) + j for j=0..n_remaining-1
    j = ks[None, :]
    times = (1.0 - frac)[:, None] + j
    valid = j < n_remaining[:, None]
    times = np.where(valid, times, np.nan)

    cfs = np.where(valid, (rates / 2.0)[:, None], np.nan)
    # add principal at maturity (last valid cashflow)
    last = (j == (n_remaining - 1)[:, None]) & valid
    cfs = np.where(last, cfs + 100.0, cfs)

    return accrued, times, cfs


def price_from_yield(y: np.ndarray, times: np.ndarray, cfs: np.ndarray) -> np.ndarray:
    """Dirty price per 100 par from semiannual yield (decimal, e.g. 0.04)."""
    disc = (1.0 + y[:, None] / 2.0) ** (-times)
    return np.nansum(cfs * disc, axis=1)


def solve_ytm(
    dirty: np.ndarray, times: np.ndarray, cfs: np.ndarray, iters: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Newton solve for semiannual YTM; returns (ytm, modified_duration_years)."""
    y = np.full(len(dirty), 0.03)
    for _ in range(iters):
        disc = (1.0 + y[:, None] / 2.0) ** (-times)
        pv = np.nansum(cfs * disc, axis=1)
        dpv = np.nansum(cfs * disc * (-times / 2.0) / (1.0 + y[:, None] / 2.0), axis=1)
        step = (pv - dirty) / np.where(np.abs(dpv) < 1e-12, np.nan, dpv)
        step = np.clip(step, -0.05, 0.05)
        y = y - step
        y = np.clip(y, -0.05, 0.60)
    disc = (1.0 + y[:, None] / 2.0) ** (-times)
    pv = np.nansum(cfs * disc, axis=1)
    dpv = np.nansum(cfs * disc * (-times / 2.0) / (1.0 + y[:, None] / 2.0), axis=1)
    mod_dur = -dpv / pv  # in half-year-rate terms; already dP/dy per unit y
    # dpv is dP/dy with y annual (semiannual compounding), times in half-years:
    # d/dy (1+y/2)^(-t) = -t/2 (1+y/2)^(-t-1)  -> handled above; mod_dur in years.
    return y, mod_dur
