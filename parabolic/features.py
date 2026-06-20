"""Pre-parabolic feature library (PIT S&P 500 panel, close/open/volume only).

Every feature at row d uses information through the CLOSE of day d only
(trailing windows; no shift(-k), no full-sample statistics). Cross-sectional
ranking is deferred to the consumer (event study / strategy) and is always done
within a single row d, so it is causal too.

The library encodes the union of what the research surfaced (see
research/literature.md): the academic mean-positive breakout anchors
(George-Hwang 52-week-high nearness, Gervais-Kaniel-Mingelgrin volume shock,
residual/continuous-information momentum), the FinTwit practitioner setups
(Minervini Trend Template + VCP contraction, Qullamaggie prior-move + ADR +
flag, Weinstein Stage-2 / Mansfield RS, Darvas/HTF breakouts, episodic-pivot
gaps), and the lottery/tail selectors (MAX, realised vol, beta) that the
literature warns are NEGATIVE mean predictors but select the fat right tail.

High/Low are not in the shipped panel, so range/ADR/ATR are approximated from
close-to-close moves (the canonical EDA already found high/low range features
dead, so nothing of value is lost). All such proxies are named *_cc.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "dca"))
import data as dca_data  # noqa: E402


# --------------------------------------------------------------------- helpers
def _rolling_max(df, w):
    return df.rolling(w, min_periods=max(20, w // 3)).max()


def _rolling_min(df, w):
    return df.rolling(w, min_periods=max(20, w // 3)).min()


def days_since_high(close, win=252):
    """Trading days since the most recent `win`-day-high close (0 = new high)."""
    rmax = close.rolling(win).max()
    is_hi = (close >= rmax * (1 - 1e-9)) & rmax.notna()
    n = len(close)
    pos = np.arange(n, dtype=float)[:, None]
    last = pd.DataFrame(np.where(is_hi.values, pos, np.nan),
                        index=close.index, columns=close.columns).ffill()
    age = pd.DataFrame(np.broadcast_to(pos, close.shape),
                       index=close.index, columns=close.columns) - last
    return age.where(rmax.notna())


def _rolling_beta_corr(R, s, w=120, min_obs=100):
    """Trailing beta and corr of each column of returns R vs market returns s."""
    sv = s.values[:, None]
    Rm = R.values
    msk = ~np.isnan(Rm) & ~np.isnan(sv)
    Rm0 = np.where(msk, Rm, 0.0)
    sm0 = np.where(msk, sv, 0.0)
    k = pd.DataFrame(msk.astype(float)).rolling(w).sum().values
    Sx = pd.DataFrame(Rm0).rolling(w).sum().values
    Sy = pd.DataFrame(sm0).rolling(w).sum().values
    Sxy = pd.DataFrame(Rm0 * sm0).rolling(w).sum().values
    Sxx = pd.DataFrame(Rm0 * Rm0).rolling(w).sum().values
    Syy = pd.DataFrame(sm0 * sm0).rolling(w).sum().values
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = Sxy / k - (Sx / k) * (Sy / k)
        vx = Sxx / k - (Sx / k) ** 2
        vy = Syy / k - (Sy / k) ** 2
        beta = np.where(vy > 0, cov / vy, np.nan)
        corr = np.where((vx > 0) & (vy > 0), cov / np.sqrt(vx * vy), np.nan)
    bad = k < min_obs
    beta[bad] = np.nan
    corr[bad] = np.nan
    idx, cols = R.index, R.columns
    return (pd.DataFrame(beta, index=idx, columns=cols),
            pd.DataFrame(corr, index=idx, columns=cols))


# --------------------------------------------------------------------- features
def build_features(P: dict | None = None) -> dict:
    """Return dict name -> DataFrame (dates x tickers), all causal."""
    if P is None:
        P = dca_data.build_panel()
    close, vol = P["close"], P["volume"]
    open_ = P["open"]
    spy = dca_data.load_benchmark("SPY")["Close"].reindex(close.index).ffill()

    R = close.pct_change()
    f = {}

    # ---- horizon returns / momentum (Qullamaggie prior move, Jegadeesh-Titman)
    f["ret_1m"] = close / close.shift(21) - 1
    f["ret_3m"] = close / close.shift(63) - 1
    f["ret_6m"] = close / close.shift(126) - 1
    f["ret_12m"] = close / close.shift(252) - 1
    f["mom_12_1"] = close.shift(21) / close.shift(252) - 1

    # ---- 52-week-high nearness (George-Hwang) and cycle position
    rmax = _rolling_max(close, 252)
    rmin = _rolling_min(close, 252)
    f["nearness_52wh"] = close / rmax            # <=1, near 1 = at highs
    f["dist_52w_low"] = close / rmin - 1         # >=0, "already turned"
    f["dist_52w_high"] = close / rmax - 1        # <=0
    f["age_52w_high"] = days_since_high(close)
    f["at_new_high"] = (close >= rmax * (1 - 1e-9)).astype(float)

    # ---- moving averages (Minervini Trend Template / Weinstein Stage 2)
    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    f["px_vs_sma50"] = close / sma50 - 1
    f["px_vs_sma200"] = close / sma200 - 1
    f["sma200_slope"] = sma200 / sma200.shift(21) - 1     # 1-month MA slope
    # Minervini 8-point Trend Template (Stage-2 gate), 1.25x-above-low version
    tt = (
        (close > sma150) & (close > sma200) &
        (sma150 > sma200) &
        (sma200 > sma200.shift(21)) &
        (sma50 > sma150) & (sma50 > sma200) &
        (close > sma50) &
        (close >= 1.25 * rmin) &
        (close >= 0.75 * rmax)
    )
    f["trend_template"] = tt.astype(float).where(sma200.notna())

    # ---- IBD-style relative strength (weighted multi-quarter return).
    # Cross-sectional percentile is taken by the consumer; store the raw score.
    f["rs_ibd_raw"] = (0.4 * (close / close.shift(63) - 1)
                       + 0.2 * (close / close.shift(126) - 1)
                       + 0.2 * (close / close.shift(189) - 1)
                       + 0.2 * (close / close.shift(252) - 1))

    # ---- RS line vs SPY: ratio, its new-high flag (IBD blue dot), Mansfield RS
    ratio = close.div(spy, axis=0)
    ratio_max = _rolling_max(ratio, 252)
    f["rs_line"] = ratio / ratio_max                       # near 1 = RS at highs
    f["rs_line_new_high"] = (
        (ratio >= ratio_max * (1 - 1e-9)) &
        (close < rmax.shift(1) * (1 - 1e-9))               # price NOT yet new high
    ).astype(float).where(ratio_max.notna())
    rp = ratio * 100.0
    f["mansfield_rs"] = (rp / rp.rolling(200).mean() - 1) * 100

    # ---- volatility (tail selectors: NEGATIVE mean, fat right tail)
    sd20 = R.rolling(20).std()
    sd60 = R.rolling(60).std()
    f["vol_20d"] = sd20 * np.sqrt(252)
    f["max_dret_21"] = R.rolling(21).max()                 # MAX / lottery (Bali)
    f["adr_cc"] = R.abs().rolling(20).mean() * 100         # ADR%% proxy (close-only)

    # ---- VCP / squeeze contraction (volatility-only timing conditioner)
    f["vcp_contraction"] = sd20 / sd60                     # <1 = contracting
    # Bollinger BandWidth percentile over 126d (low = squeeze)
    bbw = (4.0 * sd20)                                      # width / mean, mean~1 in ret space
    f["bbw_pctile"] = bbw.rolling(126).rank(pct=True)
    # consolidation tightness near the high: 20d close range as %% of price
    f["tightness_20"] = (_rolling_max(close, 20) - _rolling_min(close, 20)) / close

    # ---- volume (Gervais-Kaniel-Mingelgrin high-volume premium; accumulation)
    f["vol_shock"] = vol.rolling(5).mean() / vol.rolling(60).mean()
    f["vol_dryup"] = vol.rolling(10).mean() / vol.rolling(60).mean()   # low = dry
    up = (R > 0)
    hv = vol > vol.shift(1)
    acc = (up & hv).astype(float) - (~up & hv & (R <= -0.002)).astype(float)
    f["accum_25"] = acc.rolling(25).sum()

    # ---- episodic pivot / gap-and-go (catalyst proxy)
    gap = open_ / close.shift(1) - 1
    f["gap_1d"] = gap
    f["ep_gap_20"] = gap.rolling(20).max()                 # best recent gap
    # gap with a contemporaneous volume surge, best over last 20d
    gap_vol = gap.where(vol > 2.0 * vol.rolling(60).mean())
    f["ep_gap_vol_20"] = gap_vol.rolling(20).max()

    # ---- continuous-information / frog-in-the-pan (smooth advances sustain)
    # ID = sign(formation ret) * (%neg days - %pos days) over 126d; low = smooth up
    form = close / close.shift(126) - 1
    pos_share = (R > 0).rolling(126).mean()
    neg_share = (R < 0).rolling(126).mean()
    f["fip"] = np.sign(form) * (neg_share - pos_share)

    # ---- beta / corr vs SPY (regime-sensitive tail / stabiliser)
    beta, corr = _rolling_beta_corr(R, spy.pct_change(), w=120)
    f["beta_120"] = beta
    f["corr_120"] = corr

    return f


def regime_risk_on(close_index) -> pd.Series:
    """SPY above its 200-day MA on day d (trailing). True = risk-on."""
    spy = dca_data.load_benchmark("SPY")["Close"].reindex(close_index).ffill()
    ma200 = spy.rolling(200).mean()
    return (spy > ma200).where(ma200.notna())


if __name__ == "__main__":
    P = dca_data.build_panel()
    f = build_features(P)
    last = P["close"].index[-1]
    mem = P["member"].loc[last]
    print(f"built {len(f)} features, panel {P['close'].shape}, last day {last.date()}")
    row = {k: v.loc[last][mem].notna().sum() for k, v in f.items()}
    for k in sorted(f):
        print(f"  {k:18s} non-nan members on last day: {row[k]}")
