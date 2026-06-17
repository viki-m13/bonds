"""Momentum + risk features for the L2GMOM network model (Baz/Lim feature set).

Faithful to Pu et al. (L2GMOM) Section 3.2 — eight momentum features per asset:
  * volatility-scaled returns over 1, 21, 63, 126, 252 trading days
  * three normalised MACD signals with (S,L) in {(8,24),(16,48),(32,96)}
Plus two risk features, because the objective here is *underwater avoidance*,
not return, and the network needs risk information to predict downside:
  * trailing 126d realised volatility
  * trailing 126d downside deviation

All features are causal (use closes through day d) and standardised
cross-sectionally within each date (only that day's members), which is both
leakage-free and the right scale for the graph's feature-distance metric.

Output is a float32 array  F x T x N  (feature, day, ticker), NaN where a
feature is undefined or the name is not eligible.
"""
import numpy as np
import pandas as pd

from data import load_panel, eligibility

RET_HORIZONS = [1, 21, 63, 126, 252]
MACD_PAIRS = [(8, 24), (16, 48), (32, 96)]
FEATURE_NAMES = ([f"vsret_{h}" for h in RET_HORIZONS]
                 + [f"macd_{s}_{l}" for s, l in MACD_PAIRS]
                 + ["vol_126", "dsdev_126"])


def _phi(y):
    return y * np.exp(-y ** 2 / 4.0) / 0.89


def _macd_signal(price: pd.DataFrame, s: int, l: int) -> pd.DataFrame:
    m = price.ewm(span=s, min_periods=s).mean() - price.ewm(span=l, min_periods=l).mean()
    q = m.div(price.rolling(63).std())
    y = q.div(q.rolling(252).std())
    return _phi(y)


def build_features(standardize: bool = True) -> tuple[np.ndarray, pd.DatetimeIndex,
                                                      pd.Index, list]:
    p = load_panel()
    close = p["close"]
    r = np.log(close).diff()
    dvol = r.ewm(span=60, min_periods=20).std()       # daily vol estimate

    feats = []
    for h in RET_HORIZONS:
        ret = close / close.shift(h) - 1.0
        feats.append(ret.div(dvol * np.sqrt(h)))
    for s, l in MACD_PAIRS:
        feats.append(_macd_signal(close, s, l))
    feats.append(r.rolling(126).std())                # trailing vol
    dn = r.where(r < 0, 0.0)
    feats.append(np.sqrt((dn ** 2).rolling(126).mean()))   # downside deviation

    elig = eligibility(min_history=252)
    arr = np.full((len(feats), close.shape[0], close.shape[1]), np.nan, np.float32)
    em = elig.to_numpy(bool)
    for f, df in enumerate(feats):
        a = df.to_numpy(float).copy()
        a[~em] = np.nan
        if standardize:
            mu = np.nanmean(a, axis=1, keepdims=True)
            sd = np.nanstd(a, axis=1, keepdims=True)
            a = (a - mu) / np.where(sd > 0, sd, np.nan)
            a = np.clip(a, -5, 5)                       # winsorise
        arr[f] = a
    return arr, close.index, close.columns, FEATURE_NAMES


if __name__ == "__main__":
    arr, idx, cols, names = build_features()
    print("features:", names)
    print("shape F,T,N =", arr.shape)
    # coverage on a recent date
    pos = -1
    valid = ~np.isnan(arr[:, pos, :]).any(axis=0)
    print(f"{idx[pos].date()}: {valid.sum()} names with full feature vectors")
