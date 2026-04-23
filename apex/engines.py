"""APEX — signal engines (v2, simpler and stronger).

Each engine is `fn(open_, close_) -> weights_df` where weights_df is T×N with
rows summing to ≤ 1 (no portfolio leverage). Signals computed from close[t-1],
activated on day t.

ENGINES (all independently vol-targetable):
  BETA   — Equity LETF with trend/crash filter (200-day MA + drawdown throttle)
  ROT    — Cross-asset dual-momentum rotation across 7 risk assets
  BOND   — Treasury duration: TMF when rates trending down, cash otherwise
  GOLD   — Gold trend: UGL when GLD trending up (uncorrelated safe haven)
  VRP    — Volatility risk premium: long equity when VIX term structure flat
  CRED   — Credit-spread regime: long equity-LETF when HY spreads tight + trending

Key ideas that make this work:
  • Trend filters strip the bulk of drawdowns from LETFs (the biggest risk).
  • Each engine holds a *different* underlying (equity / bond / gold / credit),
    creating genuinely uncorrelated sleeves.
  • Daily vol targeting per engine keeps realized vol stable; the blend is
    then inverse-vol weighted so each engine contributes equally to risk.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from util import DPY, LETFS_3X, LETFS_2X

ROOT = Path("/home/user/bonds")
FRED = ROOT / "data/fred"


# --- helpers ---------------------------------------------------------------

def _load_fred(name: str, idx: pd.DatetimeIndex) -> pd.Series:
    fp = FRED / f"{name}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    col = df.columns[0]
    s = df[col].astype(float).reindex(idx).ffill()
    return s


def _dd_throttle(ret: pd.Series, win: int = 252, floor: float = -0.15) -> pd.Series:
    """Linear drawdown throttle.

    DD = NAV / max(NAV[t-win:t]) - 1
    mult = clip(1 + DD / floor, 0, 1)
    When NAV hits `floor` drawdown, multiplier is 0 (full exit).
    Lagged by 1 day.
    """
    c = (1 + ret).cumprod()
    hwm = c.rolling(win, min_periods=30).max()
    dd = c / hwm - 1
    m = (1.0 + dd / floor).clip(lower=0.0, upper=1.0).shift(1).fillna(1.0)
    return m


def _cash_ticker(close: pd.DataFrame) -> str:
    for t in ("BIL", "SHY"):
        if t in close.columns:
            return t
    return None


# --- Engine BETA: trend-filtered equity LETF --------------------------------

def engine_beta(open_: pd.DataFrame, close: pd.DataFrame,
                main: str = "TQQQ", cash_weight_off: float = 1.0,
                fast: int = 50, slow: int = 200, ret_win: int = 126) -> pd.DataFrame:
    """Long TQQQ (or alt) when QQQ trend positive; cash otherwise.

    Trend-on condition (must all be true):
        • QQQ  Close > 200d MA
        • QQQ  50d MA > 200d MA
        • QQQ  126d return > 0

    Vol filter: when QQQ 60d realized vol > 40% ann, scale LETF position by
    40%/RV (reduce exposure in vol spikes).
    """
    if main not in close.columns:
        raise KeyError(main)

    under = "QQQ" if main in ("TQQQ", "QLD") else ("SPY" if main in ("UPRO", "SSO") else main)
    u = close.get(under, close["SPY"])

    ma_s = u.rolling(slow).mean()
    ma_f = u.rolling(fast).mean()
    r = u.pct_change(ret_win)
    trend_on = ((u > ma_s) & (ma_f > ma_s) & (r > 0)).astype(float)

    rv = u.pct_change().rolling(60).std() * np.sqrt(DPY)
    vol_scale = (0.25 / rv).clip(upper=1.0).fillna(1.0)

    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w[main] = (trend_on * vol_scale).fillna(0.0)

    cash = _cash_ticker(close)
    if cash is not None:
        w[cash] = (1 - w[main]).clip(lower=0.0) * cash_weight_off
    return w


# --- Engine ROT: cross-asset dual-momentum rotation -------------------------

def engine_rot(open_: pd.DataFrame, close: pd.DataFrame,
               universe: list[str] | None = None,
               lookback: int = 126, top_n: int = 2,
               min_mom: float = 0.0) -> pd.DataFrame:
    """Dual-momentum rotation across multiple asset classes.

    Universe defaults: TQQQ, UPRO, EDC, TMF, UGL (equity/intl/bond/gold LETFs).
    Each day: rank by trailing 126d return, long top_n with return > 0.
    Equal-weight across selected. If none selected, 100% cash.
    """
    if universe is None:
        universe = ["TQQQ", "UPRO", "EDC", "TMF", "UGL", "UBT"]
    universe = [a for a in universe if a in close.columns]
    p = close[universe]
    mom = p.pct_change(lookback)
    # Skip last 21d to avoid near-term reversal
    mom_skip = p.shift(21).pct_change(lookback - 21)
    # Rank (higher = better)
    rnk = mom_skip.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= top_n) & (mom_skip > min_mom)
    n_sel = sel.sum(axis=1)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for a in universe:
        w[a] = (sel[a].astype(float) / n_sel.replace(0, np.nan)).fillna(0.0)
    # Cash when nothing selected
    cash = _cash_ticker(close)
    if cash is not None:
        w[cash] = (n_sel == 0).astype(float)
    return w


# --- Engine BOND: treasury rate-momentum -----------------------------------

def engine_bond(open_: pd.DataFrame, close: pd.DataFrame,
                fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Long TMF when TLT is in an uptrend. Cash otherwise.

    No leverage beyond TMF (3x). Drives uncorrelated-to-equity returns.
    """
    if "TMF" not in close.columns or "TLT" not in close.columns:
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)
    tlt = close["TLT"]
    ma_s = tlt.rolling(slow).mean()
    ma_f = tlt.rolling(fast).mean()
    trend_on = ((tlt > ma_s) & (ma_f > ma_s) & (tlt.pct_change(126) > 0)).astype(float)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w["TMF"] = trend_on
    cash = _cash_ticker(close)
    if cash is not None:
        w[cash] = 1 - trend_on
    return w


# --- Engine GOLD: gold trend + inflation regime -----------------------------

def engine_gold(open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """UGL (2x gold) when GLD trend positive; cash otherwise."""
    if "UGL" not in close.columns or "GLD" not in close.columns:
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)
    gld = close["GLD"]
    ma_s = gld.rolling(200).mean()
    ma_f = gld.rolling(50).mean()
    trend_on = ((gld > ma_s) & (ma_f > ma_s) & (gld.pct_change(126) > 0)).astype(float)
    rv = gld.pct_change().rolling(60).std() * np.sqrt(DPY)
    vol_scale = (0.20 / rv).clip(upper=1.0).fillna(1.0)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w["UGL"] = (trend_on * vol_scale).fillna(0.0)
    cash = _cash_ticker(close)
    if cash is not None:
        w[cash] = (1 - w["UGL"]).clip(lower=0.0)
    return w


# --- Engine VRP: vol-risk-premium proxy ------------------------------------

def engine_vrp(open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Low-vol regime → long SSO (2x SPY). High-vol → cash.

    Proxy for VIX term structure (we don't have VIX futures history). We use
    SPY 21d realized vol: when RV < 15% ann, VRP is usually in contango and
    long-vol is cheap, meaning short-vol ≈ long equity works well. When RV
    spikes above 25%, shift to cash.
    """
    if "SSO" not in close.columns or "SPY" not in close.columns:
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)
    spy = close["SPY"]
    r = spy.pct_change()
    rv21 = r.rolling(21).std() * np.sqrt(DPY)
    rv63 = r.rolling(63).std() * np.sqrt(DPY)
    # Long risk on when RV in bottom third
    regime = (rv21 < 0.18) & (rv21 < rv63)
    on = regime.astype(float)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w["SSO"] = on
    cash = _cash_ticker(close)
    if cash is not None:
        w[cash] = 1 - on
    return w


# --- Engine CRED: credit-spread regime --------------------------------------

def engine_cred(open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Long UPRO when HY credit spread tight AND tightening. Cash otherwise.

    HY OAS (BAMLH0A0HYM2) < 90th pct trailing 2y AND < 60-day avg ⇒ risk-on.
    """
    if "UPRO" not in close.columns:
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)
    idx = close.index
    hy = _load_fred("BAMLH0A0HYM2", idx)
    if hy.isna().all():
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)
    med_2y = hy.rolling(504).median()
    pct90 = hy.rolling(504).quantile(0.90)
    ma60 = hy.rolling(60).mean()
    tight = (hy < pct90)
    tightening = (hy < ma60)
    on = (tight & tightening).astype(float)
    # Scale by inverse spread (tighter = higher conviction)
    scale = (med_2y / hy.replace(0, np.nan)).clip(upper=2.0, lower=0.5).fillna(1.0)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w["UPRO"] = (on * scale).clip(upper=1.0)
    cash = _cash_ticker(close)
    if cash is not None:
        w[cash] = (1 - w["UPRO"]).clip(lower=0.0)
    return w


# --- Engine SECTOR: strong-sector rotation ----------------------------------

def engine_sector(open_: pd.DataFrame, close: pd.DataFrame,
                  universe: list[str] | None = None,
                  lookback: int = 126, top_n: int = 1) -> pd.DataFrame:
    """Rotate into the strongest *sector* LETF (TECL/FAS/SOXL/ERX).

    Single-winner selection with 200-day market filter: only active if
    SPY > 200d MA. Else cash.
    """
    if universe is None:
        universe = ["TECL", "FAS", "SOXL", "ERX"]
    universe = [a for a in universe if a in close.columns]
    if "SPY" not in close.columns:
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)

    market_filter = (close["SPY"] > close["SPY"].rolling(200).mean()).astype(float)

    p = close[universe]
    mom = p.shift(21).pct_change(lookback - 21)
    rnk = mom.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= top_n) & (mom > 0)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for a in universe:
        w[a] = sel[a].astype(float) * market_filter / top_n
    cash = _cash_ticker(close)
    if cash is not None:
        w[cash] = (1 - w.drop(columns=[cash], errors="ignore").sum(axis=1)).clip(lower=0.0)
    return w
