"""PHOENIX crypto sleeve — weekly TSMOM on spot BTC/ETH, expressed via the
spot ETFs (IBIT / ETHA) from their listing dates.

History of this sleeve: the original version traded the Grayscale trusts
(GBTC / ETHE). Review found that its backtest P&L was dominated by the
trusts' NAV premium/discount dynamics (single days of +39..+68% that were
premium explosions, not BTC moves), booked at 10 bps costs on OTC vehicles
that traded 50-300 bps wide — and that the live book (IBIT) structurally
cannot earn that stream. This version removes the premium economics:

  - Signal AND return source is spot BTC-USD (from 2014-09) / ETH-USD
    (from 2017-11), reindexed to the NYSE calendar (weekend moves fold into
    Monday's open-to-open return), chained into IBIT / ETHA returns from
    each ETF's first trading day. The chaining is done in RETURN space, so
    there is no splice jump.
  - Costs: 30 bps/side while the exposure is a spot proxy (pre-ETF era:
    realistically tradable via exchange spot/futures, wider frictions),
    10 bps/side once the ETF is the vehicle.
  - ETHE is gone entirely (its early prints were zero-volume garbage that
    drove the momentum signal).

Strategy rules (unchanged from the original design — no new fitting):
  - 63-day momentum, lagged 1 day; hold each asset with positive momentum,
    equal-weighted; else BIL.
  - Weekly Friday rebalance, executed at the open.
  - Macro gate (lagged 1 day): SPY > 200dma AND 200dma rising (20d) AND
    HY-OAS 20d change < 1.0 AND VIX < 30; gate off -> 100% BIL.

Booking follows the unified realization-dated convention
(alt/sleeve_engine.py): the value at date t is the P&L realized over
open[t-1] -> open[t] by a position decided from close[t-2] information.

Outputs:
  data/results/crypto_returns.csv   (Date, ret) — NYSE calendar
  data/results/crypto_metrics.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
R = ROOT / "data/results"

IS_END = pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-02")

START_DATE = pd.Timestamp("2010-03-11")   # sleeve calendar start (BIL until BTC tradable)
TC_BPS_PROXY = 30.0   # per side, spot-proxy era
TC_BPS_ETF = 10.0     # per side, ETF era

# Spot series -> ETF expression and its first trading day (chained in
# return space at the splice; discovered from the ETF file itself).
ASSETS = {
    "BTC": {"spot": "BTC_USD", "etf": "IBIT"},
    "ETH": {"spot": "ETH_USD", "etf": "ETHA"},
}
CASH = "BIL"
MOM_LB = 63
REBAL_DOW = 4  # Friday


def load_etf(t):
    p = ETF / f"{t}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["Close", "Open"]].apply(pd.to_numeric, errors="coerce")


def load_fred(s):
    p = FRED / f"{s}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    d = d[~d.index.duplicated(keep="first")]
    return pd.to_numeric(d.iloc[:, 0], errors="coerce")


def metrics(r):
    r = r.dropna()
    if len(r) == 0:
        return {"sharpe": 0, "cagr": 0, "mdd": 0, "vol": 0}
    mu = r.mean() * 252
    sd = r.std() * np.sqrt(252)
    sr = mu / sd if sd > 0 else 0
    c = (1 + r).cumprod()
    dd = (c / c.cummax() - 1).min()
    yrs = len(r) / 252
    cagr = c.iloc[-1] ** (1 / yrs) - 1 if c.iloc[-1] > 0 else -1
    return {"sharpe": float(sr), "cagr": float(cagr), "mdd": float(dd), "vol": float(sd)}


def build_chained_panel(dates: pd.DatetimeIndex):
    """Per asset: chained o2o/c2c return series on the NYSE calendar (spot
    until the ETF lists, ETF after), a proxy-era mask, and chained synthetic
    close levels for the momentum signal."""
    o2o, c2c, closes, proxy_era, etf_first = {}, {}, {}, {}, {}
    for name, cfg in ASSETS.items():
        spot = load_etf(cfg["spot"])
        if spot is None:
            continue
        etf = load_etf(cfg["etf"])
        s_open = spot["Open"].reindex(dates).ffill(limit=5)
        s_close = spot["Close"].reindex(dates).ffill(limit=5)
        so2o = s_open.pct_change(fill_method=None)
        sc2c = s_close.pct_change(fill_method=None)
        if etf is not None and etf["Open"].dropna().size > 0:
            first = etf["Open"].dropna().index.min()
            e_open = etf["Open"].reindex(dates)
            e_close = etf["Close"].reindex(dates)
            eo2o = e_open.pct_change(fill_method=None)
            ec2c = e_close.pct_change(fill_method=None)
            # ETF returns from the first date BOTH etf prices exist;
            # the splice day itself keeps the spot return (no fake jump).
            use_etf = dates > first
            so2o = so2o.where(~use_etf, eo2o.where(eo2o.notna(), so2o))
            sc2c = sc2c.where(~use_etf, ec2c.where(ec2c.notna(), sc2c))
            etf_first[name] = first
        else:
            etf_first[name] = None
        o2o[name] = so2o
        c2c[name] = sc2c
        closes[name] = (1 + sc2c.fillna(0.0)).cumprod()
        pe = pd.Series(True, index=dates)
        if etf_first[name] is not None:
            pe.loc[dates > etf_first[name]] = False
        proxy_era[name] = pe
    return (pd.DataFrame(o2o), pd.DataFrame(c2c), pd.DataFrame(closes),
            pd.DataFrame(proxy_era), etf_first)


def build_regime(dates: pd.DatetimeIndex) -> pd.Series:
    hy = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    spy = load_etf("SPY")["Close"].reindex(dates).ffill(limit=5)
    spy_ma = spy.rolling(200).mean()
    spy_ok = (spy > spy_ma) & (spy_ma.diff(20) > 0)
    hy_slope = hy - hy.shift(20)
    return (spy_ok & (hy_slope < 1.0) & (vix < 30)).shift(1).fillna(False)


def build_asset_weights(dates, closes, regime_ok) -> pd.DataFrame:
    """Decision-dated weights over ASSETS + BIL (weekly Friday rebalance)."""
    universe = list(closes.columns)
    mom = closes.pct_change(MOM_LB, fill_method=None).shift(1)
    cols = universe + [CASH]
    W = pd.DataFrame(0.0, index=dates, columns=cols)
    current = pd.Series(0.0, index=cols)
    current[CASH] = 1.0
    for i, dt in enumerate(dates):
        if dt.dayofweek == REBAL_DOW or i == 0:
            if bool(regime_ok.iloc[i]):
                m = mom.iloc[i].dropna()
                m = m[m > 0]
                new_w = pd.Series(0.0, index=cols)
                if len(m) > 0:
                    for t in m.index:
                        new_w[t] = 1.0 / len(m)
                else:
                    new_w[CASH] = 1.0
                current = new_w
            else:
                current = pd.Series(0.0, index=cols)
                current[CASH] = 1.0
        W.iloc[i] = current.values
    return W


def _nyse_dates() -> pd.DatetimeIndex:
    spy = load_etf("SPY")
    dates = spy["Open"].dropna().index
    return dates[dates >= START_DATE]


def build_sleeve_returns() -> tuple[pd.Series, pd.DataFrame]:
    """Unified-convention sleeve returns with era-dependent costs."""
    dates = _nyse_dates()
    o2o, _, closes, proxy_era, _ = build_chained_panel(dates)
    regime_ok = build_regime(dates)
    W = build_asset_weights(dates, closes, regime_ok)

    bil = load_etf(CASH)["Open"].reindex(dates).ffill(limit=5)
    ret_panel = o2o.copy()
    ret_panel[CASH] = bil.pct_change(fill_method=None)

    w_prev = W.shift(1).fillna(0.0)
    gross = (w_prev[ret_panel.columns] * ret_panel).sum(axis=1)

    dw = (W - w_prev).abs()
    bps = pd.DataFrame(TC_BPS_ETF, index=dates, columns=W.columns)
    for name in ASSETS:
        if name in proxy_era.columns:
            bps.loc[proxy_era[name], name] = TC_BPS_PROXY
    bps[CASH] = 2.0  # BIL trades ~1-2 bps wide
    cost = (dw * bps / 1e4).sum(axis=1)

    net = gross - cost
    return net, W


def build_weights(use_live_proxy: bool = True,
                  live_extend: bool = False) -> pd.DataFrame:
    """Canonical CRYPTO daily target-weight DataFrame for the live signal.

    Columns are TRADABLE tickers: IBIT, ETHA (from their listing dates) and
    BIL. Before an ETF lists, its asset's weight is shown under the ETF
    ticker only from the listing date; the proxy era is backtest-only (there
    is nothing to trade live today except the ETFs).

    live_extend: If True, extend the date index by one BDay forward
        (ffilled) so the last row is W[t+1] using close[t] info.
    """
    dates = _nyse_dates()
    if live_extend and len(dates) > 0:
        dates = dates.append(pd.DatetimeIndex([dates[-1] + pd.tseries.offsets.BDay()]))
    o2o, _, closes, _, etf_first = build_chained_panel(dates)
    regime_ok = build_regime(dates)
    W = build_asset_weights(dates, closes, regime_ok)

    out_cols = [ASSETS[n]["etf"] for n in ASSETS if n in W.columns] + [CASH]
    out = pd.DataFrame(0.0, index=W.index, columns=out_cols)
    out[CASH] = W[CASH]
    for name in ASSETS:
        if name not in W.columns:
            continue
        etf_t = ASSETS[name]["etf"]
        first = etf_first.get(name)
        if first is None:
            # ETF not listed: weight stays in cash for live purposes
            out[CASH] = out[CASH] + W[name]
            continue
        listed = W.index >= first
        out.loc[listed, etf_t] = W.loc[listed, name]
        out.loc[~listed, CASH] = out.loc[~listed, CASH] + W.loc[~listed, name]
    return out


def main():
    net, W = build_sleeve_returns()

    m_full = metrics(net)
    m_is = metrics(net.loc[:IS_END])
    m_oos = metrics(net.loc[OOS_START:])
    active = (W[list(ASSETS.keys())].sum(axis=1) > 0)
    m_active = metrics(net[active])

    print("CRYPTO sleeve (spot-chained, unified convention):")
    print(f"  FULL  : SR {m_full['sharpe']:5.2f}  CAGR {m_full['cagr']*100:6.1f}%  "
          f"Vol {m_full['vol']*100:5.1f}%  MDD {m_full['mdd']*100:6.1f}%")
    print(f"  IS    : SR {m_is['sharpe']:5.2f}  CAGR {m_is['cagr']*100:6.1f}%")
    print(f"  OOS   : SR {m_oos['sharpe']:5.2f}  CAGR {m_oos['cagr']*100:6.1f}%")
    print(f"  active days: {int(active.sum())} ({active.mean()*100:.0f}%)  "
          f"active-only SR {m_active['sharpe']:.2f}")

    out = pd.DataFrame({"Date": net.index, "ret": net.values})
    out.to_csv(R / "crypto_returns.csv", index=False)
    (R / "crypto_metrics.json").write_text(json.dumps({
        "full": m_full, "is": m_is, "oos": m_oos,
        "active_days": int(active.sum()),
        "params": {"mom_lb": MOM_LB, "rebal": "W-FRI",
                   "tc_bps_proxy": TC_BPS_PROXY, "tc_bps_etf": TC_BPS_ETF,
                   "assets": {k: v for k, v in ASSETS.items()},
                   "gate": "SPY>200dma & 200dma 20d slope>0 & HY-OAS 20d chg<1.0 & VIX<30 (lag 1)"},
    }, indent=2))
    print(f"Saved crypto_returns.csv and crypto_metrics.json")


if __name__ == "__main__":
    main()
