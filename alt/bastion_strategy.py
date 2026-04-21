"""
BASTION — Leveraged Risk Parity + Multi-Factor Kill Switch
===========================================================

Three truly-orthogonal leveraged sleeves (equity, rates, real-assets) with
STATIC risk-parity notional weights plus a multi-factor macro kill-switch.

Hypothesis: combining sleeve orthogonality with a kill switch keyed to
stock-bond correlation regime (catching 2022) and credit/vol/curve stress
(catching 2008/2020/2018Q4/2015) can materially lift Sharpe over the prior
ceiling ~1.0.

No daily vol targeting. Monthly base rebalance, daily kill switch.
Signal lag: close[t-1] through. Execution at open[t]; PnL open[t]->open[t+1].
TC: 5 bps one-way on |Δw|.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
ETF_DIR = ROOT / "data" / "etfs"
FRED_DIR = ROOT / "data" / "fred"
RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IS_START = pd.Timestamp("2010-03-11")
IS_END = pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-01")
OOS_END = pd.Timestamp("2026-04-02")

TC_BPS = 5.0
TC_RATE = TC_BPS / 1e4

EQ_POOL = ["UPRO", "TQQQ", "QLD", "SSO"]
RATES_POOL = ["TMF", "UBT", "TYD"]
RA_POOL = ["UGL", "UCO", "NUGT", "DRN"]
ALL_LEV = EQ_POOL + RATES_POOL + RA_POOL  # 11 leveraged ETFs
CASH = "BIL"


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_etf(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(ETF_DIR / f"{ticker}.csv", parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df[["Open", "Close"]].astype(float)


def load_fred(name: str) -> pd.Series:
    df = pd.read_csv(FRED_DIR / f"{name}.csv", parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df.iloc[:, 0].astype(float)


def build_panels(universe: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    opens, closes = {}, {}
    for t in universe:
        d = load_etf(t)
        opens[t] = d["Open"]
        closes[t] = d["Close"]
    opens = pd.DataFrame(opens).sort_index()
    closes = pd.DataFrame(closes).sort_index()
    idx = pd.bdate_range(opens.index.min(), opens.index.max())
    opens = opens.reindex(idx).ffill(limit=2)
    closes = closes.reindex(idx).ffill(limit=2)
    return opens, closes


def build_fred(idx: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    f["VIX"] = load_fred("VIXCLS").reindex(idx).ffill()
    f["HY"] = load_fred("BAMLH0A0HYM2").reindex(idx).ffill()
    f["T10Y2Y"] = load_fred("T10Y2Y").reindex(idx).ffill()
    return f


def load_underlying_close(ticker: str, idx: pd.DatetimeIndex) -> pd.Series:
    s = (
        pd.read_csv(ETF_DIR / f"{ticker}.csv", parse_dates=["Date"])
        .sort_values("Date")
        .set_index("Date")["Close"]
        .astype(float)
    )
    return s.reindex(idx).ffill(limit=2)


# --------------------------------------------------------------------------- #
# Sleeve construction — pick best name inside each pool by medium-term momentum
# --------------------------------------------------------------------------- #
def pick_sleeve(closes: pd.DataFrame, pool: list[str], mom_lb: int, sma_lb: int) -> pd.DataFrame:
    """Binary selector: on each date, return 1.0 on the top-momentum eligible
    name in the pool, 0 elsewhere. Uses close[t-1] only.

    Eligibility: name must have mom > 0 AND close > sma.
    If no eligible name, sleeve returns 0 (will be absorbed as implicit cash).
    """
    c_lag = closes[pool].shift(1)
    mom = c_lag / c_lag.shift(mom_lb) - 1.0
    above_sma = c_lag > c_lag.rolling(sma_lb).mean()
    eligible = (mom > 0) & above_sma & c_lag.notna()
    m = mom.where(eligible)
    # pick highest-momentum name
    best = m.idxmax(axis=1)  # NaN where no row qualifies
    w = pd.DataFrame(0.0, index=closes.index, columns=pool)
    for dt, nm in best.items():
        if isinstance(nm, str):
            w.at[dt, nm] = 1.0
    return w


# --------------------------------------------------------------------------- #
# Monthly rebalance mask
# --------------------------------------------------------------------------- #
def month_start_mask(idx: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(idx, index=idx)
    return s.groupby([idx.year, idx.month]).transform("first") == s


def freeze_monthly(daily_choice: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Freeze sleeve picks between monthly rebalance dates using the previous
    day's ranking (already lagged inside pick_sleeve)."""
    mask = month_start_mask(idx)
    out = pd.DataFrame(0.0, index=idx, columns=daily_choice.columns)
    cur = pd.Series(0.0, index=daily_choice.columns)
    for dt in idx:
        if mask.loc[dt]:
            row = daily_choice.loc[dt]
            if row.sum() > 0:
                cur = row.copy()
            # else keep cur (last picks) during blackout — but blank if > 0 initially
        out.loc[dt] = cur.values
    return out


# --------------------------------------------------------------------------- #
# Kill switch — multi-factor
# --------------------------------------------------------------------------- #
def compute_killswitch(
    fred: pd.DataFrame,
    spy: pd.Series,
    tlt: pd.Series,
    hy_z_thr: float = 1.5,
    vix_thr: float = 30.0,
    curve_days: int = 20,
    corr_window: int = 60,
    corr_thr: float = 0.4,
    slow_reentry: int = 7,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return a boolean series 'risk_off' and a diagnostic DataFrame.

    Fast exit: any trigger on a given day (using data through close[t-1])
    flips risk_off=True. Slow re-entry: require slow_reentry consecutive
    all-clear days before flipping back to risk_on.

    All inputs lagged by 1 bar to prevent contemporaneous bleed.
    """
    vix = fred["VIX"].shift(1)
    hy = fred["HY"].shift(1)
    t10y2y = fred["T10Y2Y"].shift(1)
    spy_l = spy.shift(1)
    tlt_l = tlt.shift(1)

    # 1) HY OAS z-score (60d) > +1.5 AND 20d change > 0
    hy_mu = hy.rolling(60).mean()
    hy_sd = hy.rolling(60).std()
    hy_z = (hy - hy_mu) / hy_sd
    hy_chg20 = hy - hy.shift(20)
    c_hy = (hy_z > hy_z_thr) & (hy_chg20 > 0)

    # 2) T10Y2Y < -0.3 for 20+ consecutive days AND widening (more negative)
    inv = (t10y2y < -0.3).astype(float)
    inv_run = inv.groupby((inv != inv.shift()).cumsum()).cumsum()
    c_curve = (inv_run >= curve_days) & (t10y2y < t10y2y.shift(5))

    # 3) VIX > 30 AND 20d slope > 0
    vix_slope20 = vix - vix.shift(20)
    c_vix = (vix > vix_thr) & (vix_slope20 > 0)

    # 4) SPY < 200d MA AND MA slope negative
    spy_ma = spy_l.rolling(200).mean()
    ma_slope = spy_ma - spy_ma.shift(20)
    c_spy = (spy_l < spy_ma) & (ma_slope < 0)

    # 5) Stock-bond 60d correlation > +0.4 (regime like 2022)
    spy_r = spy_l.pct_change()
    tlt_r = tlt_l.pct_change()
    sb_corr = spy_r.rolling(corr_window).corr(tlt_r)
    c_corr = sb_corr > corr_thr

    any_trigger = (
        c_hy.fillna(False)
        | c_curve.fillna(False)
        | c_vix.fillna(False)
        | c_spy.fillna(False)
        | c_corr.fillna(False)
    )

    # Hysteresis: fast exit, slow re-entry
    risk_off = pd.Series(False, index=any_trigger.index)
    clear_streak = 0
    state_off = False
    for dt, trig in any_trigger.items():
        if state_off:
            if not trig:
                clear_streak += 1
                if clear_streak >= slow_reentry:
                    state_off = False
                    clear_streak = 0
            else:
                clear_streak = 0
        else:
            if trig:
                state_off = True
                clear_streak = 0
        risk_off.loc[dt] = state_off

    diag = pd.DataFrame({
        "hy_z": hy_z,
        "hy_chg20": hy_chg20,
        "t10y2y": t10y2y,
        "vix": vix,
        "vix_slope20": vix_slope20,
        "spy_below_ma": c_spy.astype(float),
        "sb_corr": sb_corr,
        "c_hy": c_hy.astype(float),
        "c_curve": c_curve.astype(float),
        "c_vix": c_vix.astype(float),
        "c_spy": c_spy.astype(float),
        "c_corr": c_corr.astype(float),
        "any_trigger": any_trigger.astype(float),
        "risk_off": risk_off.astype(float),
    })
    return risk_off, diag


# --------------------------------------------------------------------------- #
# Backtest engine
# --------------------------------------------------------------------------- #
def backtest(opens: pd.DataFrame, weights: pd.DataFrame, tc_rate: float = TC_RATE) -> pd.DataFrame:
    """open-to-open backtest with TC.

    weight_t applied to open[t+1]/open[t] - 1 (signal uses close[t-1]).
    """
    o2o = opens / opens.shift(1) - 1.0
    w_lag = weights.shift(1).fillna(0.0)  # weight set at open[t-1] earns open[t-1]->open[t]
    gross = (w_lag * o2o).sum(axis=1)
    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost_lag = (turnover * tc_rate).shift(1).fillna(0.0)
    net = gross - cost_lag
    return pd.DataFrame({
        "gross_ret": gross,
        "cost": cost_lag,
        "net_ret": net,
        "turnover": turnover.shift(1).fillna(0.0),
    })


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def perf_metrics(rets: pd.Series, periods: int = 252) -> dict:
    r = rets.dropna()
    if len(r) < 5:
        return {"sharpe": np.nan, "cagr": np.nan, "vol": np.nan, "mdd": np.nan, "n": len(r)}
    mu = r.mean() * periods
    sigma = r.std(ddof=0) * np.sqrt(periods)
    sharpe = mu / sigma if sigma > 0 else np.nan
    navx = float((1.0 + r).prod())
    yrs = len(r) / periods
    cagr = navx ** (1.0 / yrs) - 1.0 if yrs > 0 else np.nan
    nav = (1.0 + r).cumprod()
    mdd = float((nav / nav.cummax() - 1.0).min())
    return {
        "sharpe": float(sharpe),
        "cagr": float(cagr),
        "vol": float(sigma),
        "mdd": mdd,
        "n": int(len(r)),
        "navx": navx,
    }


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run(
    mom_lb: int = 126,
    sma_lb: int = 200,
    w_eq: float = 0.40,
    w_rates: float = 0.40,
    w_ra: float = 0.20,
    gross: float = 1.0,
    hy_z_thr: float = 1.5,
    vix_thr: float = 30.0,
    curve_days: int = 20,
    corr_window: int = 60,
    corr_thr: float = 0.4,
    slow_reentry: int = 7,
    verbose: bool = True,
):
    universe = ALL_LEV + [CASH]
    opens, closes = build_panels(universe)
    idx = opens.index
    fred = build_fred(idx)
    spy = load_underlying_close("SPY", idx)
    tlt = load_underlying_close("TLT", idx)

    # daily sleeve picks (binary selector already lagged 1 bar inside)
    eq_daily = pick_sleeve(closes, EQ_POOL, mom_lb, sma_lb)
    rt_daily = pick_sleeve(closes, RATES_POOL, mom_lb, sma_lb)
    ra_daily = pick_sleeve(closes, RA_POOL, mom_lb, sma_lb)

    # freeze choice to monthly within each sleeve
    eq_w = freeze_monthly(eq_daily, idx)
    rt_w = freeze_monthly(rt_daily, idx)
    ra_w = freeze_monthly(ra_daily, idx)

    # Static risk-parity notional sleeve weights (tuned on IS)
    # scaled by gross multiplier
    eq_final = eq_w.mul(w_eq * gross)
    rt_final = rt_w.mul(w_rates * gross)
    ra_final = ra_w.mul(w_ra * gross)

    # Kill switch
    risk_off, diag = compute_killswitch(
        fred, spy, tlt,
        hy_z_thr=hy_z_thr, vix_thr=vix_thr, curve_days=curve_days,
        corr_window=corr_window, corr_thr=corr_thr, slow_reentry=slow_reentry,
    )
    on_mult = (~risk_off).astype(float)  # 1 when risk-on, 0 when risk-off

    # Merge weights into a single DataFrame keyed to universe columns
    weights = pd.DataFrame(0.0, index=idx, columns=universe)
    for col in EQ_POOL:
        weights[col] = eq_final[col] * on_mult
    for col in RATES_POOL:
        weights[col] = rt_final[col] * on_mult
    for col in RA_POOL:
        weights[col] = ra_final[col] * on_mult

    # Residual to BIL (cash) — whenever gross exposure < 1 and especially in risk-off
    cash_w = 1.0 - weights.sum(axis=1)
    cash_w = cash_w.clip(lower=0.0)
    weights[CASH] = cash_w

    # Track sleeve weight sums for diagnostics
    w_eq_series = weights[EQ_POOL].sum(axis=1)
    w_rt_series = weights[RATES_POOL].sum(axis=1)
    w_ra_series = weights[RA_POOL].sum(axis=1)
    w_cash_series = weights[CASH]

    bt = backtest(opens, weights, tc_rate=TC_RATE).loc[IS_START:OOS_END]
    net = bt["net_ret"]

    is_r = net.loc[IS_START:IS_END]
    oos_r = net.loc[OOS_START:OOS_END]

    metrics = {
        "full": perf_metrics(net),
        "is": perf_metrics(is_r),
        "oos": perf_metrics(oos_r),
        "avg_turnover_annual": float(bt["turnover"].sum() / ((net.index[-1] - net.index[0]).days / 365.25)),
    }

    # time in market
    in_mkt = (weights[ALL_LEV].abs().sum(axis=1) > 0).reindex(net.index).astype(float)
    metrics["avg_time_in_market"] = {
        "full": float(in_mkt.mean()),
        "is": float(in_mkt.loc[IS_START:IS_END].mean()),
        "oos": float(in_mkt.loc[OOS_START:OOS_END].mean()),
    }

    # kill-switch analytics
    # count trigger transitions (False -> True)
    transitions = (risk_off.astype(int).diff() == 1).astype(int).loc[IS_START:OOS_END]
    ks_per_year = transitions.groupby(transitions.index.year).sum().to_dict()
    metrics["killswitch_transitions_per_year"] = {int(k): int(v) for k, v in ks_per_year.items()}
    metrics["killswitch_days_off"] = {
        "full": int(risk_off.loc[IS_START:OOS_END].sum()),
        "is": int(risk_off.loc[IS_START:IS_END].sum()),
        "oos": int(risk_off.loc[OOS_START:OOS_END].sum()),
    }

    # per-sleeve standalone metrics (sleeve-weighted alone + kill-switch)
    per_sleeve = {}
    for name, sleeve_w, cols in [
        ("equity", eq_final.mul(on_mult, axis=0), EQ_POOL),
        ("rates", rt_final.mul(on_mult, axis=0), RATES_POOL),
        ("real_assets", ra_final.mul(on_mult, axis=0), RA_POOL),
    ]:
        sw = pd.DataFrame(0.0, index=idx, columns=universe)
        for c in cols:
            sw[c] = sleeve_w[c]
        sw[CASH] = (1.0 - sw.sum(axis=1)).clip(lower=0.0)
        sbt = backtest(opens, sw, tc_rate=TC_RATE).loc[IS_START:OOS_END]
        per_sleeve[name] = {
            "full": perf_metrics(sbt["net_ret"]),
            "is": perf_metrics(sbt["net_ret"].loc[IS_START:IS_END]),
            "oos": perf_metrics(sbt["net_ret"].loc[OOS_START:OOS_END]),
        }
    metrics["per_sleeve"] = per_sleeve

    # per-year returns
    yearly = (1 + net).groupby(net.index.year).prod() - 1
    metrics["yearly_returns"] = {int(y): float(v) for y, v in yearly.items()}

    metrics["params"] = {
        "mom_lb": mom_lb, "sma_lb": sma_lb,
        "w_eq": w_eq, "w_rates": w_rates, "w_ra": w_ra, "gross": gross,
        "hy_z_thr": hy_z_thr, "vix_thr": vix_thr, "curve_days": curve_days,
        "corr_window": corr_window, "corr_thr": corr_thr,
        "slow_reentry": slow_reentry, "tc_bps": TC_BPS,
        "eq_pool": EQ_POOL, "rates_pool": RATES_POOL, "ra_pool": RA_POOL,
    }

    if verbose:
        print("=" * 80)
        print("BASTION  sleeves eq={}  rates={}  ra={}".format(EQ_POOL, RATES_POOL, RA_POOL))
        print(f"  mom_lb={mom_lb}  sma_lb={sma_lb}  gross={gross}x")
        print(f"  notional w: eq={w_eq}  rates={w_rates}  ra={w_ra}")
        print(f"  killswitch: hy_z>{hy_z_thr}, vix>{vix_thr}, curve>={curve_days}d<-0.3, "
              f"sb_corr>{corr_thr} (w={corr_window}d), re-entry={slow_reentry}d")
        print("-" * 80)
        for k in ["full", "is", "oos"]:
            m = metrics[k]
            print(f"  {k.upper():4s}  Sh={m['sharpe']:6.3f}  CAGR={m['cagr']*100:6.2f}%  "
                  f"Vol={m['vol']*100:5.2f}%  MDD={m['mdd']*100:7.2f}%  "
                  f"N={m['n']:4d}  NAVx={m['navx']:.2f}")
        gap = abs(metrics["is"]["sharpe"] - metrics["oos"]["sharpe"])
        print(f"  |IS-OOS gap|={gap:.3f}  TO/yr={metrics['avg_turnover_annual']:.2f}")
        print(f"  time-in-mkt IS={metrics['avg_time_in_market']['is']:.2f}  "
              f"OOS={metrics['avg_time_in_market']['oos']:.2f}")
        print("-" * 80)
        print("  Killswitch transitions per year:")
        for y, c in sorted(metrics["killswitch_transitions_per_year"].items()):
            print(f"    {y}: {c}")
        print("-" * 80)
        print("  Per-sleeve (sleeve alone + killswitch, full window):")
        for n, sm in per_sleeve.items():
            f = sm["full"]
            print(f"    {n:12s} Sh={f['sharpe']:6.3f}  CAGR={f['cagr']*100:6.2f}%  "
                  f"Vol={f['vol']*100:5.2f}%  MDD={f['mdd']*100:7.2f}%")
        print("-" * 80)
        print("  Yearly net returns:")
        for y, v in sorted(metrics["yearly_returns"].items()):
            print(f"    {y}: {v*100:6.2f}%")
        print("=" * 80)

    return bt, weights, metrics, risk_off, diag, w_eq_series, w_rt_series, w_ra_series, w_cash_series


def save_outputs(bt, metrics, risk_off, w_eq_s, w_rt_s, w_ra_s, w_cash_s, name="bastion"):
    out_csv = RESULTS_DIR / f"{name}_returns.csv"
    out_json = RESULTS_DIR / f"{name}_metrics.json"
    df = pd.DataFrame({
        "ret": bt["net_ret"],
        "gross_ret": bt["gross_ret"],
        "cost": bt["cost"],
        "turnover": bt["turnover"],
        "risk_off": risk_off.reindex(bt.index).astype(int),
        "weight_eq": w_eq_s.reindex(bt.index),
        "weight_rates": w_rt_s.reindex(bt.index),
        "weight_ra": w_ra_s.reindex(bt.index),
        "weight_cash": w_cash_s.reindex(bt.index),
    })
    df.index.name = "Date"
    df.to_csv(out_csv)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    bt, w, metrics, ro, diag, we, wr, wra, wc = run()
    save_outputs(bt, metrics, ro, we, wr, wra, wc)
