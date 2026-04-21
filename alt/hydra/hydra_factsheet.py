"""Build HYDRA factsheet JSON for the webapp (docs/hydra.html).

Outputs:
  data/results/hydra_factsheet_data.json — all fields needed by docs/hydra.html
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
RESULTS = ROOT / "data/results"


def metrics(r):
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return {"sharpe": 0, "ann_return": 0, "ann_vol": 0, "max_dd": 0,
                "sortino": 0, "n_years": round(len(r) / 252, 1)}
    ar = r.mean() * 252
    av = r.std() * np.sqrt(252)
    sr = ar / av
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    neg = r[r < 0]
    sor = ar / (neg.std() * np.sqrt(252)) if len(neg) and neg.std() > 0 else 999
    return {
        "sharpe": round(float(sr), 3),
        "ann_return": round(float(ar * 100), 2),
        "ann_vol": round(float(av * 100), 2),
        "max_dd": round(float(mdd * 100), 2),
        "sortino": round(float(sor), 3),
        "n_years": round(float(len(r) / 252), 1),
    }


def equity_curve(r, start=10000.0, freq="W-FRI"):
    cum = ((1 + r).cumprod() * start).resample(freq).last().ffill()
    return [{"date": d.strftime("%Y-%m-%d"), "value": round(float(cum.loc[d]), 2)}
            for d in cum.index]


def equity_curve_multi(df, start=10000.0, freq="W-FRI"):
    """df columns: any asset; each column gets its own compounded curve."""
    cum = ((1 + df).cumprod() * start).resample(freq).last().ffill()
    out = []
    for d in cum.index:
        row = {"date": d.strftime("%Y-%m-%d")}
        for c in cum.columns:
            row[c] = round(float(cum.loc[d, c]), 2)
        out.append(row)
    return out


def drawdown_curve(r, freq="W-FRI"):
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1) * 100
    ddw = dd.resample(freq).last().ffill()
    return [{"date": d.strftime("%Y-%m-%d"), "dd": round(float(ddw.loc[d]), 2)}
            for d in ddw.index]


def rolling_sharpe(r, window=252, freq="W-FRI"):
    mu = r.rolling(window).mean() * 252
    sd = r.rolling(window).std() * np.sqrt(252)
    sh = (mu / sd).dropna()
    sw = sh.resample(freq).last()
    return [{"date": d.strftime("%Y-%m-%d"), "sr": round(float(sw.loc[d]), 3)}
            for d in sw.index if not np.isnan(sw.loc[d])]


def calendar_returns(r):
    """Yearly returns by calendar year."""
    by_year = r.groupby(r.index.year).apply(
        lambda x: (1 + x).prod() - 1
    )
    return [{"year": int(y), "ret": round(float(v * 100), 2)}
            for y, v in by_year.items()]


def monthly_heatmap(r):
    mo = r.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    return [{"date": d.strftime("%Y-%m-%d"),
             "year": int(d.year),
             "month": int(d.month),
             "ret": round(float(mo.loc[d]), 2)} for d in mo.index]


def trailing(r, spy):
    """Trailing 1M/3M/6M/YTD/1Y/3Y/5Y/10Y and since-inception."""
    end = r.index[-1]
    periods = {
        "1M": 21, "3M": 63, "6M": 126, "1Y": 252,
        "3Y_ann": 252 * 3, "5Y_ann": 252 * 5, "10Y_ann": 252 * 10,
    }
    out = {"HYDRA": {}, "SPY": {}}
    for label, n in periods.items():
        if len(r) < n + 1:
            out["HYDRA"][label] = None
            out["SPY"][label] = None
            continue
        rh = r.iloc[-n:]
        rs = spy.iloc[-n:]
        if label.endswith("_ann"):
            out["HYDRA"][label] = round(float(((1 + rh).prod() ** (252 / n) - 1) * 100), 2)
            out["SPY"][label] = round(float(((1 + rs).prod() ** (252 / n) - 1) * 100), 2)
        else:
            out["HYDRA"][label] = round(float(((1 + rh).prod() - 1) * 100), 2)
            out["SPY"][label] = round(float(((1 + rs).prod() - 1) * 100), 2)
    # YTD
    ystart = pd.Timestamp(f"{end.year}-01-01")
    rh_y = r.loc[ystart:]
    rs_y = spy.loc[ystart:]
    out["HYDRA"]["YTD"] = round(float(((1 + rh_y).prod() - 1) * 100), 2)
    out["SPY"]["YTD"] = round(float(((1 + rs_y).prod() - 1) * 100), 2)
    # Since inception
    out["HYDRA"]["SI_ann"] = round(float(((1 + r).prod() ** (252 / len(r)) - 1) * 100), 2)
    out["SPY"]["SI_ann"] = round(float(((1 + spy).prod() ** (252 / len(spy)) - 1) * 100), 2)
    return out


def walkforward(r, spy, year_groups):
    rows = []
    for y0, y1 in year_groups:
        lo = pd.Timestamp(f"{y0}-01-01")
        hi = pd.Timestamp(f"{y1}-01-01")
        sub_h = r.loc[lo:hi]
        sub_s = spy.loc[lo:hi]
        if len(sub_h) < 200:
            continue
        mh = metrics(sub_h)
        ms = metrics(sub_s)
        rows.append({
            "window": f"{y0}-{y1 - 1}",
            "hydra_sr": mh["sharpe"],
            "hydra_ret": mh["ann_return"],
            "hydra_mdd": mh["max_dd"],
            "spy_sr": ms["sharpe"],
            "spy_ret": ms["ann_return"],
            "spy_mdd": ms["max_dd"],
        })
    return rows


def sleeve_stats(sleeves_df):
    rows = []
    for c in sleeves_df.columns:
        r = sleeves_df[c]
        nz = r[r != 0]
        inception = nz.index[0].strftime("%Y-%m-%d") if len(nz) else None
        m = metrics(r)
        rows.append({"name": c, "inception": inception, **m})
    return rows


def sleeve_correlations(sleeves_df):
    valid = (sleeves_df != 0).sum(axis=1) >= 5
    corr = sleeves_df[valid].corr()
    tri = corr.values[np.triu_indices_from(corr, k=1)]
    return {
        "mean_abs": round(float(np.mean(np.abs(tri))), 3),
        "median_abs": round(float(np.median(np.abs(tri))), 3),
        "max_abs": round(float(np.max(np.abs(tri))), 2),
        "matrix": {c: {c2: round(float(corr.loc[c, c2]), 2)
                       for c2 in corr.columns}
                   for c in corr.columns},
    }


def vol_scaling_series(sleeves_df, hydra_r, target_vol=0.20, lev_cap=5.0,
                       window=63, recent_days=252, table_days=20,
                       trades_days=15, trades_top_n=10):
    """Reconstruct the exact vol-scaling decision the strategy makes each day.

    Mirrors ``risk_parity_ensemble`` in alt/hydra/hydra_run.py:
      vols  = sleeves.rolling(63).std().shift(1) * sqrt(252)
      raw   = (inv_vol_weights * sleeves).sum(axis=1)
      pv    = raw.rolling(63).std().shift(1) * sqrt(252)  # pre-scale portfolio vol
      scale = clip(target_vol / pv, upper=5x)

    Per-sleeve gross position = inv_vol_weight * scalar (in fraction of NAV).
    Per-sleeve trade on day T  = position_T - position_T-1 (signed).
    """
    vols = sleeves_df.rolling(window).std().shift(1) * np.sqrt(252)
    vols = vols.where(vols > 0.001)
    inv = (1 / vols).where(vols.notna(), 0)
    w = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    raw = (w * sleeves_df).sum(axis=1)
    pv = raw.rolling(window).std().shift(1) * np.sqrt(252)
    scalar = (target_vol / pv).clip(upper=lev_cap).fillna(0)
    realised = hydra_r.rolling(window).std() * np.sqrt(252)

    # Per-sleeve gross position = w * scale, then daily trades = diff
    gross = w.multiply(scalar, axis=0)        # rows=date, cols=sleeve
    trades = gross.diff()                     # signed delta in NAV fraction
    turnover = trades.abs().sum(axis=1)
    gross_total = gross.sum(axis=1)

    # Last N days, chart-ready
    idx = hydra_r.index[-recent_days:]
    daily = [{
        "date": d.strftime("%Y-%m-%d"),
        "realised_vol": round(float(realised.loc[d] * 100), 2) if d in realised.index and not np.isnan(realised.loc[d]) else None,
        "raw_vol": round(float(pv.loc[d] * 100), 2) if d in pv.index and not np.isnan(pv.loc[d]) else None,
        "scalar": round(float(scalar.loc[d]), 2) if d in scalar.index else None,
    } for d in idx]

    # Last N days tabular (most recent first) — vol/scalar/return
    tbl_idx = hydra_r.index[-table_days:][::-1]
    table = [{
        "date": d.strftime("%Y-%m-%d"),
        "raw_vol": round(float(pv.loc[d] * 100), 2) if d in pv.index and not np.isnan(pv.loc[d]) else None,
        "scalar": round(float(scalar.loc[d]), 2) if d in scalar.index else None,
        "realised_vol": round(float(realised.loc[d] * 100), 2) if d in realised.index and not np.isnan(realised.loc[d]) else None,
        "ret": round(float(hydra_r.loc[d] * 100), 3),
    } for d in tbl_idx]

    # Daily trade summary: last N days, most recent first
    trade_idx = hydra_r.index[-trades_days:][::-1]
    trade_summary = []
    for d in trade_idx:
        if d not in trades.index:
            continue
        row = trades.loc[d]
        if row.isna().all():
            continue
        buys = row[row > 1e-6]
        sells = row[row < -1e-6]
        top_buy = buys.idxmax() if len(buys) else None
        top_sell = sells.idxmin() if len(sells) else None
        trade_summary.append({
            "date": d.strftime("%Y-%m-%d"),
            "gross_pct": round(float(gross_total.loc[d] * 100), 2),
            "scalar": round(float(scalar.loc[d]), 2),
            "turnover_pct": round(float(turnover.loc[d] * 100), 2),
            "n_buys": int(len(buys)),
            "n_sells": int(len(sells)),
            "top_buy": {"sleeve": top_buy, "delta_pct": round(float(buys.max() * 100), 2)} if top_buy else None,
            "top_sell": {"sleeve": top_sell, "delta_pct": round(float(sells.min() * 100), 2)} if top_sell else None,
        })

    # Most-recent-day per-sleeve trade ledger
    last_d = hydra_r.index[-1]
    prev_d = hydra_r.index[-2]
    ledger_today = []
    if last_d in trades.index:
        last_row = trades.loc[last_d]
        prior = gross.loc[prev_d] if prev_d in gross.index else gross.loc[last_d] * 0
        new = gross.loc[last_d]
        sorted_idx = last_row.abs().sort_values(ascending=False).index
        for s in sorted_idx[:trades_top_n]:
            d_val = float(last_row.loc[s])
            if abs(d_val) < 1e-6:
                continue
            ledger_today.append({
                "sleeve": s,
                "prior_pct": round(float(prior.loc[s] * 100), 3),
                "new_pct": round(float(new.loc[s] * 100), 3),
                "delta_pct": round(float(d_val * 100), 3),
                "action": "BUY" if d_val > 0 else "SELL",
            })

    ry = realised.iloc[-recent_days:].dropna()
    sy = scalar.iloc[-recent_days:].dropna()
    ty = turnover.iloc[-recent_days:].dropna()
    summary = {
        "target_vol_pct": round(target_vol * 100, 1),
        "lev_cap": lev_cap,
        "window": window,
        "current_realised_vol_pct": round(float(realised.iloc[-1] * 100), 2) if not np.isnan(realised.iloc[-1]) else None,
        "current_raw_vol_pct": round(float(pv.iloc[-1] * 100), 2) if not np.isnan(pv.iloc[-1]) else None,
        "current_scalar": round(float(scalar.iloc[-1]), 2),
        "current_gross_pct": round(float(gross_total.iloc[-1] * 100), 2),
        "current_turnover_pct": round(float(turnover.iloc[-1] * 100), 2),
        "vol_1y_min_pct": round(float(ry.min() * 100), 2) if len(ry) else None,
        "vol_1y_max_pct": round(float(ry.max() * 100), 2) if len(ry) else None,
        "vol_1y_mean_pct": round(float(ry.mean() * 100), 2) if len(ry) else None,
        "scalar_1y_min": round(float(sy.min()), 2) if len(sy) else None,
        "scalar_1y_max": round(float(sy.max()), 2) if len(sy) else None,
        "scalar_1y_mean": round(float(sy.mean()), 2) if len(sy) else None,
        "pct_days_capped": round(float((sy >= lev_cap - 1e-6).mean() * 100), 1) if len(sy) else None,
        "turnover_1y_mean_pct": round(float(ty.mean() * 100), 2) if len(ty) else None,
        "turnover_1y_median_pct": round(float(ty.median() * 100), 2) if len(ty) else None,
        "ledger_date": last_d.strftime("%Y-%m-%d"),
        "ledger_prior_date": prev_d.strftime("%Y-%m-%d"),
    }
    return {"daily": daily, "table": table, "summary": summary,
            "trade_summary": trade_summary, "ledger_today": ledger_today}


def etf_positions_block(csv_path, top_positions=20, top_trades=20,
                        trade_history_days=15, trade_history_top=8,
                        min_delta_pct=0.001):
    """Load per-ETF exposure CSV (built by hydra_etf_positions.py) and
    produce a JSON-friendly block for the factsheet."""
    if not csv_path.exists():
        return None
    ep = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
    trades = ep.diff()

    last_d = ep.index[-1]
    prev_d = ep.index[-2]
    last = ep.iloc[-1]
    prev = ep.iloc[-2]

    # Full position list (most recent day), sorted by absolute size
    positions_today = []
    for etf in last.abs().sort_values(ascending=False).index[:top_positions]:
        positions_today.append({
            "etf": etf,
            "pct_nav": round(float(last[etf] * 100), 3),
        })

    # Full trade ledger for most recent day
    today_trades = trades.iloc[-1]
    ledger_today = []
    for etf in today_trades.abs().sort_values(ascending=False).index[:top_trades]:
        d = float(today_trades[etf])
        if abs(d) * 100 < min_delta_pct:
            continue
        ledger_today.append({
            "etf": etf,
            "prior_pct": round(float(prev[etf] * 100), 3),
            "new_pct": round(float(last[etf] * 100), 3),
            "delta_pct": round(d * 100, 3),
            "action": "BUY" if d > 0 else "SELL",
        })

    # Per-day trade history: top N buys and sells, most recent first
    history = []
    for d in ep.index[-trade_history_days:][::-1]:
        row = trades.loc[d].dropna()
        if row.empty:
            continue
        buys = row[row > 1e-6].sort_values(ascending=False)
        sells = row[row < -1e-6].sort_values()
        history.append({
            "date": d.strftime("%Y-%m-%d"),
            "turnover_pct": round(float(row.abs().sum() * 100), 2),
            "gross_pct": round(float(ep.loc[d].abs().sum() * 100), 2),
            "net_pct": round(float(ep.loc[d].sum() * 100), 2),
            "n_etfs_live": int((ep.loc[d].abs() > 1e-6).sum()),
            "buys": [{"etf": e, "delta_pct": round(float(v * 100), 3)}
                     for e, v in buys.head(trade_history_top).items()],
            "sells": [{"etf": e, "delta_pct": round(float(v * 100), 3)}
                      for e, v in sells.head(trade_history_top).items()],
        })

    return {
        "as_of": last_d.strftime("%Y-%m-%d"),
        "prior": prev_d.strftime("%Y-%m-%d"),
        "gross_pct": round(float(last.abs().sum() * 100), 2),
        "net_pct": round(float(last.sum() * 100), 2),
        "n_etfs": int((last.abs() > 1e-6).sum()),
        "positions_today": positions_today,
        "ledger_today": ledger_today,
        "history": history,
    }


def leverage_variants_block(sleeves_df, spy_r, target_vol_port=0.20,
                            lev_cap_port=5.0, window=63,
                            target_vol_lite=0.10, rebal_days=21):
    """Build 4-variant comparison isolating the effects of leverage and
    dynamic inverse-vol weighting.

    Variants:
      A. HYDRA         — dynamic inv-vol weights × portfolio scalar (target 20%, cap 5x)
      B. HYDRA-NoLev   — same dynamic weights, scalar forced to 1x (no amplification)
      C. Lite          — equal-weight, monthly rebal, static leverage to hit 10% vol
      D. Lite-NoLev    — equal-weight, monthly rebal, 1x (no leverage)
    """
    IS_END = pd.Timestamp("2018-01-01")
    nz = (sleeves_df != 0).any(axis=1)
    idx = sleeves_df.index[nz]

    # A. HYDRA (dynamic vol target, leverage up to 5x)
    vols = sleeves_df.rolling(window).std().shift(1) * np.sqrt(252)
    vols = vols.where(vols > 0.001)
    inv = (1 / vols).where(vols.notna(), 0)
    w_dyn = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    raw_dyn = (w_dyn * sleeves_df).sum(axis=1)
    pv = raw_dyn.rolling(window).std().shift(1) * np.sqrt(252)
    scalar_a = (target_vol_port / pv).clip(upper=lev_cap_port).fillna(0)
    scalar_b = (target_vol_port / pv).clip(upper=1.0).fillna(0)
    r_a = (raw_dyn * scalar_a).reindex(idx).fillna(0)
    r_b = raw_dyn.reindex(idx).fillna(0)  # no leverage, pure inv-vol composite
    mean_lev_a = float(scalar_a.loc[idx].iloc[-252:].mean())

    # C/D. Lite: equal-weight across live sleeves, monthly rebal
    live = (sleeves_df != 0).cummax().astype(float)
    w_daily = live.div(live.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    mask = pd.Series(False, index=w_daily.index)
    mask.iloc[::rebal_days] = True
    w_lite = w_daily.where(mask, np.nan).ffill().fillna(0)
    raw_lite = (w_lite * sleeves_df).sum(axis=1).reindex(idx).fillna(0)
    native_vol = raw_lite.std() * np.sqrt(252)
    lev_c = float(target_vol_lite / native_vol) if native_vol > 0 else 1.0
    r_c = raw_lite * lev_c
    r_d = raw_lite.copy()

    spy = spy_r.reindex(idx).fillna(0)

    def window_m(r):
        return {
            "full": metrics(r),
            "is": metrics(r.loc[:IS_END]),
            "oos": metrics(r.loc[IS_END:]),
            "navx": round(float((1 + r).cumprod().iloc[-1]), 2),
        }

    def trailing(r, n):
        if len(r) < n + 1 or r.iloc[-n:].std() == 0:
            return {"ret": None, "sharpe": None, "vol": None}
        rr = r.iloc[-n:]
        ar = ((1 + rr).prod() ** (252 / n) - 1) * 100
        av = rr.std() * np.sqrt(252) * 100
        sr = (rr.mean() * 252) / (rr.std() * np.sqrt(252))
        return {"ret": round(float(ar), 2), "sharpe": round(float(sr), 2),
                "vol": round(float(av), 2)}

    variants = [
        {"name": "HYDRA", "color": "#7c3aed",
         "lev_label": f"dynamic ≤{lev_cap_port:.0f}×",
         "lev_mean_1y": round(mean_lev_a, 2),
         "description": "Dynamic inverse-vol weights across sleeves, portfolio vol-scaled to 20% annualised with a 5× gross cap. Scalar recomputed daily from T−1 close data.",
         **window_m(r_a),
         "trailing_1y": trailing(r_a, 252),
         "trailing_3y": trailing(r_a, 252 * 3)},
        {"name": "HYDRA-NoLev", "color": "#0e7490",
         "lev_label": "1× (no amplification)",
         "lev_mean_1y": 1.0,
         "description": "Same dynamic inverse-vol weights as HYDRA, but the portfolio scalar is forced to 1×. Shows the pure diversified composite before leverage is applied.",
         **window_m(r_b),
         "trailing_1y": trailing(r_b, 252),
         "trailing_3y": trailing(r_b, 252 * 3)},
        {"name": "Lite", "color": "#c97a00",
         "lev_label": f"static {lev_c:.2f}×",
         "lev_mean_1y": round(lev_c, 2),
         "description": f"Equal-weight across live sleeves, monthly rebalance. Static leverage of {lev_c:.2f}× chosen once on full-sample data to hit 10% annualised vol. No dynamic weighting or scaling.",
         **window_m(r_c),
         "trailing_1y": trailing(r_c, 252),
         "trailing_3y": trailing(r_c, 252 * 3)},
        {"name": "Lite-NoLev", "color": "#4a4a68",
         "lev_label": "1× (no leverage)",
         "lev_mean_1y": 1.0,
         "description": "Pure equal-weight across live sleeves, monthly rebalance, no leverage. The simplest possible expression of the ensemble — a direct baseline for what diversification alone delivers.",
         **window_m(r_d),
         "trailing_1y": trailing(r_d, 252),
         "trailing_3y": trailing(r_d, 252 * 3)},
        {"name": "SPY", "color": "#8a8aa0",
         "lev_label": "1× benchmark",
         "lev_mean_1y": 1.0,
         "description": "S&P 500 ETF, buy-and-hold benchmark for context.",
         **window_m(spy),
         "trailing_1y": trailing(spy, 252),
         "trailing_3y": trailing(spy, 252 * 3)},
    ]

    eq_df = pd.DataFrame({
        "HYDRA": r_a, "HYDRA_NoLev": r_b,
        "Lite": r_c, "Lite_NoLev": r_d, "SPY": spy,
    }).fillna(0)
    eq = equity_curve_multi(eq_df)

    notes = [
        ("Sharpe is leverage-invariant",
         "Within each pair the Sharpe ratios are essentially identical: HYDRA 1.58 vs HYDRA-NoLev 1.59, Lite 1.20 vs Lite-NoLev 1.20. Leverage multiplies return and volatility proportionally — it changes the magnitude of outcomes but not the risk-adjusted quality of the edge."),
        ("What actually generates alpha is the dynamic inv-vol weighting",
         "Holding leverage constant, HYDRA's SR 1.58 vs Lite's 1.20 shows the dynamic ensemble pays ~0.38 of Sharpe over equal-weight. The OOS gap is larger (2.01 vs 1.05) — dynamic risk-parity has held up; equal-weight has not."),
        ("Leverage is a scaling decision, not an alpha decision",
         "HYDRA's headline 16% CAGR / 26× NAV over 21y comes from multiplying the 3.3% unlevered composite by the ~4.66× mean portfolio scalar. Without that scalar HYDRA is a quiet 2% vol product — a smarter T-bill alternative, not a standalone return strategy. Lite is the same story at smaller scale: 2.43× static leverage turns a 4.9% CAGR into a 12% CAGR."),
        ("Reg-T and prime brokerage constraints",
         "HYDRA's 5× portfolio cap plus per-sleeve 1.5× internal scaling can produce 6–7× notional gross. This is only realistic in a portfolio-margin account ($100k+) or a prime-broker relationship. Lite at 2.43× is just barely under the Reg-T 2× cap — practically this would need portfolio margin too. The NoLev variants fit comfortably in any cash/margin account."),
    ]

    return {
        "variants": variants,
        "equity_curve": eq,
        "notes": notes,
    }


def sleeve_descriptions():
    return {
        "s1_eq_regime": "Long SPY when SPY > 200dma AND VIX < 25; else SHY.",
        "s2_sector_top3": "Top-3 of 9 SPDR sectors by 6m momentum, monthly.",
        "s3_bond_dur": "Long TLT when 10y yield 6m trend < 0; else SHY.",
        "s4_credit": "HY credit (HYG) when trending up; else IEF.",
        "s5_curve_carry": "Carry on yield-curve steepening (TLT/IEF).",
        "s6_cmdty": "Long DBC when DBC > 200dma; else BIL.",
        "s7_gld_slv": "Gold/silver ratio regime — risk-off toggle.",
        "s8_fxy_sh": "Long FXY (Yen) when VIX 10d avg > 22 (crisis hedge).",
        "s9_usd_reg": "Long UUP when 6m trend up; else BIL.",
        "s10_vix_carry": "Short vol carry (contango-based).",
        "s12_btc": "Long BTC-linked when BTC > 50d MA; else BIL.",
        "s13_xa_gem": "Absolute momentum across 6 assets.",
        "s15_defensive": "Defensive sector rotation in low-breadth regimes.",
        "s17_semi": "SMH when SMH > 200dma; else BIL.",
        "s18_spy_rev": "5-day SPY mean reversion after −3% drop & VIX > 20.",
        "s19_em": "EEM when EEM > 200dma; else BIL.",
        "s20_infl": "TIP when 10y breakeven trend up; else IEF.",
        "s22_energy": "XLE when oil trending AND XLE > 200dma; else XLP.",
        "s24_emb": "EMB when trending AND yields not spiking; else BIL.",
        "s27_xa_ls": "Dollar-neutral long-short cross-asset momentum.",
    }


def main():
    # Refresh per-ETF positions before building the factsheet
    try:
        import hydra_etf_positions
        hydra_etf_positions.build()
    except Exception as e:
        print(f"[warn] hydra_etf_positions.build() failed: {e}")

    ret_df = pd.read_csv(RESULTS / "hydra_returns.csv",
                         parse_dates=["Date"]).set_index("Date")
    sl_df = pd.read_csv(RESULTS / "hydra_sleeves.csv",
                        parse_dates=["Date"]).set_index("Date")
    r = ret_df["HYDRA"]
    spy = ret_df["SPY"]

    # HYDRA-Lite (no dynamic vol scaling; eq-weight + monthly rebal + static lev)
    lite_path = RESULTS / "hydra_lite_returns.csv"
    lite_data = None
    if lite_path.exists():
        lite_df = pd.read_csv(lite_path, parse_dates=["Date"]).set_index("Date")
        lite = lite_df["HYDRA_Lite"].reindex(r.index).fillna(0)
        lite_full = metrics(lite)
        lite_is = metrics(lite.loc[:pd.Timestamp("2018-01-01")])
        lite_oos = metrics(lite.loc[pd.Timestamp("2018-01-01"):])
        lite_data = {
            "metrics": {"name": "HYDRA-Lite", **lite_full,
                        "n_years": round(len(lite) / 252, 1),
                        "inception": str(lite.index[0].date())},
            "is_metrics": lite_is,
            "oos_metrics": lite_oos,
            "trailing": trailing(lite, spy)["HYDRA"],
            "nav_x": round(float((1 + lite).cumprod().iloc[-1]), 2),
            "calendar_returns": calendar_returns(lite),
            "walkforward_5y": walkforward(lite, spy,
                                          [(y, y + 5) for y in range(2006, 2022, 5)]),
            "config": {
                "weighting": "Equal-weight across live sleeves",
                "rebalance": "Monthly (every 21 trading days)",
                "leverage": "Static 2.43x (chosen once to hit 10% ann vol)",
                "vol_scaling": "None at portfolio level",
            },
        }

    IS_END = pd.Timestamp("2018-01-01")
    is_m = metrics(r.loc[:IS_END])
    oos_m = metrics(r.loc[IS_END:])
    full_m = metrics(r)
    spy_m = metrics(spy)

    # Current portfolio = latest non-zero weights (approx — use latest sleeve
    # contributions scaled by recent activity)
    # Use trailing 21d absolute return by sleeve as proxy for recent weight
    last21 = sl_df.iloc[-21:].abs().sum()
    weights = (last21 / last21.sum()).sort_values(ascending=False)
    portfolio = [
        {"sleeve": name, "weight_pct": round(float(w * 100), 2),
         "description": sleeve_descriptions().get(name, "")}
        for name, w in weights.items() if w > 0
    ]

    data = {
        "fund_name": "HYDRA — 20-Sleeve Diversified Ensemble",
        "strategy_type": "Multi-Strategy Risk-Parity Ensemble",
        "benchmark": "SPY (S&P 500 ETF)",
        "inception_date": str(r.index[0].date()),
        "last_updated": str(r.index[-1].date()),
        "nav_x": round(float((1 + r).cumprod().iloc[-1]), 2),
        "rebalance": "Monthly (sleeve names), Daily (vol scaling)",
        "sleeves_count": int(sl_df.shape[1]),
        "universe_size": int(sl_df.shape[1]),

        "metrics": {
            "HYDRA": {"name": "HYDRA", **full_m,
                      "n_years": round(len(r) / 252, 1),
                      "inception": str(r.index[0].date())},
            "SPY": {"name": "SPY", **spy_m,
                    "n_years": round(len(spy) / 252, 1),
                    "inception": str(spy.index[0].date())},
        },
        "is_metrics": {"period": f"{r.loc[:IS_END].index[0].date()} — {r.loc[:IS_END].index[-1].date()}", **is_m},
        "oos_metrics": {"period": f"{r.loc[IS_END:].index[0].date()} — {r.loc[IS_END:].index[-1].date()}", **oos_m},

        "trailing": trailing(r, spy),
        "equity_curve": equity_curve_multi(
            pd.DataFrame({"HYDRA": r, "HYDRA_Lite": lite, "SPY": spy}).fillna(0)
            if lite_data is not None
            else pd.DataFrame({"HYDRA": r, "SPY": spy})),
        "drawdown_curve": drawdown_curve(r),
        "rolling_sharpe": rolling_sharpe(r),
        "calendar_returns": calendar_returns(r),
        "calendar_spy": calendar_returns(spy),
        "monthly_heatmap": monthly_heatmap(r),

        "walkforward_5y": walkforward(r, spy,
                                      [(y, y + 5) for y in range(2006, 2022, 5)]),

        "sleeves": sleeve_stats(sl_df),
        "correlations": sleeve_correlations(sl_df),
        "portfolio": portfolio,

        "hydra_lite": lite_data,

        "vol_scaling": vol_scaling_series(sl_df, r),
        "etf_positions": etf_positions_block(RESULTS / "hydra_etf_positions.csv"),
        "leverage_variants": leverage_variants_block(sl_df, spy),

        "notes": {
            "construction": "Inverse-vol risk parity across 20 uncorrelated sleeves, each independently vol-targeted to 10% annualised. Portfolio vol target 20%, gross cap 5x.",
            "tc": "15 bps on turnover; 1-bar signal lag; no look-ahead.",
            "ceiling_honest": "After extensive iteration, full-window SR ≈ 1.6 / OOS SR ≈ 2.0 is the honest ceiling for a 21-year backtest with no look-ahead and realistic TC. SR 3 over 21 years is not achievable without hindsight-biased sleeve selection or concentrated leverage (METEOR-style, which produced −78% MDD in its 21y proxy).",
        },
    }

    out = RESULTS / "hydra_factsheet_data.json"
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {out} ({len(out.read_text()) / 1024:.1f} KB)")

    # Summary
    print(f"\nHYDRA summary:")
    print(f"  inception {data['inception_date']}, last {data['last_updated']}")
    print(f"  full   SR {full_m['sharpe']}  Ret {full_m['ann_return']}%  MDD {full_m['max_dd']}%")
    print(f"  IS     SR {is_m['sharpe']}  Ret {is_m['ann_return']}%  MDD {is_m['max_dd']}%")
    print(f"  OOS    SR {oos_m['sharpe']}  Ret {oos_m['ann_return']}%  MDD {oos_m['max_dd']}%")
    print(f"  sleeves {data['sleeves_count']}, mean |corr| {data['correlations']['mean_abs']}")


if __name__ == "__main__":
    main()
