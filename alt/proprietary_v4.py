"""
Proprietary V4 — Target Sharpe 3 with ZERO vol scaling.

Core bet: Diversification is the only "free lunch". If we build many
low-correlation, positive-Sharpe streams and combine them with static
(frozen) weights, portfolio Sharpe can exceed each stream's Sharpe.

Rules of engagement:
- Monthly rebalance only (21 trading days).
- NO daily vol scaling. NO weekly vol scaling. NO rolling vol targeting.
- Weights are FIXED when a stream is selected: either equal weight,
  or inverse-in-sample-vol, or "one-shot" inverse-vol frozen at
  selection time (uses only data through T-1).
- Selection uses trailing metrics (standard lookback), no look-ahead.

Experiments:
  E1. "Carry Spread Basket" — 20 long-carry + short-rates pairs that
      isolate credit spread. Fixed equal-weight, no regime gate.
  E2. E1 + macro regime gate (HY OAS, SPY 200DMA, VIX).
  E3. E2 + momentum filter (only fund streams with trailing
      63-day return > 0).
  E4. E3 + trailing-Sharpe ranking (top-K, equal weight among top-K).
  E5. E4 + inverse-vol frozen weights (still no daily scaling — the
      weight is fixed at rebalance using in-sample vol up to T-1).
  E6. Add "ballast" streams (MBS, CLOs, floating rate, preferred).
  E7. Final — tune K, lookback, regime thresholds.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/home/user/bonds/data")
ETF = DATA / "etfs"
FRED = DATA / "fred"
RESULTS = Path("/home/user/bonds/alt/results")


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_etf(ticker):
    path = ETF / f"{ticker}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")["Close"]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def load_fred(series):
    path = FRED / f"{series}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    df = pd.to_numeric(df, errors="coerce")
    return df.sort_index()


def load_prices(tickers):
    out = {}
    for t in tickers:
        s = load_etf(t)
        if s is None or len(s) < 252:
            continue
        out[t] = s
    return pd.DataFrame(out)


# ---------------------------------------------------------------------
# Stream construction — a stream is a long-short ETF pair.
# The "stream return" = w_long * r_long + w_short * r_short
# With both legs being long-only ETFs (short = inverse ETF),
# the pair isolates a spread while staying long-only.
# ---------------------------------------------------------------------

def pair_return(long_ret, short_ret, long_wt=0.5, short_wt=0.5):
    """Compute static-weight pair return. No scaling."""
    return long_wt * long_ret + short_wt * short_ret


def build_streams():
    """Define all candidate streams. Each stream is a tuple:
    (name, long_ticker, short_ticker_or_None, long_wt, short_wt, category)
    """
    # Credit-spread streams (long-carry credit vs short-duration treasury)
    credit = [
        ("lqd_tbf",   "LQD",  "TBF", 0.5, 0.5, "credit_ig"),
        ("lqd_tyo",   "LQD",  "TYO", 0.5, 0.5, "credit_ig"),
        ("vcit_tbf",  "VCIT", "TBF", 0.5, 0.5, "credit_ig"),
        ("vcit_tyo",  "VCIT", "TYO", 0.5, 0.5, "credit_ig"),
        ("igib_tbf",  "IGIB", "TBF", 0.5, 0.5, "credit_ig"),
        ("hyg_tbf",   "HYG",  "TBF", 0.5, 0.5, "credit_hy"),
        ("jnk_tbf",   "JNK",  "TBF", 0.5, 0.5, "credit_hy"),
        ("shyg_tbf",  "SHYG", "TBF", 0.5, 0.5, "credit_hy_short"),
        ("ushy_tbf",  "USHY", "TBF", 0.5, 0.5, "credit_hy"),
        ("angl_tbf",  "ANGL", "TBF", 0.5, 0.5, "credit_fa"),
        ("emb_tbf",   "EMB",  "TBF", 0.5, 0.5, "em_debt"),
        ("mbb_tbf",   "MBB",  "TBF", 0.5, 0.5, "mbs"),
        ("mub_tbf",   "MUB",  "TBF", 0.5, 0.5, "muni"),
        ("tip_tbf",   "TIP",  "TBF", 0.5, 0.5, "tips"),
        ("schp_tbf",  "SCHP", "TBF", 0.5, 0.5, "tips"),
    ]
    # Standalone "naturally hedged" or ultra-short duration streams.
    # These have floating-rate or CLO structure so duration is ~0.
    # They earn spread with no treasury hedge needed.
    solo_carry = [
        ("bkln_solo", "BKLN", None, 1.0, 0.0, "floating"),
        ("srln_solo", "SRLN", None, 1.0, 0.0, "floating"),
        ("jaaa_solo", "JAAA", None, 1.0, 0.0, "clo_aaa"),
        ("jbbb_solo", "JBBB", None, 1.0, 0.0, "clo_bbb"),
        ("cloi_solo", "CLOI", None, 1.0, 0.0, "clo_aaa"),
        ("flot_solo", "FLOT", None, 1.0, 0.0, "floating"),
        ("jpst_solo", "JPST", None, 1.0, 0.0, "short_duration"),
        ("mint_solo", "MINT", None, 1.0, 0.0, "short_duration"),
    ]
    # Preferred stock hedged against equity (isolates yield).
    prefs = [
        ("pff_sh",  "PFF", "SH",  0.5, 0.5, "preferred"),
        ("pff_tbf", "PFF", "TBF", 0.5, 0.5, "preferred"),
    ]
    # Dividend equity hedged against broad equity (isolates div yield + defensiveness)
    divs = [
        ("schd_sh", "SCHD", "SH",  0.5, 0.5, "div_eq"),
        ("dvy_sh",  "DVY",  "SH",  0.5, 0.5, "div_eq"),
        ("hdv_sh",  "HDV",  "SH",  0.5, 0.5, "div_eq"),
        ("vig_sh",  "VIG",  "SH",  0.5, 0.5, "div_eq"),
        ("xlu_sh",  "XLU",  "SH",  0.5, 0.5, "div_eq"),
        ("xlp_sh",  "XLP",  "SH",  0.5, 0.5, "div_eq"),
    ]
    # Convertible hedged
    convs = [
        ("cwb_sh", "CWB", "SH", 0.5, 0.5, "convertible"),
    ]
    return credit + solo_carry + prefs + divs + convs


def build_stream_returns(streams, price_df):
    rets = price_df.pct_change().fillna(0)
    out = {}
    meta = {}
    for s in streams:
        name, lg, sh, lw, sw, cat = s
        if lg not in rets.columns:
            continue
        if sh is not None and sh not in rets.columns:
            continue
        r_long = rets[lg]
        r_short = rets[sh] if sh is not None else pd.Series(0, index=rets.index)
        r = pair_return(r_long, r_short, lw, sw)
        # Align to first common valid date
        if sh is not None:
            first_valid = max(r_long.first_valid_index(), r_short.first_valid_index())
        else:
            first_valid = r_long.first_valid_index()
        r = r.loc[first_valid:]
        out[name] = r
        meta[name] = {"long": lg, "short": sh, "lw": lw, "sw": sw, "cat": cat}
    return pd.DataFrame(out), meta


# ---------------------------------------------------------------------
# Regime detection — macro gate. Zero daily vol scaling.
# Signals are computed on day T-1 closes only and used on day T.
# ---------------------------------------------------------------------

def build_regime(dates):
    """Return a Series of regime multipliers (0 or 1, or fractional).
    This gates the whole portfolio on/off based on macro regime.
    NO daily vol scaling — it's a discrete macro gate.
    """
    spy = load_etf("SPY").reindex(dates).ffill()
    hy_oas = load_fred("BAMLH0A0HYM2")
    vix = load_etf("VIXY") or load_etf("VXX")  # proxy for VIX level
    t10y2y = load_fred("T10Y2Y")

    if hy_oas is not None:
        hy_oas = hy_oas.reindex(dates).ffill()
    if t10y2y is not None:
        t10y2y = t10y2y.reindex(dates).ffill()

    # SPY trend regime: above 200-day MA = risk-on
    spy_sma = spy.rolling(200, min_periods=50).mean()
    risk_on = (spy > spy_sma).astype(float)

    # HY spread regime: below 7% = normal, above 9% = crisis
    if hy_oas is not None:
        oas_ok = (hy_oas < 7.5).astype(float)
        # 0.5 weight if 7.5 < oas < 9.5
        oas_partial = ((hy_oas >= 7.5) & (hy_oas < 9.5)).astype(float) * 0.5
        oas_regime = oas_ok + oas_partial
    else:
        oas_regime = pd.Series(1.0, index=dates)

    # Combine: we stay on when EITHER risk-on OR credit spreads tight
    # We dampen when BOTH signals are negative
    combined = pd.concat([risk_on, oas_regime], axis=1).max(axis=1)
    # SHIFT by 1 day — use T-1 info only on day T
    return combined.shift(1).fillna(1.0)


def build_simple_regime(dates, price_df, hy_oas):
    """Simpler regime: just HY OAS gate. Off when HY OAS > threshold."""
    if hy_oas is None:
        return pd.Series(1.0, index=dates)
    oas = hy_oas.reindex(dates).ffill()
    # Binary: off when OAS > 8%
    regime = (oas < 8.0).astype(float)
    return regime.shift(1).fillna(1.0)


# ---------------------------------------------------------------------
# Monthly portfolio construction
# ---------------------------------------------------------------------

def run_backtest(
    stream_rets: pd.DataFrame,
    meta: dict,
    rebalance_days: int = 21,
    lookback: int = 252,
    top_k: int = 15,
    weighting: str = "equal",   # "equal" | "inv_vol" | "sharpe"
    min_sharpe: float = 0.0,
    regime: pd.Series = None,
    momentum_filter: bool = False,
    momentum_lookback: int = 63,
    category_cap: int = None,
    cash_return: float = 0.02,  # annualized fallback when no streams qualify
    tc_bps: float = 5.0,
) -> dict:
    """Run the monthly rebalance backtest.

    Returns dict with 'returns', 'weights_history', 'selected_history'.
    """
    dates = stream_rets.index
    port = pd.Series(0.0, index=dates)
    weights = pd.DataFrame(0.0, index=dates, columns=stream_rets.columns)

    current_wts = pd.Series(0.0, index=stream_rets.columns)
    last_rebal_idx = -rebalance_days  # force first-day rebalance
    selections_log = []

    for i, d in enumerate(dates):
        # On rebalance day (using data up to i-1 only)
        if i - last_rebal_idx >= rebalance_days and i > lookback:
            # Evaluate each stream on trailing window [i-lookback, i-1)
            window = stream_rets.iloc[i - lookback : i]  # excludes i (Python slice)
            mu = window.mean() * 252
            sd = window.std() * np.sqrt(252)
            sr = mu / sd.replace(0, np.nan)

            # Filter: stream must have full history
            mask_full = window.notna().all()
            # Filter: Sharpe > threshold
            mask_sr = sr > min_sharpe
            mask = mask_full & mask_sr

            # Momentum filter: trailing momentum_lookback return > 0
            if momentum_filter and momentum_lookback > 0:
                mom = (1 + stream_rets.iloc[i - momentum_lookback : i]).prod() - 1
                mask = mask & (mom > 0)

            eligible = sr[mask].dropna()
            if len(eligible) == 0:
                new_wts = pd.Series(0.0, index=stream_rets.columns)
            else:
                # Rank by Sharpe, take top_k
                ranked = eligible.sort_values(ascending=False)
                if category_cap is not None:
                    picked = []
                    cat_counts = {}
                    for nm in ranked.index:
                        cat = meta[nm]["cat"]
                        if cat_counts.get(cat, 0) < category_cap:
                            picked.append(nm)
                            cat_counts[cat] = cat_counts.get(cat, 0) + 1
                        if len(picked) >= top_k:
                            break
                    picked = pd.Index(picked)
                else:
                    picked = ranked.head(top_k).index

                if len(picked) == 0:
                    new_wts = pd.Series(0.0, index=stream_rets.columns)
                else:
                    if weighting == "equal":
                        w = pd.Series(1.0 / len(picked), index=picked)
                    elif weighting == "inv_vol":
                        # inverse in-sample vol, FROZEN at rebalance
                        v = sd.loc[picked]
                        inv = 1.0 / v.replace(0, np.nan)
                        w = inv / inv.sum()
                    elif weighting == "sharpe":
                        s = eligible.loc[picked]
                        w = s / s.sum()
                    else:
                        raise ValueError(weighting)
                    new_wts = pd.Series(0.0, index=stream_rets.columns)
                    new_wts.loc[picked] = w.values

            # Apply transaction cost
            turnover = (new_wts - current_wts).abs().sum()
            tc = turnover * (tc_bps / 1e4)
            # Apply cost to today's return
            port.iloc[i] -= tc

            current_wts = new_wts
            last_rebal_idx = i

            # Log selection
            picked_names = current_wts[current_wts > 0].index.tolist()
            selections_log.append({
                "date": str(d.date()),
                "picks": picked_names,
                "weights": {k: float(current_wts[k]) for k in picked_names},
            })

        # Apply current weights today
        weights.iloc[i] = current_wts
        day_ret = (stream_rets.iloc[i] * current_wts).sum()

        # Apply regime gate
        if regime is not None:
            g = regime.get(d, 1.0)
            day_ret = day_ret * g
            # When gated off, earn cash (0% assumed; real cash would be BIL)
            if g < 1.0:
                day_ret = day_ret + (1 - g) * (cash_return / 252)

        port.iloc[i] += day_ret

    return {
        "returns": port,
        "weights": weights,
        "selections": selections_log,
    }


def metrics(ret: pd.Series) -> dict:
    if len(ret) == 0 or ret.std() == 0:
        return {"sharpe": 0, "ann_return": 0, "ann_vol": 0, "max_dd": 0,
                "sortino": 0, "calmar": 0, "total_return": 0, "n_years": 0}
    ann_ret = ret.mean() * 252
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1)
    max_dd = dd.min()
    neg = ret[ret < 0]
    sortino = (ret.mean() * 252) / (neg.std() * np.sqrt(252)) if len(neg) and neg.std() > 0 else float("inf")
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else float("inf")
    return {
        "sharpe": float(sharpe),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "max_dd": float(max_dd),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "total_return": float(cum.iloc[-1] - 1),
        "n_years": float(len(ret) / 252),
    }


# ---------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------

def main():
    # 1. Build universe
    streams = build_streams()
    all_tickers = set()
    for s in streams:
        all_tickers.add(s[1])
        if s[2]:
            all_tickers.add(s[2])
    prices = load_prices(sorted(all_tickers))
    print(f"Loaded {len(prices.columns)} ETFs, {len(prices)} rows.")

    stream_rets, meta = build_stream_returns(streams, prices)
    print(f"Built {len(stream_rets.columns)} streams.")

    # Align to intersecting date range
    start_date = "2012-01-01"  # most ETFs available by then (BKLN, JAAA later)
    stream_rets = stream_rets.loc[start_date:]
    dates = stream_rets.index
    print(f"Date range: {dates[0]} to {dates[-1]}, {len(dates)} days")

    # Report stream-level Sharpes (full sample, for diagnostic only)
    print("\n--- Stream Sharpe (full sample, diagnostic only) ---")
    full_mu = stream_rets.mean() * 252
    full_sd = stream_rets.std() * np.sqrt(252)
    full_sr = (full_mu / full_sd).sort_values(ascending=False)
    for n, s in full_sr.items():
        cat = meta[n]["cat"]
        print(f"  {n:14s} ({cat:16s}) SR={s:.2f}  Ret={full_mu[n]:.2%}  Vol={full_sd[n]:.2%}")

    # Regime data
    hy_oas = load_fred("BAMLH0A0HYM2")
    regime = build_simple_regime(dates, prices, hy_oas)

    # -------------------------------
    # Experiments
    # -------------------------------
    experiments = {}

    configs = [
        # E1: baseline — equal-weight top-K by Sharpe, no regime, no momentum
        dict(name="E1_base",           top_k=15, weighting="equal",   min_sharpe=0.0, regime=None,   momentum_filter=False, category_cap=None),
        # E2: add regime gate
        dict(name="E2_regime",         top_k=15, weighting="equal",   min_sharpe=0.0, regime=regime, momentum_filter=False, category_cap=None),
        # E3: add momentum filter
        dict(name="E3_regime_mom",     top_k=15, weighting="equal",   min_sharpe=0.0, regime=regime, momentum_filter=True,  category_cap=None),
        # E4: category cap for diversification
        dict(name="E4_catcap2",        top_k=15, weighting="equal",   min_sharpe=0.0, regime=regime, momentum_filter=True,  category_cap=2),
        dict(name="E4b_catcap3",       top_k=20, weighting="equal",   min_sharpe=0.0, regime=regime, momentum_filter=True,  category_cap=3),
        # E5: inv-vol weighting (frozen at rebalance)
        dict(name="E5_invvol",         top_k=20, weighting="inv_vol", min_sharpe=0.0, regime=regime, momentum_filter=True,  category_cap=3),
        # E6: higher top_k — take everyone with Sharpe > 0
        dict(name="E6_all_positive",   top_k=50, weighting="inv_vol", min_sharpe=0.0, regime=regime, momentum_filter=True,  category_cap=None),
        # E7: stricter Sharpe
        dict(name="E7_sharpe_gate_05", top_k=20, weighting="inv_vol", min_sharpe=0.5, regime=regime, momentum_filter=True,  category_cap=3),
        dict(name="E7b_sharpe_10",     top_k=20, weighting="inv_vol", min_sharpe=1.0, regime=regime, momentum_filter=True,  category_cap=3),
        # E8: pure equal weight in all streams (no selection, no regime)
        dict(name="E8_all_equal",      top_k=len(stream_rets.columns), weighting="equal", min_sharpe=-99, regime=None, momentum_filter=False, category_cap=None),
        # E9: all streams inverse vol, no regime
        dict(name="E9_all_invvol",     top_k=len(stream_rets.columns), weighting="inv_vol", min_sharpe=-99, regime=None, momentum_filter=False, category_cap=None),
    ]

    for cfg in configs:
        name = cfg.pop("name")
        res = run_backtest(stream_rets, meta, **cfg)
        m = metrics(res["returns"].loc[res["returns"].ne(0).idxmax():])  # from first nonzero
        experiments[name] = {"metrics": m, "config": cfg}
        print(f"\n{name}: Sharpe={m['sharpe']:.3f}  AnnRet={m['ann_return']:.2%}  "
              f"Vol={m['ann_vol']:.2%}  MaxDD={m['max_dd']:.2%}  N={m['n_years']:.1f}y")

    # Save
    with open(RESULTS / "proprietary_v4_experiments.json", "w") as f:
        json.dump(experiments, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS / 'proprietary_v4_experiments.json'}")


if __name__ == "__main__":
    main()
