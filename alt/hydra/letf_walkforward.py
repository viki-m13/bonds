"""Step 9 — walk-forward sub-period validation.

Re-run the top contenders in non-overlapping 3-year windows (plus full-sample)
to see whether the CAGR/Sharpe generalises.

Windows:
  2011-2013, 2014-2016, 2017-2019, 2020-2022, 2023-2026
  plus 2018-2022 (crypto-era) and 2022-2026 (post-crypto-peak stress)

Strategies include the top-3 LETF-only and top-3 LETF+BTC configs from
previous sweeps.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import (common_window_returns, run_backtest, summarise,
                         w_fixed)
from letf_crypto_universe import load_with_crypto


OUT = Path("/home/user/bonds/data/results")

WINDOWS = [
    ("2011-2013", "2011-01-01", "2014-01-01"),
    ("2014-2016", "2014-01-01", "2017-01-01"),
    ("2017-2019", "2017-01-01", "2020-01-01"),
    ("2020-2022", "2020-01-01", "2023-01-01"),
    ("2023-2026", "2023-01-01", "2026-12-31"),
    ("2018-2022 crypto-era", "2018-01-01", "2023-01-01"),
    ("2022-2026 post-peak", "2022-01-01", "2026-12-31"),
    ("Full LETF 2011-2026", "2011-01-01", "2026-12-31"),
    ("Full LETF+BTC 2015-2026", "2015-01-01", "2026-12-31"),
]


def invvol_fn(tickers, lookback):
    def fn(d, hist):
        if len(hist) < lookback + 5: return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0: return None
        inv = 1 / r.std().replace(0, np.nan).fillna(0)
        w = inv / inv.sum()
        out = pd.Series(0.0, index=hist.columns)
        out.loc[w.index] = w
        return out
    return fn


def invvol_scaled_fn(tickers, lookback, target_vol):
    def fn(d, hist):
        if len(hist) < lookback + 5: return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0: return None
        sig = r.std() * np.sqrt(252)
        if (sig <= 0).all(): return None
        inv = 1 / sig.replace(0, np.nan).fillna(0)
        S = inv.sum()
        if S <= 0: return None
        w = inv / S
        c = 1.0 / S
        n = r.shape[1]
        naive = c * np.sqrt(n)
        k = min(target_vol / naive, 5.0) if naive > 0 else 1.0
        w = w * k
        out = pd.Series(0.0, index=hist.columns)
        out.loc[w.index] = w
        return out
    return fn


def run_on_window(cohort_name, rets, start, end, strategies):
    mask = (rets.index >= pd.Timestamp(start)) & (rets.index < pd.Timestamp(end))
    sub = rets.loc[mask]
    if len(sub) < 100:
        return []
    results = []
    for sname, fn_builder in strategies:
        fn = fn_builder()
        r, _ = run_backtest(sub, fn, rebal_days=21, exec_lag=1)
        s = summarise(r, sname)
        s["window"] = cohort_name
        s["cohort"] = cohort_name
        results.append(s)
    return results


def main():
    # Build the full return series once per universe
    px_letf = load_with_crypto([], start="2011-01-01")
    rets_letf = common_window_returns(px_letf)
    px_btc = load_with_crypto(["BTC_USD"], start="2015-01-01")
    rets_btc = common_window_returns(px_btc)
    px_btceth = load_with_crypto(["BTC_USD","ETH_USD"], start="2018-01-01")
    rets_btceth = common_window_returns(px_btceth)

    # Define the shortlist of strategies (each is a closure creating a weights_fn)
    letf_strats = [
        ("EW5 UPRO/TQQQ/SOXL/TMF/UGL",
         lambda: w_fixed({"UPRO":0.2,"TQQQ":0.2,"SOXL":0.2,"TMF":0.2,"UGL":0.2})),
        ("EW6 UPRO/TQQQ/SOXL/TECL/TMF/UGL",
         lambda: w_fixed({"UPRO":1/6,"TQQQ":1/6,"SOXL":1/6,"TECL":1/6,"TMF":1/6,"UGL":1/6})),
        ("3sleeve 70/10/20 TQQQ/TMF/UGL",
         lambda: w_fixed({"TQQQ":0.7,"TMF":0.1,"UGL":0.2})),
        ("3sleeve 50/20/30 TQQQ/TMF/UGL",
         lambda: w_fixed({"TQQQ":0.5,"TMF":0.2,"UGL":0.3})),
        ("theme4 SSO/TQQQ/UBT/UGL 25%",
         lambda: w_fixed({"SSO":0.25,"TQQQ":0.25,"UBT":0.25,"UGL":0.25})),
        ("HFEA 55/45 UPRO/TMF",
         lambda: w_fixed({"UPRO":0.55,"TMF":0.45})),
        ("HFEA-Tech 60/40 TQQQ/TMF",
         lambda: w_fixed({"TQQQ":0.6,"TMF":0.4})),
        ("invvol core6 lb=63",
         lambda: invvol_fn(["UPRO","TQQQ","SOXL","TECL","TMF","UGL"], 63)),
        ("invvol-s core6 lb=63 tv=25%",
         lambda: invvol_scaled_fn(["UPRO","TQQQ","SOXL","TECL","TMF","UGL"], 63, 0.25)),
        ("100% TQQQ",
         lambda: w_fixed({"TQQQ":1.0})),
    ]

    btc_strats = [
        ("EW5 + 20% BTC",
         lambda: w_fixed({"UPRO":0.16,"TQQQ":0.16,"SOXL":0.16,"TMF":0.16,"UGL":0.16,"BTC_USD":0.20})),
        ("EW5 + 10% BTC",
         lambda: w_fixed({"UPRO":0.18,"TQQQ":0.18,"SOXL":0.18,"TMF":0.18,"UGL":0.18,"BTC_USD":0.10})),
        ("TQQQ 60 / TMF 20 / BTC 20",
         lambda: w_fixed({"TQQQ":0.6,"TMF":0.2,"BTC_USD":0.2})),
        ("invvol core6+BTC lb=63",
         lambda: invvol_fn(["UPRO","TQQQ","SOXL","TECL","TMF","UGL","BTC_USD"], 63)),
        ("invvol-s core6+BTC lb=63 tv=25%",
         lambda: invvol_scaled_fn(["UPRO","TQQQ","SOXL","TECL","TMF","UGL","BTC_USD"], 63, 0.25)),
    ]

    btceth_strats = [
        ("EW5 + 15% BTC + 10% ETH",
         lambda: w_fixed({"UPRO":0.15,"TQQQ":0.15,"SOXL":0.15,"TMF":0.15,"UGL":0.15,"BTC_USD":0.15,"ETH_USD":0.10})),
        ("invvol core6+BTC+ETH lb=63",
         lambda: invvol_fn(["UPRO","TQQQ","SOXL","TECL","TMF","UGL","BTC_USD","ETH_USD"], 63)),
        ("invvol-s core6+BTC+ETH lb=63 tv=25%",
         lambda: invvol_scaled_fn(["UPRO","TQQQ","SOXL","TECL","TMF","UGL","BTC_USD","ETH_USD"], 63, 0.25)),
    ]

    rows = []
    for wname, s, e in WINDOWS:
        if "BTC+ETH" in wname or "crypto-era" in wname:
            rets = rets_btceth
            strats = letf_strats + btc_strats + btceth_strats
        elif "BTC" in wname:
            rets = rets_btc
            strats = letf_strats + btc_strats
        else:
            rets = rets_letf
            strats = letf_strats
        rows += run_on_window(wname, rets, s, e, strats)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_walkforward.csv", index=False)
    print(f"Saved {len(df)} rows to letf_walkforward.csv")

    # Pivot: rows=strategy, columns=window, value=CAGR
    piv = df.pivot_table(index="label", columns="window", values="cagr",
                         aggfunc="first")
    # Column order
    col_order = [w[0] for w in WINDOWS if w[0] in piv.columns]
    piv = piv[col_order]
    print("\n=== Walk-forward CAGR by window ===")
    print(piv.round(1).to_string())

    piv_sr = df.pivot_table(index="label", columns="window", values="sharpe",
                            aggfunc="first")[col_order]
    print("\n=== Walk-forward Sharpe by window ===")
    print(piv_sr.round(2).to_string())

    piv_mdd = df.pivot_table(index="label", columns="window", values="mdd",
                             aggfunc="first")[col_order]
    print("\n=== Walk-forward MDD% by window ===")
    print(piv_mdd.round(1).to_string())


if __name__ == "__main__":
    main()
