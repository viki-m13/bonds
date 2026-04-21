"""Step 8 — full sweep including crypto.

For an apples-to-apples comparison, run each strategy FAMILY on two
paired universes:
  (A) LETFs only — 2018-01-01 onward
  (B) LETFs + BTC + ETH — same 2018-01-01 window

Families tested:
  * Fixed recipes (a shortlist of best performers from static sweep)
  * Inv-vol across core basket +/- crypto
  * Momentum top-N +/- crypto
  * Inv-vol scaled to 25/40% naive vol target +/- crypto

We also run a third window with BTC only (2015-2026, longer history).

Output: data/results/letf_sweep_crypto.csv with `has_crypto` column
so you can compare matched pairs.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import (common_window_returns, run_backtest, summarise,
                         w_fixed)
from letf_universe import LETF_LONG_2011
from letf_crypto_universe import load_with_crypto


OUT = Path("/home/user/bonds/data/results")


# --- weight factories duplicated here so this file is self-contained ---

def invvol_fn(tickers, lookback):
    def fn(d, hist):
        if len(hist) < lookback + 5:
            return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0: return None
        vol = r.std()
        inv = 1 / vol.replace(0, np.nan)
        inv = inv.fillna(0)
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


def mom_fn(tickers, lookback, top_n):
    def fn(d, hist):
        if len(hist) < lookback + 5: return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0: return None
        cum = (1 + r).prod() - 1
        picks = cum.sort_values(ascending=False).head(top_n).index.tolist()
        out = pd.Series(0.0, index=hist.columns)
        if picks:
            out.loc[picks] = 1.0 / len(picks)
        return out
    return fn


FIXED_RECIPES = {
    "EW5 UPRO/TQQQ/SOXL/TMF/UGL":
        {"UPRO":0.20,"TQQQ":0.20,"SOXL":0.20,"TMF":0.20,"UGL":0.20},
    "EW6 UPRO/TQQQ/SOXL/TECL/TMF/UGL":
        {"UPRO":1/6,"TQQQ":1/6,"SOXL":1/6,"TECL":1/6,"TMF":1/6,"UGL":1/6},
    "HFEA-Tech 60/40 TQQQ/TMF": {"TQQQ":0.60,"TMF":0.40},
    "HFEA-Tech 80/20 TQQQ/TMF": {"TQQQ":0.80,"TMF":0.20},
    "3sleeve 50/20/30 TQQQ/TMF/UGL": {"TQQQ":0.50,"TMF":0.20,"UGL":0.30},
    "3sleeve 70/10/20 TQQQ/TMF/UGL": {"TQQQ":0.70,"TMF":0.10,"UGL":0.20},
    "3sleeve 80/10/10 TQQQ/TMF/UGL": {"TQQQ":0.80,"TMF":0.10,"UGL":0.10},
    "theme4 SSO/TQQQ/UBT/UGL 25/25/25/25":
        {"SSO":0.25,"TQQQ":0.25,"UBT":0.25,"UGL":0.25},
}

# Crypto-added variants (inject BTC/ETH at specific weights)
FIXED_RECIPES_CRYPTO = {
    # EW5 variants
    "EW5 + 10% BTC":
        {"UPRO":0.18,"TQQQ":0.18,"SOXL":0.18,"TMF":0.18,"UGL":0.18,"BTC_USD":0.10},
    "EW5 + 20% BTC":
        {"UPRO":0.16,"TQQQ":0.16,"SOXL":0.16,"TMF":0.16,"UGL":0.16,"BTC_USD":0.20},
    "EW5 + 10% BTC + 5% ETH":
        {"UPRO":0.17,"TQQQ":0.17,"SOXL":0.17,"TMF":0.17,"UGL":0.17,
         "BTC_USD":0.10,"ETH_USD":0.05},
    "EW5 + 15% BTC + 10% ETH":
        {"UPRO":0.15,"TQQQ":0.15,"SOXL":0.15,"TMF":0.15,"UGL":0.15,
         "BTC_USD":0.15,"ETH_USD":0.10},
    # 3sleeve + crypto
    "3sleeve 50/20/20 TQQQ/TMF/UGL + 10% BTC":
        {"TQQQ":0.50,"TMF":0.20,"UGL":0.20,"BTC_USD":0.10},
    "3sleeve 50/20/20 TQQQ/TMF/UGL + 10% BTC + 5% ETH":
        {"TQQQ":0.45,"TMF":0.20,"UGL":0.20,"BTC_USD":0.10,"ETH_USD":0.05},
    "3sleeve 50/20/10 TQQQ/TMF/UGL + 15% BTC + 5% ETH":
        {"TQQQ":0.50,"TMF":0.20,"UGL":0.10,"BTC_USD":0.15,"ETH_USD":0.05},
    # Concentrated risk-on + crypto
    "TQQQ 60 / TMF 20 / BTC 20":
        {"TQQQ":0.60,"TMF":0.20,"BTC_USD":0.20},
    "TQQQ 50 / TMF 20 / UGL 10 / BTC 15 / ETH 5":
        {"TQQQ":0.50,"TMF":0.20,"UGL":0.10,"BTC_USD":0.15,"ETH_USD":0.05},
}


def run_family(rets, label_prefix, rebal_days=21):
    rows = []
    for name, w in FIXED_RECIPES.items():
        if not set(w).issubset(set(rets.columns)):
            continue
        r, _ = run_backtest(rets, w_fixed(w), rebal_days=rebal_days, exec_lag=1)
        s = summarise(r, f"{label_prefix} :: {name}")
        s["family"] = "fixed"
        s["cohort"] = label_prefix
        rows.append(s)

    # Crypto-bearing fixed recipes only on crypto universe
    for name, w in FIXED_RECIPES_CRYPTO.items():
        if not set(w).issubset(set(rets.columns)):
            continue
        r, _ = run_backtest(rets, w_fixed(w), rebal_days=rebal_days, exec_lag=1)
        s = summarise(r, f"{label_prefix} :: {name}")
        s["family"] = "fixed-crypto"
        s["cohort"] = label_prefix
        rows.append(s)

    # Inv-vol basket variants — core6 (no crypto) vs core6+BTC vs core6+BTC+ETH
    core6 = ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"]
    bases = [("core6", core6)]
    if "BTC_USD" in rets.columns:
        bases.append(("core6+BTC", core6 + ["BTC_USD"]))
    if "BTC_USD" in rets.columns and "ETH_USD" in rets.columns:
        bases.append(("core6+BTC+ETH", core6 + ["BTC_USD","ETH_USD"]))
    for lb in (63, 126):
        for bname, ts in bases:
            if not set(ts).issubset(rets.columns): continue
            r, _ = run_backtest(rets, invvol_fn(ts, lb),
                                rebal_days=rebal_days, exec_lag=1)
            rows.append({**summarise(r, f"{label_prefix} :: invvol {bname} lb={lb}"),
                         "family": "invvol", "cohort": label_prefix})
            for tv in (0.25, 0.40):
                r2, _ = run_backtest(rets, invvol_scaled_fn(ts, lb, tv),
                                     rebal_days=rebal_days, exec_lag=1)
                rows.append({**summarise(r2, f"{label_prefix} :: invvol-s {bname} lb={lb} tv={int(tv*100)}%"),
                             "family": "invvol-scaled", "cohort": label_prefix})

    # Momentum — core6 vs core6+BTC vs core6+BTC+ETH, top-N
    for lb in (63, 126):
        for n in (2, 3, 4):
            for bname, ts in bases:
                if not set(ts).issubset(rets.columns): continue
                r, _ = run_backtest(rets, mom_fn(ts, lb, n),
                                    rebal_days=rebal_days, exec_lag=1)
                rows.append({**summarise(r, f"{label_prefix} :: mom {bname} lb={lb} top{n}"),
                             "family": "mom", "cohort": label_prefix})

    return rows


def main():
    all_rows = []

    # Cohort A: LETFs only from 2018-01-01
    px_a = load_with_crypto([], start="2018-01-01")
    rets_a = common_window_returns(px_a)
    print(f"Cohort A (LETF only, 2018+): {rets_a.shape}")
    all_rows += run_family(rets_a, "LETF-only 2018+")

    # Cohort B: LETF + BTC + ETH from 2018-01-01
    px_b = load_with_crypto(["BTC_USD", "ETH_USD"], start="2018-01-01")
    rets_b = common_window_returns(px_b)
    print(f"Cohort B (LETF+BTC+ETH, 2018+): {rets_b.shape}")
    all_rows += run_family(rets_b, "LETF+BTCETH 2018+")

    # Cohort C: LETF + BTC from 2015-01-01 (longer window)
    px_c = load_with_crypto(["BTC_USD"], start="2015-01-01")
    rets_c = common_window_returns(px_c)
    print(f"Cohort C (LETF+BTC, 2015+): {rets_c.shape}")
    all_rows += run_family(rets_c, "LETF+BTC 2015+")

    # Cohort D: LETF only from 2015-01-01 (paired with C)
    px_d = load_with_crypto([], start="2015-01-01")
    rets_d = common_window_returns(px_d)
    print(f"Cohort D (LETF only, 2015+): {rets_d.shape}")
    all_rows += run_family(rets_d, "LETF-only 2015+")

    df = pd.DataFrame(all_rows).sort_values("cagr", ascending=False).reset_index(drop=True)
    df.to_csv(OUT / "letf_sweep_crypto.csv", index=False)
    print(f"\nSaved {len(df)} rows to letf_sweep_crypto.csv")

    for coh in ["LETF-only 2018+", "LETF+BTCETH 2018+",
                "LETF-only 2015+", "LETF+BTC 2015+"]:
        sub = df[df.cohort == coh].sort_values("cagr", ascending=False)
        print(f"\n=== {coh} — top 8 by CAGR ===")
        for _, r in sub.head(8).iterrows():
            print(f"  {r['label']:70s}  CAGR={r['cagr']:>6.2f}%  "
                  f"Vol={r['vol']:>5.1f}%  MDD={r['mdd']:>7.2f}%  SR={r['sharpe']:>4.2f}")

    # Focused comparison: matching-pairs A vs B (same strategy +/- crypto)
    print("\n=== Paired comparison (2018+ window) ===")
    cohort_a = df[df.cohort == "LETF-only 2018+"].set_index("label")
    cohort_b = df[df.cohort == "LETF+BTCETH 2018+"].set_index("label")
    common = set(x.split(" :: ", 1)[1] for x in cohort_a.index) & \
             set(x.split(" :: ", 1)[1] for x in cohort_b.index)
    for rec in sorted(common):
        la = f"LETF-only 2018+ :: {rec}"
        lb = f"LETF+BTCETH 2018+ :: {rec}"
        if la in cohort_a.index and lb in cohort_b.index:
            a, b = cohort_a.loc[la], cohort_b.loc[lb]
            print(f"  {rec:55s}  A(no crypto) CAGR={a['cagr']:>5.2f}% "
                  f"MDD={a['mdd']:>6.2f}% | B(+crypto) CAGR={b['cagr']:>5.2f}% "
                  f"MDD={b['mdd']:>6.2f}%  ΔCAGR={b['cagr']-a['cagr']:>+5.2f}%")


if __name__ == "__main__":
    main()
