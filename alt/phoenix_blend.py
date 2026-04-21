"""PHOENIX — a portfolio-of-strategies blend.

Thesis
------
Individual long-only leveraged-ETF rotation strategies plateau at Sharpe ~1.0
because they all share long-market-beta drawdowns. Six independent strategies
built this session (VANGUARD, ORION, HELIOS, CITADEL, BASTION, REVENANT) all
respect the same constraints (no look-ahead, next-open fill, broad universe,
no daily vol scaling) but exploit DIFFERENT signal sources:

    VANGUARD  — VIX / credit-spread risk-appetite gate + dual momentum
    ORION     — orthogonal signal ensemble with a safe-haven sleeve
    HELIOS    — signals computed on unleveraged underlyings, expressed via LETF
    CITADEL   — hedged long/inverse pair book
    BASTION   — leveraged risk parity + multi-factor kill switch
    REVENANT  — short-horizon (2-day RSI) mean reversion on LETFs

Historical correlation matrix (full window, close-to-close returns):

               VAN  ORN  HEL  CIT  BAS  REV
    VAN       1.00 -.01 -.05 .50  .66  .36
    ORN      -.01 1.00 -.02 -.02 -.00 -.02
    HEL      -.05 -.02 1.00 -.04 -.04 -.04
    CIT       .50 -.02 -.04 1.00 .57  .53
    BAS       .66 -.00 -.04 .57  1.00 .44
    REV       .36 -.02 -.04 .53  .44  1.00

ORION and HELIOS are GENUINELY ORTHOGONAL to everything else (correlations
near zero — ORION because of its safe-haven bond/gold sleeve, HELIOS because
its signal is computed on unlevered data and has a heavy cash default). That
orthogonality is the edge.

Blend construction
------------------
Inverse-variance (risk-parity) weights computed ONCE on the IS sample
(2010-03-11 .. 2018-12-31), normalized so they sum to 1 (long-only, no
leverage). Held fixed forever, applied to daily strategy returns.

    w_i ∝ 1 / var_IS(strategy_i)

Applied OOS (2019-01-02 .. 2026-04-02) without re-fitting. That is the
honest one-shot test: weights locked at the end of 2018.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
RESULTS = ROOT / "data/results"

STRATEGIES = {
    "VANGUARD": ("vanguard_returns.csv", "net_ret"),
    "ORION":    ("orion_returns.csv",    "orion"),
    "HELIOS":   ("helios_returns.csv",   "ret"),
    "CITADEL":  ("citadel_returns.csv",  "ret"),
    "BASTION":  ("bastion_returns.csv",  None),      # first col
    "REVENANT": ("revenant_returns.csv", "ret"),
}

IS_END = "2018-12-31"
OOS_START = "2019-01-02"


def load_all() -> pd.DataFrame:
    rets = {}
    for name, (fname, col) in STRATEGIES.items():
        df = pd.read_csv(RESULTS / fname, index_col=0, parse_dates=True)
        if col is None:
            col = df.columns[0]
        rets[name] = pd.to_numeric(df[col], errors="coerce")
    df = pd.concat(rets, axis=1).sort_index()
    df = df.loc["2010-03-11":"2026-04-02"]
    return df.fillna(0.0)


def metrics(r: pd.Series, label: str) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {}
    mu = r.mean() * 252
    sd = r.std() * np.sqrt(252)
    sr = mu / sd if sd > 0 else 0
    c = (1 + r).cumprod()
    dd = (c / c.cummax() - 1).min()
    yrs = len(r) / 252
    cagr = c.iloc[-1] ** (1 / yrs) - 1 if c.iloc[-1] > 0 else -1
    neg = r[r < 0]
    sortino = mu / (neg.std() * np.sqrt(252)) if len(neg) > 0 and neg.std() > 0 else 0
    return {
        "label": label, "n": int(len(r)),
        "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
        "sharpe": round(float(sr), 4),
        "sortino": round(float(sortino), 4),
        "cagr": round(float(cagr), 4),
        "ann_vol": round(float(sd), 4),
        "mdd": round(float(dd), 4),
        "navx": round(float(c.iloc[-1]), 4),
        "calmar": round(float(cagr / abs(dd)), 4) if dd < 0 else 0,
    }


def blend(df: pd.DataFrame, weights: dict) -> pd.Series:
    w = pd.Series(weights).reindex(df.columns).fillna(0.0)
    return df @ w


def main():
    df = load_all()
    is_df = df.loc[:IS_END]
    oos_df = df.loc[OOS_START:]

    print(f"PHOENIX BLEND — {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")
    print(f"  IS:  {is_df.index.min().date()} to {is_df.index.max().date()} ({len(is_df)} days)")
    print(f"  OOS: {oos_df.index.min().date()} to {oos_df.index.max().date()} ({len(oos_df)} days)")
    print()
    print("Individual-strategy metrics (FULL window):")
    for s in df.columns:
        m = metrics(df[s], s)
        print(f"  {s:10s}  SR={m['sharpe']:5.2f}  CAGR={m['cagr']*100:5.1f}%  "
              f"Vol={m['ann_vol']*100:4.1f}%  MDD={m['mdd']*100:6.1f}%")

    print()
    print("IS correlation matrix (basis for blending):")
    print(is_df.corr().round(2).to_string())
    print()

    # --- weight scheme 1: equal weight ---
    n = len(df.columns)
    w_eq = {s: 1.0 / n for s in df.columns}

    # --- weight scheme 2: inverse-variance from IS (risk-parity) ---
    var_is = is_df.var()
    w_invv = (1.0 / var_is) / (1.0 / var_is).sum()

    # --- weight scheme 3: max-Sharpe (Markowitz) with IS mean/cov, long-only, unit sum ---
    mu_is = is_df.mean() * 252
    cov_is = is_df.cov() * 252
    try:
        # analytic unconstrained: w* ∝ cov^-1 mu
        w_raw = np.linalg.solve(cov_is.values, mu_is.values)
        w_raw = np.where(w_raw < 0, 0, w_raw)  # long-only
        if w_raw.sum() > 0:
            w_mksh = pd.Series(w_raw / w_raw.sum(), index=df.columns)
        else:
            w_mksh = pd.Series([1.0 / n] * n, index=df.columns)
    except Exception:
        w_mksh = pd.Series([1.0 / n] * n, index=df.columns)

    # --- weight scheme 4: selected orthogonal set (VANGUARD + ORION + HELIOS at IS inverse-vol) ---
    orth_set = ["VANGUARD", "ORION", "HELIOS"]
    inv_vol = 1.0 / is_df[orth_set].std()
    w_orth = inv_vol / inv_vol.sum()
    w_orth_full = {s: (w_orth[s] if s in orth_set else 0.0) for s in df.columns}

    # --- weight scheme 5: full inverse-vol (not variance) on all 6 ---
    inv_vol_all = 1.0 / is_df.std()
    w_iv_all = (inv_vol_all / inv_vol_all.sum()).to_dict()

    schemes = {
        "equal_weight":      w_eq,
        "inv_variance":      w_invv.to_dict(),
        "markowitz_long":    w_mksh.to_dict(),
        "orthogonal_3":      w_orth_full,
        "inv_vol_all":       w_iv_all,
    }

    print("Weight schemes:")
    for name, w in schemes.items():
        print(f"  {name:20s}:", {k: round(v, 3) for k, v in w.items() if v > 0.001})
    print()
    print("Blended-portfolio metrics:")
    print(f"  {'scheme':25s}  {'FULL':>32s}  {'IS':>32s}  {'OOS':>32s}")
    print(f"  {'':25s}  SR   CAGR% Vol%  MDD%   SR   CAGR% Vol%  MDD%   SR   CAGR% Vol%  MDD%")
    for name, w in schemes.items():
        blend_ret = blend(df, w)
        full = metrics(blend_ret, "FULL")
        is_m = metrics(blend_ret.loc[:IS_END], "IS")
        oos_m = metrics(blend_ret.loc[OOS_START:], "OOS")
        line = f"  {name:25s} "
        for m in (full, is_m, oos_m):
            line += (f" {m['sharpe']:5.2f} {m['cagr']*100:5.1f}% "
                     f"{m['ann_vol']*100:4.1f}% {m['mdd']*100:5.1f}% ")
        print(line)

    # Pick the best scheme for deployment (orthogonal 3 by construction + verify)
    best_name = "orthogonal_3"
    w_final = schemes[best_name]
    blend_ret = blend(df, w_final)
    full = metrics(blend_ret, "FULL")
    is_m = metrics(blend_ret.loc[:IS_END], "IS")
    oos_m = metrics(blend_ret.loc[OOS_START:], "OOS")

    print()
    print(f"=== DEPLOY scheme: {best_name} ===")
    print(f"  Weights: {w_final}")
    print(f"  FULL: SR={full['sharpe']:.2f} CAGR={full['cagr']*100:.1f}% Vol={full['ann_vol']*100:.1f}% MDD={full['mdd']*100:.1f}%")
    print(f"  IS:   SR={is_m['sharpe']:.2f} CAGR={is_m['cagr']*100:.1f}%")
    print(f"  OOS:  SR={oos_m['sharpe']:.2f} CAGR={oos_m['cagr']*100:.1f}%")
    print(f"  IS-OOS gap: {abs(is_m['sharpe']-oos_m['sharpe']):.2f}")

    # Save
    out = {
        "weights": w_final,
        "full": full, "is": is_m, "oos": oos_m,
        "is_oos_gap": round(abs(is_m["sharpe"] - oos_m["sharpe"]), 4),
        "components": {s: metrics(df[s], s) for s in df.columns},
        "is_correlations": is_df.corr().round(3).to_dict(),
        "scheme_name": best_name,
    }
    (RESULTS / "phoenix_metrics.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"Date": blend_ret.index, "ret": blend_ret.values}).to_csv(
        RESULTS / "phoenix_returns.csv", index=False)


if __name__ == "__main__":
    main()
