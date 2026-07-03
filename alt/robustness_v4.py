"""Regenerate the PHOENIX robustness suite honestly against v4.

The v2-era suite (bootstrap SR 2.31 etc.) was computed on return series the
2026-07 audits showed to be artifacts; it was removed from
phoenix_v2_audit.json. This script rebuilds every test against the v4
system (equal-weight-active allocator, gross<=1 sleeves, investable crypto)
and writes the `robustness` block back in the exact shapes docs/phoenix.html
renders:

  1. survivorship        — reruns alt/robust_survivorship.py (roster-independent
                           momentum-construction test) and reuses its output.
  2. tc_sensitivity      — extra bps/side scaled by each sleeve's own daily
                           |dW| turnover, re-blended EW-active + overlay.
  3. weight_perturbation — +/-30% multiplicative jitter on the 1/N weights,
                           2000 trials, RAW blend (no overlay), vs the v3
                           walk-forward allocator as the alternative row.
  4. walk_forward        — shipped EW-active vs the v3 walk-forward allocator
                           (both net of overlay), full + 2019+.
  5. bootstrap           — 21d circular block bootstrap, 10k iter, CI and
                           p-value vs the zero-mean null, on v4 net returns.
  6. rulebased_quantum   — repurposed as the QUANTUM-retirement ablation:
                           the debunked in-sample ML series vs the honest
                           walk-forward rebuild, standalone and added to the
                           v4 blend as an 8th sleeve.
  7. extended_2005       — RETIRED (synthetic-LETF builder had documented
                           flaws and is quarantined); an explanation is kept.

Run after phoenix_production.py. Deterministic (seeded).
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "data/results"
ALT = ROOT / "alt"
sys.path.insert(0, str(ALT))

import phoenix_production as prod

IS_END = "2018-12-31"
OOS_START = "2019-01-02"
SEED = 42


def _metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) == 0 or r.std() == 0:
        return {"sharpe": 0.0, "cagr": 0.0, "vol": 0.0, "mdd": 0.0, "navx": 1.0,
                "sortino": 0.0}
    mu, sd = r.mean() * 252, r.std() * np.sqrt(252)
    c = (1 + r).cumprod()
    neg = r[r < 0]
    yrs = len(r) / 252
    return {
        "sharpe": float(mu / sd),
        "cagr": float(c.iloc[-1] ** (1 / yrs) - 1),
        "vol": float(sd),
        "mdd": float((c / c.cummax() - 1).min()),
        "navx": float(c.iloc[-1]),
        "sortino": float(mu / (neg.std() * np.sqrt(252))) if len(neg) and neg.std() > 0 else 0.0,
    }


def _sr(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) and r.std() > 0 else 0.0


def sleeve_turnover() -> pd.DataFrame:
    """Daily sum |dW| per sleeve from each module's decision-dated weights."""
    import vanguard_strategy as van
    import orion_strategy as ori
    import helios_strategy as hel
    import reversal_strategy as rev
    import tom_strategy as tom
    import bondtrend_strategy as bnd
    import phoenix_v2_crypto as cry

    frames = {}
    for tag, W in [
        ("VAN", van.build_weights()),
        ("ORI", ori.build_weights()),
        ("HEL", hel.build_weights()),
        ("CRY", cry.build_weights()),
        ("REV", rev.build_weights()),
        ("TOM", tom.build_weights()),
        ("BND", bnd.build_weights()),
    ]:
        frames[tag] = (W - W.shift(1)).abs().sum(axis=1).fillna(0.0)
    return pd.DataFrame(frames).fillna(0.0)


def main():
    sleeve_df = prod.load_sleeve_returns()
    Wy = prod.build_equal_active_weight_frame(sleeve_df)
    raw = (sleeve_df.loc[Wy.index] * Wy).sum(axis=1)
    net, _ = prod.apply_overlay(raw)
    rng = np.random.default_rng(SEED)

    out = {}

    # ---- 1. survivorship (rerun the standalone script, reuse its JSON) ----
    print("[1/6] survivorship (rerun robust_survivorship.py)...")
    subprocess.run([sys.executable, str(ALT / "robust_survivorship.py")],
                   check=True, capture_output=True)
    sv = json.loads((R / "robustness_survivorship.json").read_text())
    out["survivorship"] = sv

    # ---- 2. TC sensitivity ----
    print("[2/6] tc sensitivity (per-sleeve turnover scaling)...")
    tno = sleeve_turnover().reindex(sleeve_df.index).fillna(0.0)
    rows = []
    for extra in [0, 5, 10, 15, 20, 30]:
        adj = sleeve_df - tno[sleeve_df.columns] * (extra / 1e4)
        r_adj = (adj.loc[Wy.index] * Wy).sum(axis=1)
        n_adj, _ = prod.apply_overlay(r_adj)
        m = _metrics(n_adj)
        drag = float((tno.loc[Wy.index] * Wy).sum(axis=1).mean() * 252 * extra / 1e4)
        rows.append({
            "extra_bps_per_side": extra,
            # base sleeve costs are 5-10 bps/side; report vs the 5 bps floor
            "total_bps_per_side": 5 + extra,
            "ann_drag": round(drag, 4),
            "full_sharpe": m["sharpe"], "full_cagr": m["cagr"],
            "oos_sharpe": _sr(n_adj.loc[OOS_START:]),
            "is_sharpe": _sr(n_adj.loc[:IS_END]),
        })
    out["tc_sensitivity"] = rows

    # ---- 3. weight perturbation (raw blend, no overlay) ----
    print("[3/6] weight perturbation (2000 trials)...")
    full_s, is_s, oos_s = [], [], []
    W_np = Wy.values
    S_np = sleeve_df.loc[Wy.index].values
    idx = Wy.index
    is_mask = idx <= pd.Timestamp(IS_END)
    oos_mask = idx >= pd.Timestamp(OOS_START)
    for _ in range(2000):
        jit = rng.uniform(0.7, 1.3, size=W_np.shape[1])
        Wj = W_np * jit
        Wj = Wj / np.maximum(Wj.sum(axis=1, keepdims=True), 1e-12)
        rj = (S_np * Wj).sum(axis=1)
        rj = pd.Series(rj, index=idx)
        full_s.append(_sr(rj)); is_s.append(_sr(rj[is_mask])); oos_s.append(_sr(rj[oos_mask]))
    pct = lambda a: {str(p): float(np.percentile(a, p)) for p in (5, 25, 50, 75, 95)}
    wf_Wy = prod.build_wf_weight_frame(sleeve_df)
    wf_raw = (sleeve_df.loc[wf_Wy.index] * wf_Wy).sum(axis=1)
    out["weight_perturbation"] = {
        "n_trials": 2000, "perturbation": "+/-30% multiplicative, renormalized",
        "full_sharpe_percentiles": pct(full_s),
        "is_sharpe_percentiles": pct(is_s),
        "oos_sharpe_percentiles": pct(oos_s),
        "full_mean": float(np.mean(full_s)), "oos_mean": float(np.mean(oos_s)),
        "base_full_sharpe": _sr(raw), "base_oos_sharpe": _sr(raw.loc[OOS_START:]),
        "equal_weight_full_sharpe": _sr(wf_raw),   # rendered as the v3 WF row
        "equal_weight_oos_sharpe": _sr(wf_raw.loc[OOS_START:]),
        "note": "RAW blend (no overlay). base = shipped 1/N; the alternative "
                "row is the retired v3 walk-forward allocator.",
    }

    # ---- 4. allocator comparison (shipped EW vs v3 walk-forward), net ----
    print("[4/6] allocator comparison...")
    wf_net, _ = prod.apply_overlay(wf_raw)
    out["walk_forward"] = {
        "static_full_sharpe": _sr(net), "static_oos_sharpe": _sr(net.loc[OOS_START:]),
        "wf_full_sharpe": _sr(wf_net), "wf_oos_sharpe": _sr(wf_net.loc[OOS_START:]),
        "note": "static = shipped v4 equal-weight-active (nothing fitted); "
                "wf = retired v3 trailing-SR budget allocator, same sleeves "
                "and overlay. 1/N wins out-of-era — see PHOENIX_V4_REVIEW.md.",
    }

    # ---- 5. block bootstrap ----
    print("[5/6] block bootstrap (10k x 3 windows)...")
    def boot(r: pd.Series):
        r = r.dropna().values
        n = len(r); nb = max(n // 21, 1)
        obs = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252))
        srs = np.empty(10000); null = np.empty(10000)
        r0 = r - r.mean()
        for i in range(10000):
            starts = rng.integers(0, n, size=nb)
            sel = (starts[:, None] + np.arange(21)[None, :]).ravel() % n
            x = r[sel[:n]]; y = r0[sel[:n]]
            srs[i] = x.mean() / x.std(ddof=1) * np.sqrt(252)
            null[i] = y.mean() / y.std(ddof=1) * np.sqrt(252)
        return {"observed_sharpe": obs,
                "ci_5pct": float(np.percentile(srs, 5)),
                "ci_95pct": float(np.percentile(srs, 95)),
                "se": float(np.std(srs, ddof=1)),
                "p_value": float((null >= obs).mean()),
                "p_value_vs_null": float((null >= obs).mean())}
    out["bootstrap"] = {
        "PHOENIX__FULL": boot(net),
        "PHOENIX__IS": boot(net.loc[:IS_END]),
        "PHOENIX__OOS": boot(net.loc[OOS_START:]),
    }

    # ---- 6. QUANTUM retirement ablation ----
    print("[6/6] QUANTUM ablation...")
    qm = {}
    try:
        qmj = json.loads((R / "quantum_metrics.json").read_text())
        qm = qmj if isinstance(qmj, dict) else {}
    except Exception:
        pass
    q = pd.read_csv(R / "quantum_returns.csv", parse_dates=["Date"]).set_index("Date")
    qcol = [c for c in q.columns if c.lower() in ("ret", "quantum", "net_ret")][0]
    qr = pd.to_numeric(q[qcol], errors="coerce")
    df8 = sleeve_df.copy(); df8["QUA"] = qr.reindex(sleeve_df.index).fillna(0.0)
    prod.SLEEVE_ACTIVATION.setdefault("QUA", pd.Timestamp(prod.BLEND_START))
    W8 = prod.build_equal_active_weight_frame(df8)
    raw8 = (df8.loc[W8.index] * W8).sum(axis=1)
    net8, _ = prod.apply_overlay(raw8)
    prod.SLEEVE_ACTIVATION.pop("QUA", None)
    out["rulebased_quantum"] = {
        # standalone = honest walk-forward QUANTUM rebuild
        "standalone": {"full": _metrics(qr),
                       "is": _metrics(qr.loc[:IS_END]),
                       "oos": _metrics(qr.loc[OOS_START:])},
        # vs_quantum_ml row = what the debunked in-sample series claimed
        "vs_quantum_ml": {"full": {"sharpe": 1.73}, "is": {"sharpe": 2.73},
                          "oos": {"sharpe": 0.87},
                          "note": "the ORIGINAL published QUANTUM series - "
                                  "in-sample XGBoost output (memorization)"},
        "blend_overlayed": {"full": _metrics(net8),
                            "is": _metrics(net8.loc[:IS_END]),
                            "oos": _metrics(net8.loc[OOS_START:])},
        "blend_without": {"full": _metrics(net),
                          "oos": _metrics(net.loc[OOS_START:])},
        "note": "QUANTUM retired in v3/v4: its published 2010-2018 history "
                "was in-sample ML output; rebuilt walk-forward it shows no "
                "robust edge, and adding it back as an 8th equal-weight "
                "sleeve lowers the blend.",
    }

    # ---- 7. extended-2005: retired ----
    out["extended_2005"] = {
        "retired": True,
        "reason": "The 2005+ synthetic-LETF extension used a builder with "
                  "documented flaws (no swap-financing spread, splice jumps, "
                  "ERX built at the wrong leverage) and modeled the OLD "
                  "sleeve roster. It is quarantined from production and its "
                  "numbers are no longer quoted. See alt/PHOENIX_REVIEW.md "
                  "(synthetic-data findings) and alt/PHOENIX_V4_REVIEW.md.",
    }

    audit_p = R / "phoenix_v2_audit.json"
    audit = json.loads(audit_p.read_text())
    audit["robustness"] = out
    audit["robustness_meta"] = {
        "generated": "v4 suite (alt/robustness_v4.py)",
        "series": "phoenix_production v4-equal-active, net of costs",
    }
    audit_p.write_text(json.dumps(audit, separators=(",", ":")))
    print("Wrote v4 robustness block into phoenix_v2_audit.json")
    b = out["bootstrap"]["PHOENIX__FULL"]
    print(f"  bootstrap FULL: SR {b['observed_sharpe']:.2f} "
          f"CI [{b['ci_5pct']:.2f},{b['ci_95pct']:.2f}] p={b['p_value']:.4f}")


if __name__ == "__main__":
    main()
