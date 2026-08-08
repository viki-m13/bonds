"""FLOW + ML round — adapting equity-quant methods to the corp dislocation book.

[A] ORDER-FLOW conditioning (novel use of the two-sided customer tape):
    sell-pressure exhaustion features — consecutive sell-only days, 10d
    print imbalance — univariate IS splits.
[C] BOND-SPECIFIC vs ISSUER-WIDE dislocation: is a same-issuer sibling also
    dislocated (news) or is this bond alone (flow)? Univariate IS split.
[D] CROWDING: monthly signal density (systemic vs idiosyncratic episodes).
[B] WALK-FORWARD ML ENTRY RANKER (the flagship): HistGradientBoosting on 14
    point-in-time features, predicting each XL trade's return.

    Leakage protocol (pre-registered, frozen):
      - candidate fills = the frozen GRANITE-XL spec (entries + recovery exit)
      - for entry year Y: train ONLY on fills whose EXIT completed before
        Jan 1 of Y (completed-trades embargo)
      - selection threshold = median of the TRAINING set's own predictions
        (fully point-in-time; no within-year peeking)
      - model hyperparameters written once, never tuned:
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
        max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0)
      - Ridge(alpha=1) reported as robustness, never selected post hoc
      - IS evaluation years 2008-2015; pre-registered gate: ML top-half must
        beat the full XL book on BOTH mean/trade and MTM Sharpe in IS.
        OOS years 2016-2024 run once, reported as printed.

  python corps/research/flowml.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from granite_experiments import sig_disc, gate_mat5, issuer_cap_filter  # noqa: E402
from oos2 import limit_filter  # noqa: E402
from combos import dynamic_exit  # noqa: E402

MAXH = 455
FULL_LO, FULL_HI = e2.D("2003-01-01"), e2.D("2025-03-31") - MAXH
IS_EVAL = (2008, 2015)
OOS_EVAL = (2016, 2024)

FEATURES = ["depth", "cs", "mat", "qv90", "spr60", "qv_spike", "vol90",
            "dd30", "sell_days10", "imb10", "sell_streak", "sib_dislo",
            "n_sibs", "crowding", "rf", "drf90"]


def xl_fills(bonds):
    f = e2.run_events(bonds, sig_disc(3.0), min_hold=365, max_hold=MAXH,
                      date_lo=FULL_LO, date_hi=FULL_HI, extra_gate=gate_mat5)
    f = issuer_cap_filter(f, cap=1)
    f = limit_filter(bonds, f, cap=0.25)
    return dynamic_exit(bonds, f)


def build_features(bonds, fills):
    imap = {}
    for six in bonds:
        imap.setdefault(six[:6], []).append(six)
    # global monthly crowding: signals fired / eligible bonds
    sig_month, elig_month = {}, {}
    for six, b in bonds.items():
        med60 = b.get("med60")
        if med60 is None:
            continue
        g = b["elig"] & (b["mat"] <= 5)
        s = g & ((b["s_px"] - med60) <= -3.0)
        for d in b["day"][g][::5]:
            elig_month[d // 30] = elig_month.get(d // 30, 0) + 1
        for d in b["day"][s]:
            sig_month[d // 30] = sig_month.get(d // 30, 0) + 1
    rows = []
    for f in fills:
        b = bonds[f.six]
        day = b["day"]
        i = np.searchsorted(day, f.entry_day, side="left") - 1
        if i < 0:
            continue
        med = b["med60"][i]
        j90 = np.searchsorted(day, day[i] - 90)
        j30 = np.searchsorted(day, day[i] - 30)
        j10 = np.searchsorted(day, day[i] - 10)
        mid_seg = b["mid"][j90:i + 1]
        mid30 = b["mid"][j30:i + 1]
        s_seg = b["s_px"][j10:i + 1]
        p_seg = b["p_px"][j10:i + 1]
        has_s = np.isfinite(s_seg); has_p = np.isfinite(p_seg)
        sell_only = has_p & ~has_s
        streak = 0
        for k in range(len(sell_only) - 1, -1, -1):
            if sell_only[k]:
                streak += 1
            elif has_s[k]:
                break
        sib_d = 0; n_sibs = 0
        for p in imap.get(f.six[:6], []):
            if p == f.six:
                continue
            bp = bonds[p]
            jj = np.searchsorted(bp["day"], f.entry_day, side="right") - 1
            if jj < 0 or f.entry_day - bp["day"][jj] > 5:
                continue
            n_sibs += 1
            mp = bp.get("med60")
            if mp is not None and np.isfinite(mp[jj]) and \
               np.isfinite(bp["s_px"][jj]) and (bp["s_px"][jj] - mp[jj]) <= -3.0:
                sib_d = 1
        mb = f.entry_day // 30
        crowd = sig_month.get(mb, 0) / max(elig_month.get(mb, 1), 1)
        rf_now = float(e2.load_rf(np.array([f.entry_day]))[0])
        rf_90 = float(e2.load_rf(np.array([f.entry_day - 90]))[0])
        rows.append({
            "six": f.six, "entry_day": f.entry_day, "exit_day": f.exit_day,
            "ret": f.ret, "hold": f.hold,
            "depth": float(med - f.entry_px) if np.isfinite(med) else 3.0,
            "cs": float(b["cs"][i]) if np.isfinite(b["cs"][i]) else 0.02,
            "mat": int(b["mat"][i]),
            "qv90": float(b["qv90"][i]),
            "spr60": float(b["spr60"][i]) if np.isfinite(b["spr60"][i]) else 1.0,
            "qv_spike": float(np.nan_to_num(b["qv"][i]) / max(float(b["qvmed90"][i]), 1e-3)),
            "vol90": float(np.nanstd(mid_seg)) if len(mid_seg) > 3 else 1.0,
            "dd30": float(np.nanmax(mid30) - b["mid"][i]) if np.isfinite(mid30).any() else 0.0,
            "sell_days10": int(sell_only.sum()),
            "imb10": float((has_s.sum() - has_p.sum()) / max(has_s.sum() + has_p.sum(), 1)),
            "sell_streak": int(streak),
            "sib_dislo": sib_d, "n_sibs": n_sibs,
            "crowding": float(crowd), "rf": rf_now, "drf90": rf_now - rf_90,
        })
    return rows


def uni_split(rows, key, cuts, label, yr_hi=2015):
    sub = [r for r in rows if r["entry_day"] // 365 + 1970 <= yr_hi]
    print(f"\n[{label}] IS univariate split on {key}:", flush=True)
    out = {}
    for tag, lo, hi in cuts:
        grp = [r["ret"] for r in sub if lo <= r[key] < hi]
        if len(grp) < 30:
            continue
        out[tag] = {"n": len(grp), "mean": float(np.mean(grp)),
                    "win": float(np.mean(np.array(grp) > 0))}
        print(f"  {tag:22} n={len(grp):5} mean={np.mean(grp)*100:+6.2f}% "
              f"win={np.mean(np.array(grp)>0)*100:.0f}%", flush=True)
    return out


def walk_forward(rows):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    X = np.array([[r[k] for k in FEATURES] for r in rows])
    y = np.array([r["ret"] for r in rows])
    yr = np.array([1970 + r["entry_day"] // 365 for r in rows])
    xd = np.array([r["exit_day"] for r in rows])
    picks_gbm = np.zeros(len(rows), bool)
    picks_r = np.zeros(len(rows), bool)
    preds = np.full(len(rows), np.nan)
    for Y in range(IS_EVAL[0], OOS_EVAL[1] + 1):
        jan1 = e2.D(f"{Y}-01-01")
        tr = xd < jan1
        te = yr == Y
        if tr.sum() < 300 or te.sum() == 0:
            continue
        gbm = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=50, l2_regularization=1.0, random_state=7)
        gbm.fit(X[tr], y[tr])
        thr = float(np.median(gbm.predict(X[tr])))
        pt = gbm.predict(X[te])
        preds[te] = pt
        picks_gbm[te] = pt >= thr
        rg = Ridge(alpha=1.0).fit(X[tr], y[tr])
        thr_r = float(np.median(rg.predict(X[tr])))
        picks_r[te] = rg.predict(X[te]) >= thr_r
    return picks_gbm, picks_r, preds, yr


def eval_book(bonds, rows, mask, label):
    sel = [i for i in range(len(rows)) if mask[i]]
    if not sel:
        print(f"  {label}: empty", flush=True)
        return None
    rr = np.array([r["ret"] for i, r in enumerate(rows) if mask[i]])
    print(f"  {label:26} n={len(rr):5} mean={rr.mean()*100:+6.2f}% "
          f"win={(rr>0).mean()*100:.0f}%", flush=True)
    return {"n": int(len(rr)), "mean": float(rr.mean()),
            "win": float((rr > 0).mean())}


def mtm_book(bonds, rows, mask, label):
    fl = []
    for i, r in enumerate(rows):
        if not mask[i]:
            continue
        b = bonds[r["six"]]
        day = b["day"]
        ii = np.searchsorted(day, r["entry_day"], side="left")
        # reconstruct entry/exit prices from the stored return is lossy; use
        # actual prints: entry = first ask at entry_day, exit solves ret
        fl.append((r["six"], r["entry_day"], r["exit_day"], r["ret"], r["hold"]))
    # simple NAV: daily equal-weight of per-trade daily-compounded returns
    if not fl:
        return None
    d0 = min(x[1] for x in fl); d1 = max(x[2] for x in fl)
    days = np.arange(d0, d1 + 1)
    sret = np.zeros(len(days)); cnt = np.zeros(len(days))
    for six, ed, xdy, ret, hold in fl:
        if hold <= 0:
            continue
        dr = (1 + ret) ** (1 / hold) - 1
        a = ed + 1 - d0; z = xdy + 1 - d0
        sret[a:z] += dr; cnt[a:z] += 1
    rf = e2.load_rf(days)
    daily = np.where(cnt > 0, sret / np.where(cnt > 0, cnt, 1), rf / 365)
    nav = np.cumprod(1 + daily)
    ps = e2.perf_stats(days, nav, daily)
    print(f"    {label} NAV*: cagr={ps['cagr']*100:+6.2f}% sharpe_m={ps['sharpe_m']:.2f} "
          f"maxdd={ps['maxdd']*100:.1f}%  (*linear attribution — comparative only)",
          flush=True)
    return ps


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)
    fills = xl_fills(bonds)
    print(f"XL fills (full sample): {len(fills)}", flush=True)
    rows = build_features(bonds, fills)
    print(f"feature rows: {len(rows)}", flush=True)
    out = {}

    out["flow_streak"] = uni_split(rows, "sell_streak",
                                   [("streak 0", -1, 1), ("streak 1-2", 1, 3),
                                    ("streak >=3", 3, 99)], "A1 sell-streak")
    out["flow_imb"] = uni_split(rows, "imb10",
                                [("sell-heavy (<-0.3)", -2, -0.3),
                                 ("balanced", -0.3, 0.3),
                                 ("buy-heavy (>0.3)", 0.3, 2)], "A2 imbalance")
    out["sibling"] = uni_split(rows, "sib_dislo",
                               [("bond-specific (no sib)", 0, 1),
                                ("issuer-wide (sib also)", 1, 2)], "C sibling")
    out["crowding"] = uni_split(rows, "crowding",
                                [("calm (<2%)", 0, 0.02),
                                 ("normal (2-6%)", 0.02, 0.06),
                                 ("crisis (>6%)", 0.06, 1)], "D crowding")

    print("\n[B] walk-forward ML ranker:", flush=True)
    picks_gbm, picks_r, preds, yr = walk_forward(rows)
    for tag, (y0, y1) in [("IS 2008-2015", IS_EVAL), ("OOS 2016-2024", OOS_EVAL)]:
        span = (yr >= y0) & (yr <= y1)
        print(f"\n  {tag}:", flush=True)
        out[f"ml_{tag[:2]}_all"] = eval_book(bonds, rows, span, "all XL trades")
        mtm_book(bonds, rows, span, "all")
        out[f"ml_{tag[:2]}_top"] = eval_book(bonds, rows, span & picks_gbm, "GBM top-half")
        mtm_book(bonds, rows, span & picks_gbm, "GBM top")
        out[f"ml_{tag[:2]}_bot"] = eval_book(bonds, rows, span & ~picks_gbm, "GBM bottom-half")
        out[f"ml_{tag[:2]}_ridge"] = eval_book(bonds, rows, span & picks_r, "Ridge top-half (robustness)")

    (ROOT / "research" / "flowml_results.json").write_text(json.dumps(out, default=float))
    np.save(ROOT / "research" / "flowml_preds.npy", preds)
    print("\nwrote corps/research/flowml_results.json", flush=True)


if __name__ == "__main__":
    main()
