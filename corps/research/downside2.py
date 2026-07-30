"""E5 — walk-forward ML tail-avoider (IS evaluation).

Unlike stops (exit into the crisis) or price floors (skip the alpha), this
skips ENTRIES predicted to be disasters ex-ante: a walk-forward classifier
P(trade return < -10%) on the 16 point-in-time features, completed-trades
embargo, frozen hyperparameters. Skip the worst decile by training-set
threshold. Same admission gate as E1-E4.

  python corps/research/downside2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from flowml import build_features, FEATURES  # noqa: E402
from downside import cl_entries, book, IS_LO, IS_HI  # noqa: E402
from combos import dynamic_exit  # noqa: E402


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)

    entries = cl_entries(bonds, IS_LO, IS_HI)
    xl = dynamic_exit(bonds, entries)
    rows = build_features(bonds, xl)
    key2fill = {(f.six, f.entry_day): f for f in xl}
    afills = [key2fill[(r["six"], r["entry_day"])] for r in rows]
    X = np.array([[r[k] for k in FEATURES] for r in rows])
    y = (np.array([r["ret"] for r in rows]) < -0.10).astype(int)
    yr = np.array([1970 + r["entry_day"] // 365 for r in rows])
    xd = np.array([r["exit_day"] for r in rows])
    print(f"rows {len(rows)}, tail rate {y.mean()*100:.1f}%", flush=True)

    keep = np.ones(len(rows), bool)
    for Y in range(2008, 2016):
        jan1 = e2.D(f"{Y}-01-01")
        tr = xd < jan1
        te = yr == Y
        if tr.sum() < 300 or te.sum() == 0:
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=50, l2_regularization=1.0, random_state=7)
        clf.fit(X[tr], y[tr])
        thr = float(np.percentile(clf.predict_proba(X[tr])[:, 1], 90))
        keep[te] = clf.predict_proba(X[te])[:, 1] < thr

    ev = (yr >= 2008) & (yr <= 2015)
    print("\n[E5] evaluation years 2008-2015:", flush=True)
    base = book(bonds, [f for f, m in zip(afills, ev) if m], "XL base (08-15)")
    kept = [f for f, m1, m2 in zip(afills, ev, keep) if m1 and m2]
    res = book(bonds, kept, "tail-avoider", base)
    skipped = [f for f, m1, m2 in zip(afills, ev, keep) if m1 and not m2]
    if skipped:
        rr = np.array([f.ret for f in skipped])
        print(f"  skipped trades: n={len(skipped)} mean={rr.mean()*100:+.2f}% "
              f"tail_rate={(rr<-0.10).mean()*100:.0f}% (model's targets)", flush=True)
    out = {"base_0815": base, "e5": res,
           "skipped": {"n": len(skipped),
                       "mean": float(np.mean([f.ret for f in skipped])) if skipped else None,
                       "tail_rate": float(np.mean([f.ret < -0.10 for f in skipped])) if skipped else None}}
    (ROOT / "research" / "downside2_is.json").write_text(json.dumps(out, default=float))
    print("\nwrote corps/research/downside2_is.json", flush=True)


if __name__ == "__main__":
    main()
