"""BULWARK loss-tail audit: 'bust-linked = 0' is a LABEL statement, not an
economic one. A bond can lose 70-90% and keep printing (so never be labelled
bust). This measures credit losses convention-free, from realised returns.

  python corps/research/bulwark_tail.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from bulwark_v1 import build_xsec  # noqa: E402
from bulwark_v2 import pipeline  # noqa: E402

FULL = (e2.D("2003-01-01"), e2.D("2024-01-01"))
SPECS = {"BULWARK-U": {}, "BULWARK-S4": dict(s4=True),
         "BULWARK-Z": dict(s1=True, s2=True, s4=True, s6=True)}


def main():
    bonds = e2.load_cache()
    issuers = {}
    for six in bonds:
        issuers.setdefault(six[:6], []).append(six)
    ldf = pd.read_parquet(ROOT / "research" / "bulwark_labels.parquet")
    lab = {r.six: (bool(r.bust), int(r.last_day)) for r in ldf.itertuples()}
    q97, amed = build_xsec(bonds)
    out = {}
    for name, sc in SPECS.items():
        fl = pipeline(bonds, issuers, q97, amed, *FULL, **sc)
        r = np.array([f.ret for f in fl])
        nb = sum(1 for f in fl if (lab.get(f.six) and lab[f.six][0]
                                   and f.entry_day <= lab[f.six][1] <= f.exit_day + 30))
        o = {"n": len(fl), "bust_label": nb, "bust_label_rate": nb / len(fl)}
        print(f"\n=== {name} (n={len(fl)}) ===", flush=True)
        print(f"  labelled bust-linked : {nb} ({nb/len(fl)*100:.2f}%)", flush=True)
        for thr in (-0.50, -0.30, -0.20):
            k = int((r < thr).sum())
            o[f"lt{int(abs(thr)*100)}"] = k
            o[f"lt{int(abs(thr)*100)}_rate"] = k / len(fl)
            print(f"  realised ret < {thr*100:+.0f}%   : {k:4} ({k/len(fl)*100:5.2f}%)"
                  + (f"  mean {r[r<thr].mean()*100:+.1f}%" if k else ""), flush=True)
        # how much of total return the tail destroys
        o["tail_drag_pp"] = float((r.mean() - r[r >= -0.30].mean()) * 100) if (r < -0.30).any() else 0.0
        print(f"  mean/trade {r.mean()*100:+.2f}%  vs  {r[r>=-0.30].mean()*100:+.2f}% "
              f"excluding the <-30% tail  (drag {o['tail_drag_pp']:.2f}pp)", flush=True)
        deep = [f for f in fl if f.ret < -0.50]
        o["deep"] = [{"six": f.six,
                      "entry": str(np.int64(f.entry_day).astype("datetime64[D]")),
                      "ret": float(f.ret),
                      "labelled_bust": bool(lab.get(f.six, (False, 0))[0])} for f in deep]
        miss = sum(1 for d in o["deep"] if not d["labelled_bust"])
        o["deep_unlabelled"] = miss
        print(f"  of the {len(deep)} trades losing >50%, {miss} are NOT labelled bust "
              f"-> the label understates credit losses", flush=True)
        out[name] = o
    (ROOT / "research" / "bulwark_tail.json").write_text(json.dumps(out, default=float))
    print("\nwrote bulwark_tail.json", flush=True)


if __name__ == "__main__":
    main()
