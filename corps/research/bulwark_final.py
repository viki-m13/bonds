"""BULWARK finishing pass: era decomposition, slippage grid, worst-trade
anatomy, and the pooled IS+OOS default record for the three specs.

  python corps/research/bulwark_final.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from bulwark_v1 import build_xsec, IS, OOS, MAXH  # noqa: E402
from bulwark_v2 import pipeline, stats_h  # noqa: E402

FULL = (e2.D("2003-01-01"), e2.D("2024-01-01"))
SPECS = {"BULWARK-U": {}, "BULWARK-S4": dict(s4=True),
         "BULWARK-Z": dict(s1=True, s2=True, s4=True, s6=True)}
ERAS = [("2003-2007", "2003-01-01", "2007-12-31"),
        ("2008-2009", "2008-01-01", "2009-12-31"),
        ("2010-2014", "2010-01-01", "2014-12-31"),
        ("2015-2016", "2015-01-01", "2016-12-31"),
        ("2017-2019", "2017-01-01", "2019-12-31"),
        ("2020", "2020-01-01", "2020-12-31"),
        ("2021-2023", "2021-01-01", "2023-12-31")]


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
        days, nav, daily = e2.mtm_nav(bonds, fl)
        ps = e2.perf_stats(days, nav, daily)
        print(f"\n=== {name}: pooled 2003-2024 ===", flush=True)
        print(f"  n={len(fl)} issuers={len({f.six[:6] for f in fl})} "
              f"mean={r.mean()*100:+.2f}% win={(r>0).mean()*100:.0f}% "
              f"BUST-LINKED={nb} ({nb/len(fl)*100:.2f}%) "
              f"cagr={ps['cagr']*100:+.2f}% sh={ps['sharpe_m']:.2f} "
              f"dd={ps['maxdd']*100:.1f}%", flush=True)
        o = {"n": len(fl), "mean": float(r.mean()), "win": float((r > 0).mean()),
             "bust_n": nb, "bust_rate": nb / len(fl), **ps, "era": {}, "slip": {}}

        print("  eras (entry vintage):", flush=True)
        for lab_e, elo, ehi in ERAS:
            lo_d, hi_d = e2.D(elo), e2.D(ehi)
            sub = [f for f in fl if lo_d <= f.entry_day <= hi_d]
            if len(sub) < 10:
                continue
            rr = np.array([f.ret for f in sub])
            nbe = sum(1 for f in sub if (lab.get(f.six) and lab[f.six][0]
                                         and f.entry_day <= lab[f.six][1] <= f.exit_day + 30))
            o["era"][lab_e] = {"n": len(sub), "mean": float(rr.mean()),
                               "win": float((rr > 0).mean()), "bust": nbe}
            print(f"    {lab_e:10} n={len(sub):5} mean={rr.mean()*100:+6.2f}% "
                  f"win={(rr>0).mean()*100:3.0f}% bust={nbe}", flush=True)

        print("  slippage (h pts on entry ask AND exit bid):", flush=True)
        for h in (0.125, 0.25, 0.5):
            fh = [e2.Fill(f.six, f.entry_day, f.entry_px + h, f.exit_day,
                          max(f.exit_px - h, 1.0), f.coupon, f.stale) for f in fl]
            rh = np.array([f.ret for f in fh])
            d2, n2, dl2 = e2.mtm_nav(bonds, fh)
            p2 = e2.perf_stats(d2, n2, dl2)
            o["slip"][str(h)] = {"mean": float(rh.mean()), "cagr": p2["cagr"],
                                 "sharpe_m": p2["sharpe_m"]}
            print(f"    h={h:5.3f} mean={rh.mean()*100:+6.2f}% "
                  f"cagr={p2['cagr']*100:+6.2f}% sh={p2['sharpe_m']:5.2f}", flush=True)

        worst = sorted(fl, key=lambda f: f.ret)[:5]
        o["worst"] = [{"six": f.six, "entry": str(np.int64(f.entry_day).astype("datetime64[D]")),
                       "ret": float(f.ret), "bust": bool(lab.get(f.six, (False, 0))[0])}
                      for f in worst]
        print("  worst 5 trades:", flush=True)
        for w in o["worst"]:
            print(f"    {w['six']} {w['entry']} {w['ret']*100:+7.1f}% "
                  f"{'(bond later busts)' if w['bust'] else ''}", flush=True)
        out[name] = o
    (ROOT / "research" / "bulwark_final.json").write_text(json.dumps(out, default=float))
    print("\nwrote bulwark_final.json", flush=True)


if __name__ == "__main__":
    main()
