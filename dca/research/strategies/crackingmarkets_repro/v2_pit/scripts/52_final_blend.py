"""FINAL BLEND — best honest portfolio found across the whole research arc:
EOD 6-sleeve book + ORB-QQQ intraday + QQQ overnight session, 2016-2026.

Correlations (daily): eod/orb -0.07, eod/ovn 0.35, orb/ovn 0.01.
Peak: 40% EOD-k2 / 20% ORB / 40% overnight -> 13.4% CAGR, 1.19 Sh(d),
1.39 Sh(m), -21.5% maxDD (ORB & overnight priced at BEST-CASE 0.25bp/side;
at 1bp/side both sleeves degrade materially - see LOOP_NOTES.md).
"""
import sys, os, importlib.util
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import stats, fmt, riskfree_daily, load_etf, OUT

spec = importlib.util.spec_from_file_location("o", os.path.join(HERE, "50_orb_intraday.py"))
om = importlib.util.module_from_spec(spec)
spec.loader.exec_module(om)
om.COST["QQQ"] = 0.25
om.STOP_SLIP = 0.5

eod = pd.read_parquet(os.path.join(OUT, "ensemble_v2.parquet"))["evol_k2"]
orb = om.orb("QQQ")
q = load_etf("QQQ")
ovn = (q["Open"] / q["Close"].shift(1) - 1 - 2 * 0.25 / 1e4).dropna()

X = pd.concat([eod.rename("eod"), orb.rename("orb"), ovn.rename("ovn")],
              axis=1).dropna()
rf = riskfree_daily(X.index)
r = 0.4 * X["eod"] + 0.2 * X["orb"] + 0.4 * X["ovn"]
print(fmt(stats(r, rf, "FINAL BLEND 40/20/40 (best-case costs)")))
r.to_frame("blend").to_parquet(os.path.join(OUT, "final_blend.parquet"))
print("saved -> out/final_blend.parquet")
