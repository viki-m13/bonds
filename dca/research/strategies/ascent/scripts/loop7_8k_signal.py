"""LOOP iter 7 — 8-K ITEM-type event signals vs the tail-precision spec.
Item taxonomy (SEC 8-K items, post-2004 numbering):
  NEG: 4.01 auditor change, 4.02 non-reliance/restatement, 1.03 bankruptcy,
       3.01 delisting notice, 2.05 exit/disposal, 2.06 impairment
  POS: 1.01 material agreement, 2.01 completed acquisition, 5.03? no.
  AMB: 5.02 officer dep/appointment, 8.01 other, 2.02 earnings, 7.01 RegFD
Tests (monthly, top-1500 $vol pond, 2005-2025, era-sliced):
  1. event-cohort mean fwd-12m rank (with vs without event, trailing 3m)
  2. tail rates: P(top decile) and P(bottom decile) per cohort
  3. factor profile of cohorts (mom12/vol/size ranks) — orthogonality check
  4. If a NEG veto or POS tilt looks era-stable: harness era backtest.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os as _os; HERE = _os.environ.get("ASCENT_WORK", "/tmp/ascent_work"); _os.makedirs(HERE, exist_ok=True)
REPO = _os.environ.get("BONDS_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..")))
sys.path.insert(0, HERE)

t0 = time.time()
def p(*a): print(*a, flush=True)
D = pd.read_pickle(f"{HERE}/_featmat.pkl")
FEAT, liq, me, dv, bench, cols = D["FEAT"], D["liq"], D["me"], D["dv"], D["bench"], D["cols"]
M = me.index
ma10 = me.rolling(10, min_periods=10).mean()
dvr = dv.rank(axis=1, ascending=False)
top1500 = dvr <= 1500
fwd12 = (me.shift(-12) / me - 1)
FR = fwd12.where(liq & top1500).rank(axis=1, pct=True)
mom12r = (me / me.shift(12) - 1).where(liq & top1500).rank(axis=1, pct=True)
vol6r = FEAT["vol6"].where(liq & top1500).rank(axis=1, pct=True)

E = pd.read_parquet(f"{HERE}/_8k_items.parquet")
E["date"] = pd.to_datetime(E.date)
E["ym"] = E.date.values.astype("datetime64[M]")
E = E[E.tk.isin(set(cols))]
p(f"events {len(E)} filings, {E.tk.nunique()} tickers, {E.date.min().date()}..{E.date.max().date()}")

ITEMS = {
    "auditor_4.01": "4.01", "restate_4.02": "4.02", "bankrupt_1.03": "1.03",
    "delist_3.01": "3.01", "exitcost_2.05": "2.05", "impair_2.06": "2.06",
    "matagree_1.01": "1.01", "acq_2.01": "2.01", "officer_5.02": "5.02",
    "earn_2.02": "2.02", "other_8.01": "8.01",
}
# monthly indicator: any filing with that item in trailing 3 months
IND = {}
for nm, code in ITEMS.items():
    sub = E[E["items"].str.contains(code, regex=False)]
    pv = sub.assign(v=1.0).pivot_table(index="ym", columns="tk", values="v", aggfunc="sum")
    pv = pv.reindex(index=M, columns=cols).fillna(0.0)
    IND[nm] = (pv.rolling(3, min_periods=1).sum() > 0)
NEG = (IND["auditor_4.01"] | IND["restate_4.02"] | IND["bankrupt_1.03"]
       | IND["delist_3.01"] | IND["exitcost_2.05"] | IND["impair_2.06"])

ERAS = [("2005-01", "2009-12"), ("2010-01", "2014-12"), ("2015-01", "2019-12"), ("2020-01", "2025-06")]
p(f"\n{'cohort':16} era: mean-fwd-rank(event) - mean(no-event) | P(top10%) evt/no | P(bot10%) evt/no | n_evt")
for nm in list(ITEMS) + ["NEG_any"]:
    ind = NEG if nm == "NEG_any" else IND[nm]
    line = f"{nm:16}"
    for st, en in ERAS:
        dts = M[(M >= pd.Timestamp(st + "-01")) & (M <= pd.Timestamp(en + "-01"))]
        de, dn, te, tn, be, bn, ne = [], [], [], [], [], [], 0
        for dt in dts[::2]:
            el = (liq.loc[dt] & top1500.loc[dt] & (me.loc[dt] >= 3))
            f = FR.loc[dt].where(el).dropna()
            ev = ind.loc[dt].reindex(f.index).fillna(False)
            if ev.sum() < 5: continue
            de.append(f[ev].mean()); dn.append(f[~ev].mean())
            te.append((f[ev] >= 0.9).mean()); tn.append((f[~ev] >= 0.9).mean())
            be.append((f[ev] <= 0.1).mean()); bn.append((f[~ev] <= 0.1).mean())
            ne += int(ev.sum())
        if de:
            line += (f" | {np.mean(de)-np.mean(dn):+.3f} T{np.mean(te):.0%}/{np.mean(tn):.0%}"
                     f" B{np.mean(be):.0%}/{np.mean(bn):.0%} n{ne}")
        else:
            line += " |  --"
    p(line)

# factor profile of the NEG cohort (orthogonality)
p("\nfactor profile (mean ranks) of NEG cohort vs rest, 2010-2025:")
mm, vv, base_m, base_v = [], [], [], []
for dt in M[(M >= pd.Timestamp("2010-01-01")) & (M <= pd.Timestamp("2025-06-01"))][::3]:
    el = (liq.loc[dt] & top1500.loc[dt])
    ev = NEG.loc[dt].where(el).fillna(False)
    mm.append(mom12r.loc[dt][ev].mean()); base_m.append(mom12r.loc[dt][el & ~ev].mean())
    vv.append(vol6r.loc[dt][ev].mean()); base_v.append(vol6r.loc[dt][el & ~ev].mean())
p(f"  mom12 rank: NEG {np.nanmean(mm):.3f} vs rest {np.nanmean(base_m):.3f}")
p(f"  vol6 rank:  NEG {np.nanmean(vv):.3f} vs rest {np.nanmean(base_v):.3f}")
p(f"\nDONE t={time.time()-t0:.0f}s")
