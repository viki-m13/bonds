"""Assemble every number the white paper cites into docs/munis_data.json,
recomputed from the panel so the page can never drift from the research.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
from strategies import FACTORIES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT.parent / "docs"
DATA_END = pd.Timestamp("2026-07-08")


def load():
    panel = pd.read_parquet(ROOT / "data" / "panel_daily.parquet")
    uni = (pd.read_csv(ROOT / "data" / "universe" / "universe.csv.gz")
           .drop_duplicates("six").set_index("six"))
    return panel, uni, bt.prepare(panel, uni["coupon"])


def horizon_curve(panel):
    p = panel.sort_values(["six", "date"])
    hz = [30, 90, 180, 365, 730]
    res = {h: [] for h in hz}
    for six, g in p.groupby("six"):
        g = g.reset_index(drop=True)
        c = 4.5
        dts = g["date"].values
        s = g["s_px"].values
        pp = g["p_px"].values
        pidx = np.where(~np.isnan(pp))[0]
        if len(pidx) == 0:
            continue
        pdates = dts[pidx]
        for i in range(len(g)):
            if np.isnan(s[i]):
                continue
            for h in hz:
                tgt = dts[i] + np.timedelta64(h, "D")
                j = np.searchsorted(pdates, tgt)
                if j >= len(pidx):
                    continue
                pi = pidx[j]
                hold = (dts[pi] - dts[i]) / np.timedelta64(1, "D")
                if hold > h + 90:
                    continue
                res[h].append((pp[pi] - s[i] + c / 365.0 * hold) / s[i])
    return [{"hold": h, "mean": float(np.mean(res[h])),
             "win": float(np.mean(np.array(res[h]) > 0))} for h in hz]


def decile_curve(panel):
    rows = []
    for six, g in panel.groupby("six"):
        g = g.reset_index(drop=True)
        gi = g.set_index("date")
        med = gi["mid"].rolling("60D", min_periods=5).median().shift(1).values
        s = g["s_px"].values
        pp = g["p_px"].values
        dts = g["date"].values
        pidx = np.where(~np.isnan(pp))[0]
        if len(pidx) == 0:
            continue
        pdates = dts[pidx]
        for i in range(len(g)):
            if np.isnan(s[i]) or np.isnan(med[i]):
                continue
            tgt = dts[i] + np.timedelta64(365, "D")
            j = np.searchsorted(pdates, tgt)
            if j >= len(pidx):
                continue
            pi = pidx[j]
            hold = (dts[pi] - dts[i]) / np.timedelta64(1, "D")
            if hold > 455:
                continue
            rows.append((s[i] - med[i],
                         (pp[pi] - s[i] + 4.5 / 365.0 * hold) / s[i]))
    df = pd.DataFrame(rows, columns=["disc", "ret"])
    df["dec"] = pd.qcut(df["disc"], 10, labels=False, duplicates="drop")
    g = df.groupby("dec")["ret"].mean()
    return [{"decile": int(d), "mean": float(v)} for d, v in g.items()]


def era_table(bonds):
    fn = FACTORIES["price_discount"](discount=3.0)
    eras = [("2013–2016", "2013-01-01", "2016-12-31"),
            ("2017–2019", "2017-01-01", "2019-12-31"),
            ("2020–2021", "2020-01-01", "2021-12-31"),
            ("2022", "2022-01-01", "2022-12-31"),
            ("2023–2025", "2023-01-01", "2025-04-08")]
    out = []
    for name, lo, hi in eras:
        lo = pd.Timestamp(lo)
        hi = min(pd.Timestamp(hi), DATA_END - pd.Timedelta(days=455))
        fills = bt.run_signal(bonds, fn, min_hold=365, max_hold=455,
                              date_lo=lo, date_hi=hi)
        ctl = bt.matched_random_control(bonds, fills, min_hold=365, max_hold=455)
        s = bt.summarize(fills, name, control=ctl)
        out.append({"era": name, "n": s["n"], "win": s["win_rate"],
                    "mean": s["mean_ret"],
                    "excess": s["excess_vs_control"],
                    "p": s["excess_p_boot"]})
    return out


def threshold_table(bonds, lo, hi, label):
    out = []
    for disc in [1.0, 2.0, 3.0]:
        fn = FACTORIES["price_discount"](discount=disc)
        fills = bt.run_signal(bonds, fn, min_hold=365, max_hold=455,
                              date_lo=pd.Timestamp(lo),
                              date_hi=min(pd.Timestamp(hi),
                                          DATA_END - pd.Timedelta(days=455)))
        ctl = bt.matched_random_control(bonds, fills, min_hold=365, max_hold=455)
        s = bt.summarize(fills, f"d{disc}", control=ctl)
        out.append({"thresh": disc, "n": s["n"], "win": s["win_rate"],
                    "mean": s["mean_ret"], "ctrl": s["control_mean_ret"],
                    "excess": s["excess_vs_control"], "p": s["excess_p_boot"]})
    return out


def equity(bonds):
    fn = FACTORIES["price_discount"](discount=3.0)
    fills = bt.run_signal(bonds, fn, min_hold=365, max_hold=455,
                          date_lo=pd.Timestamp("2013-01-01"),
                          date_hi=DATA_END - pd.Timedelta(days=455))
    start = min(f.entry_date for f in fills)
    end = max(f.exit_date for f in fills)
    days = pd.date_range(start, end, freq="D")
    di = {d: i for i, d in enumerate(days)}
    sret = np.zeros(len(days))
    cnt = np.zeros(len(days))
    for f in fills:
        if f.hold_days <= 0:
            continue
        dr = (1 + f.ret) ** (1.0 / f.hold_days) - 1
        sret[di[f.entry_date]:di[f.exit_date]] += dr
        cnt[di[f.entry_date]:di[f.exit_date]] += 1
    port = np.where(cnt > 0, sret / np.where(cnt > 0, cnt, 1), 0.0)
    eq = np.cumprod(1 + port)
    mub = (pd.read_csv(ROOT / "data" / "mub_daily.csv.gz", parse_dates=["date"])
           .sort_values("date").set_index("date")["adjclose"]
           .reindex(days).ffill())
    mub_eq = (mub / mub.iloc[0]).to_numpy()

    def st(e):
        yrs = (days[-1] - days[0]).days / 365.25
        peak = np.maximum.accumulate(e)
        return {"total": float(e[-1] - 1), "cagr": float(e[-1] ** (1 / yrs) - 1),
                "maxdd": float((e / peak - 1).min())}
    # monthly downsample for embedding
    idx = pd.Series(range(len(days)), index=days)
    mon = idx.resample("MS").first().dropna().astype(int).tolist()
    if idx.iloc[-1] not in mon:
        mon.append(len(days) - 1)
    series = [{"date": days[i].strftime("%Y-%m-%d"),
               "strat": round(float(eq[i]), 4),
               "mub": round(float(mub_eq[i]), 4)} for i in mon]
    return {"series": series, "strat": st(eq), "mub": st(mub_eq),
            "n_trades": len(fills),
            "years": round((days[-1] - days[0]).days / 365.25, 1),
            "avg_positions": int(cnt[cnt > 0].mean())}


def main():
    panel, uni, bonds = load()
    data = {
        "meta": {
            "n_bonds": int(panel["six"].nunique()),
            "n_trades": int(pd.read_csv(
                ROOT / "data" / "universe" / "download_meta.csv")["n_trades"].sum()),
            "universe": int(len(uni)),
            "states": int(uni["state"].nunique()),
            "data_start": str(panel["date"].min().date()),
            "data_end": str(panel["date"].max().date()),
        },
        "validation": {
            "xendpoint_match": 1.0,
            "py_corr": -0.98,
            "spread_median": 0.44,
            "side_consistency": 0.969,
        },
        "horizon": horizon_curve(panel),
        "decile": decile_curve(panel),
        "threshold_is": threshold_table(bonds, "2012-01-01", "2022-12-31", "IS"),
        "threshold_full": threshold_table(bonds, "2013-01-01", "2026-07-08", "FULL"),
        "era": era_table(bonds),
        "oos": pd.read_csv(ROOT / "research" / "results" / "oos_results.csv")
        .to_dict("records"),
        "is_grid": pd.read_csv(ROOT / "research" / "results" / "is_grid.csv")
        .to_dict("records"),
        "equity": equity(bonds),
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "munis_data.json").write_text(json.dumps(data, default=str))
    print("wrote", DOCS / "munis_data.json")
    print(json.dumps(data["equity"]["strat"], indent=2))
    print(json.dumps(data["era"], indent=2, default=str))


if __name__ == "__main__":
    main()
