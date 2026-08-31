"""BEDROCK issuance-pressure event study (corp) — Helwege-Wang (JFI 2021)
replication on our tape. Measurement only.

Event: a new CUSIP's first panel appearance at date T where the issuer
(CUSIP6) has existing bonds that started trading >=180d before T. "Large"
deals = top tercile of the new bond's first-30d mean daily volume within the
annual new-issue cohort. For each existing sibling:

  diff-in-diff cs change (bp): [T-2,T+10] median minus [T-30,T-10) baseline,
  minus the market-wide median cs change over the same calendar windows;
  reversion leg: [T+30,T+60] vs the same baseline.

Literature prediction: ~+9bp cheapening, reverting within ~6 weeks.

  python corps/research/bedrock_i_event.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402

LO, HI = e2.D("2003-06-01"), e2.D("2024-09-30")


def main():
    bonds = e2.load_cache()
    print(f"loaded {len(bonds)} bonds", flush=True)

    # market-wide daily median cs (for the diff-in-diff market leg)
    print("building market cs median series ...", flush=True)
    days_all, cs_all = [], []
    for b in bonds.values():
        ok = np.isfinite(b["cs"])
        days_all.append(b["day"][ok]); cs_all.append(b["cs"][ok])
    mk = pd.DataFrame({"day": np.concatenate(days_all),
                       "cs": np.concatenate(cs_all)})
    mkt = mk.groupby("day")["cs"].median()
    del mk, days_all, cs_all
    mkt_d = mkt.index.to_numpy()
    mkt_v = mkt.to_numpy()

    def mkt_med(lo, hi):
        i0 = np.searchsorted(mkt_d, lo); i1 = np.searchsorted(mkt_d, hi, side="right")
        if i1 <= i0:
            return np.nan
        return float(np.median(mkt_v[i0:i1]))

    issuers = {}
    for six, b in bonds.items():
        issuers.setdefault(six[:6], []).append(six)

    # new-issue events with existing siblings
    first = {six: int(b["day"][0]) for six, b in bonds.items() if len(b["day"])}
    vol30 = {}
    events = []
    for six, b in bonds.items():
        T = first[six]
        if not (LO <= T <= HI):
            continue
        sibs = [s for s in issuers[six[:6]]
                if s != six and first.get(s, 10**9) <= T - 180]
        if len(sibs) < 1:
            continue
        m = b["day"] <= T + 30
        v = b["qv"][m]
        vol30[six] = float(np.nanmean(v)) if np.isfinite(v).any() else 0.0
        events.append((T, six, sibs))
    print(f"new-issue events with existing siblings: {len(events)}", flush=True)

    # large tercile within annual cohort
    ev_df = pd.DataFrame({"T": [e[0] for e in events], "six": [e[1] for e in events]})
    ev_df["yr"] = (ev_df["T"] // 365) + 1970
    ev_df["v30"] = ev_df["six"].map(vol30)
    ev_df["big"] = ev_df.groupby("yr")["v30"].transform(
        lambda s: s >= s.quantile(2 / 3))
    big = set(ev_df.loc[ev_df["big"], "six"])
    print(f"large-deal events: {len(big)}", flush=True)

    def sib_cs(sb, lo, hi):
        i0 = np.searchsorted(sb["day"], lo); i1 = np.searchsorted(sb["day"], hi, side="right")
        if i1 - i0 < 1:
            return np.nan
        v = sb["cs"][i0:i1]
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else np.nan

    rows = []
    for T, six, sibs in events:
        w = {"base": (T - 30, T - 10), "evt": (T - 2, T + 10), "rev": (T + 30, T + 60)}
        mk0 = mkt_med(*w["base"]); mk1 = mkt_med(*w["evt"]); mk2 = mkt_med(*w["rev"])
        if not all(np.isfinite(x) for x in (mk0, mk1, mk2)):
            continue
        for s in sibs[:6]:                    # cap siblings per event
            sb = bonds[s]
            c0 = sib_cs(sb, *w["base"]); c1 = sib_cs(sb, *w["evt"]); c2 = sib_cs(sb, *w["rev"])
            if not all(np.isfinite(x) for x in (c0, c1, c2)):
                continue
            rows.append({"six": s, "issuer": six[:6], "big": six in big,
                         "d_evt": (c1 - c0) - (mk1 - mk0),
                         "d_rev": (c2 - c0) - (mk2 - mk0)})
    df = pd.DataFrame(rows)
    print(f"sibling observations: {len(df):,} ({df['issuer'].nunique():,} issuers)", flush=True)

    res = {"n": int(len(df))}
    for tag, sub in [("all", df), ("large", df[df["big"]]), ("small", df[~df["big"]])]:
        if not len(sub):
            continue
        # cluster by issuer
        g = sub.groupby("issuer")[["d_evt", "d_rev"]].mean()
        e_bp = g["d_evt"] * 1e4; r_bp = g["d_rev"] * 1e4
        te = float(e_bp.mean() / (e_bp.std() / np.sqrt(len(e_bp))))
        tr = float(r_bp.mean() / (r_bp.std() / np.sqrt(len(r_bp))))
        res[tag] = {"n_issuers": int(len(g)),
                    "evt_bp": float(e_bp.mean()), "t_evt": te,
                    "rev_bp": float(r_bp.mean()), "t_rev": tr}
        print(f"  {tag:6} issuers={len(g):5}  event-window {e_bp.mean():+6.1f}bp (t={te:+5.2f})"
              f"  +30..60d {r_bp.mean():+6.1f}bp (t={tr:+5.2f})", flush=True)

    p = ROOT / "research" / "bedrock_i_event.json"
    p.write_text(json.dumps(res, default=float))
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
