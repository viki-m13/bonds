"""IS-ONLY screening of new-strategy candidates (design window 2002-2015).

Protocol: every candidate is evaluated ONLY on 2002-2015 here. No OOS numbers
are computed by this script — the one-shot OOS test happens once, later, for
the shortlisted winners (oos_validate.py). This is the anti-overfitting wall.

Candidates (all point-in-time, engine-enforced honesty):
  carry5      monthly rank of credit spread within <=5y non-distressed bonds
  carry3      same, <=3y (pull-to-par zone)
  mom_hy      6M momentum (skip last month) within HY-proxy (cs>3%) bonds
  rev_event   short-horizon price-pressure reversal (>=3pt drop vs 10d-ago
              mid on a >=3x volume spike; hold 21-90d)
  rv_issuer   cheap vs same-issuer peers (cs - peer median cs, |dmat|<=2y)

  python corps/research/candidates.py [names...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2        # noqa: E402
import portfolio as pf      # noqa: E402

IS_LO, IS_HI = e2.D("2003-01-01"), e2.D("2015-12-31")


# ------------------------------------------------------------- universe gates

def gate_carry5(b):
    return (b["mat"] <= 5) & (b["cs"] <= 0.05) & (b["cs"] >= 0.005)


def gate_carry3(b):
    return (b["mat"] <= 3) & (b["cs"] <= 0.05) & (b["cs"] >= 0.005)


def gate_hy(b):
    return (b["cs"] > 0.03) & (b["cs"] <= 0.10)


# ------------------------------------------------------------------- scorers

def score_carry(b, t, i):
    cs = float(b["cs"][i])
    return cs if np.isfinite(cs) else None


def score_mom(b, t, i):
    day = b["day"]; mid = b["mid"]
    i30 = np.searchsorted(day, t - 30, side="right") - 1
    i180 = np.searchsorted(day, t - 180, side="right") - 1
    if i180 < 0 or i30 <= i180:
        return None
    m0, m1 = float(mid[i180]), float(mid[i30])
    if not (np.isfinite(m0) and np.isfinite(m1)) or m0 <= 0:
        return None
    if day[i180] < t - 240:      # too stale a base
        return None
    return m1 / m0 - 1.0


_ISSUER = None


def issuer_map(bonds):
    global _ISSUER
    if _ISSUER is None:
        _ISSUER = {}
        for six in bonds:
            _ISSUER.setdefault(six[:6], []).append(six)
    return _ISSUER


def make_score_rv(bonds):
    imap = issuer_map(bonds)

    def score(b_self, t, i):
        six = None
        # identify self by object match (cheap: attach name once)
        six = b_self.get("_six")
        if six is None:
            return None
        peers = imap.get(six[:6], [])
        if len(peers) < 3:
            return None
        own_cs = float(b_self["cs"][i]); own_mat = int(b_self["mat"][i])
        if not np.isfinite(own_cs):
            return None
        pcs = []
        for p in peers:
            if p == six:
                continue
            bp = bonds[p]
            j = np.searchsorted(bp["day"], t, side="right") - 1
            if j < 0 or t - bp["day"][j] > 30:
                continue
            if abs(int(bp["mat"][j]) - own_mat) > 2:
                continue
            c = float(bp["cs"][j])
            if np.isfinite(c):
                pcs.append(c)
        if len(pcs) < 2:
            return None
        return own_cs - float(np.median(pcs))   # higher = cheap vs peers
    return score


# --------------------------------------------------------------- event sleeve

def sig_reversal(b):
    """>=3pt drop vs the mid 10 days ago, on >=3x trailing volume — the
    price-pressure signature. Uses only past/current rows."""
    day = b["day"]; mid = b["mid"]; qv = b["qv"]; qv90 = b.get("qv90")
    if qv90 is None or len(day) < 12:
        return None
    i10 = np.searchsorted(day, day - 10, side="right") - 1
    prev = np.where(i10 >= 0, mid[np.clip(i10, 0, None)], np.nan)
    ok_gap = (i10 >= 0) & (day - day[np.clip(i10, 0, None)] <= 21)
    drop = prev - mid
    vol_spike = qv > 3.0 * np.maximum(qv90, 1.0)
    return (drop >= 3.0) & ok_gap & vol_spike & np.isfinite(prev)


# ------------------------------------------------------------------ evaluate

def eval_portfolio(bonds, name, score_fn, gate, top_n, controls=3):
    print(f"\n[{name}] top_n={top_n} (IS {2003}-{2015})", flush=True)
    closed = pf.run_portfolio(bonds, score_fn, IS_LO, IS_HI, top_n=top_n,
                              extra_gate=gate)
    fills = pf.positions_to_fills(closed)
    if not fills:
        print("  no fills", flush=True); return None
    r = e2.mtm_nav(bonds, fills)
    days, nav, daily = r
    ps = e2.perf_stats(days, nav, daily)
    n_open = None
    turn = len(fills) / max((IS_HI - IS_LO) / 365.25, 1)
    print(f"  n_roundtrips={len(fills)} trades/yr={turn:.0f} "
          f"cagr={ps['cagr']*100:+.2f}% maxdd={ps['maxdd']*100:.1f}% "
          f"sharpe_m={ps['sharpe_m']:.2f} sharpe_a={ps['sharpe_a']}", flush=True)
    ctl = pf.random_control_portfolio(bonds, lambda b, t, i: (
        score_fn(b, t, i)), IS_LO, IS_HI, top_n, n_reps=controls, extra_gate=gate)
    if ctl:
        cs_ = [c["sharpe_m"] for c in ctl if c["sharpe_m"] is not None]
        cc = [c["cagr"] for c in ctl]
        print(f"  control (random picks, same universe/mechanics): "
              f"cagr={np.mean(cc)*100:+.2f}% sharpe_m={np.mean(cs_):.2f} "
              f"(n={len(ctl)})", flush=True)
        ps["ctl_cagr"] = float(np.mean(cc)); ps["ctl_sharpe_m"] = float(np.mean(cs_))
    ps["n_roundtrips"] = len(fills)
    return ps


def eval_reversal(bonds):
    print(f"\n[rev_event] price-pressure reversal, hold 21-90d (IS)", flush=True)
    fills = e2.run_events(bonds, sig_reversal, min_hold=21, max_hold=90,
                          date_lo=IS_LO, date_hi=IS_HI, cooldown=30)
    ctl = e2.matched_control(bonds, fills, min_hold=21, max_hold=90)
    s = e2.summarize(fills, control=ctl)
    print(f"  n={s.get('n',0)} win={s.get('win_rate',0)*100:.0f}% "
          f"mean={s.get('mean_ret',0)*100:+.2f}% "
          f"excess={s.get('excess_vs_control',0)*100:+.2f}% "
          f"p={s.get('excess_p_boot',1):.3f} hold={s.get('mean_hold',0):.0f}d", flush=True)
    if fills:
        days, nav, daily = e2.mtm_nav(bonds, fills)
        ps = e2.perf_stats(days, nav, daily)
        s["mtm"] = ps
        print(f"  MTM: cagr={ps['cagr']*100:+.2f}% sharpe_m={ps['sharpe_m']:.2f} "
              f"maxdd={ps['maxdd']*100:.1f}%", flush=True)
    return s


def main():
    want = set(sys.argv[1:]) or {"carry5", "carry3", "mom_hy", "rev_event", "rv_issuer"}
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)
    out = {}
    if "carry5" in want:
        out["carry5"] = eval_portfolio(bonds, "carry5", score_carry, gate_carry5, 100)
    if "carry3" in want:
        out["carry3"] = eval_portfolio(bonds, "carry3", score_carry, gate_carry3, 100)
    if "mom_hy" in want:
        out["mom_hy"] = eval_portfolio(bonds, "mom_hy", score_mom, gate_hy, 75)
    if "rev_event" in want:
        out["rev_event"] = eval_reversal(bonds)
    if "rv_issuer" in want:
        out["rv_issuer"] = eval_portfolio(bonds, "rv_issuer", make_score_rv(bonds),
                                          None, 75, controls=2)
    f = ROOT / "research" / "candidates_is.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    old.update({k: v for k, v in out.items() if v})
    f.write_text(json.dumps(old, default=float))
    print(f"\nwrote {f}", flush=True)


if __name__ == "__main__":
    main()
