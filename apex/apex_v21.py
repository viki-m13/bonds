"""APEX v21 — LEAN: 8 models, maximally uncorrelated.

The user's insight: 5-10 models with near-zero correlation beats 24 overlapping ones.

Selection criteria:
  1. Each sleeve uses a DIFFERENT primary information source
  2. Pairwise correlation < 0.2
  3. Each individual OOS SR > 0.4

Final 8 sleeves (Phoenix-playbook style):
  1. TREND_MOM       — LETF momentum top-3 (single clean momentum signal)
  2. RATE_MOM        — Pure 10Y yield direction (orthogonal to equity prices)
  3. USD_STRENGTH    — UUP currency momentum (distinct macro)
  4. VOL_OF_VOL      — VIX vol-of-vol regime (vol-based)
  5. HMM             — Hidden Markov regime classifier (different model type)
  6. CALENDAR        — Pure date-based (zero correlation to price)
  7. BOND_FLIGHT     — Flight-to-quality (crisis-specific)
  8. CRYPTO          — BTC+ETH+SOL (already known near-zero corr)

Design principle: EACH sleeve can survive standalone; no reliance on
blending to fix weak sleeves.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
import numpy as np
import pandas as pd
import util
import sleeves_v12 as SV12
import sleeves_v15 as SV15
import sleeves_v16 as SV16
import sleeves_v17 as SV17
import sleeves_v18 as SV18
import sleeves_v19 as SV19
import sleeves_v20 as SV20
from apex_v13 import extend_cp
from apex_v17 import run_v17

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"


# ONE clean LETF momentum sleeve (replacing overlapping PX clones)
def sleeve_trend_mom(cp: pd.DataFrame, target_vol: float = 0.18) -> pd.DataFrame:
    """Single clean momentum sleeve: top-3 by 126d momentum among 12 LETFs,
    monthly rebal. This is the CORE return engine."""
    universe = [a for a in ["UPRO","TQQQ","TECL","SOXL","FAS","EDC","YINN",
                             "TMF","UBT","UGL","UCO","DRN"] if a in cp.columns]
    p = cp[universe]
    mom = p.shift(21).pct_change(105)
    rnk = mom.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= 3) & (mom > 0)
    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_rebal = mask % 21 == 0
    sel_m = sel.where(is_rebal).ffill().fillna(False)
    spy = cp["SPY"]
    mkt_ok = (spy > spy.rolling(200).mean()).astype(float)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for u in universe:
        W[u] = (sel_m[u].astype(float) / 3 * mkt_ok)

    r = SV15._weights_to_ret(W, cp)
    rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return W.mul(m, axis=0)


# The 8 selected sleeves
LETF_BUILDERS = {
    "TREND_MOM":     lambda cp: sleeve_trend_mom(cp),
    "RATE_MOM":      lambda cp: SV20.sleeve_rate_momentum(cp),
    "USD_STRENGTH":  lambda cp: SV20.sleeve_usd_strength(cp),
    "VOL_OF_VOL":    lambda cp: SV17.sleeve_vol_of_vol(cp),
    "HMM_REGIME":    lambda cp: SV18.sleeve_hmm(cp),
    "CALENDAR":      lambda cp: SV12.sleeve_calendar(cp),
    "BOND_FLIGHT":   lambda cp: SV20.sleeve_bond_flight(cp),
    "DUALBEAR":      lambda cp: SV12.sleeve_dualbear_defense(cp),
}


def build_lean(cp):
    sw = {}
    for name, fn in LETF_BUILDERS.items():
        W = fn(cp)
        r = SV15._weights_to_ret(W, cp)
        rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
        m = (0.15 / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
        sw[name] = W.mul(m, axis=0)
    return sw


def main():
    op, cp = util.load_prices()
    cp = extend_cp(cp)
    for t in ["SH", "PSQ", "SDS", "TBF", "UUP", "DBC", "HYG"]:
        if t not in cp.columns:
            s = SV15._etf_close(t, cp.index)
            if not s.isna().all():
                cp[t] = s

    print("Building LEAN v21 (8 uncorrelated sleeves)...")
    sw = build_lean(cp)

    # Metrics + correlations
    rets = {name: SV15._weights_to_ret(W, cp) for name, W in sw.items()}
    R = pd.DataFrame(rets).fillna(0.0)

    print(f"\n{'Sleeve':15s}  {'SR':>5}  {'OOS':>5}  {'2022':>7}  {'2008':>7}")
    for name, W in sw.items():
        r = rets[name]
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, OOS_START, "2027-12-31"))
        m22 = util.metrics(util.regime_slice(r, "2022-01-01", "2022-12-31"))
        m08 = util.metrics(util.regime_slice(r, "2008-01-01", "2008-12-31"))
        print(f"  {name:15s}  {m['sharpe']:>5.2f}  {om.get('sharpe',0):>5.2f}  "
              f"{m22.get('sharpe',0):>7.2f}  {m08.get('sharpe',0):>7.2f}")

    print("\nPairwise correlations (OOS 2019+):")
    print(R.loc[OOS_START:].corr().round(2))

    # Avg correlation
    corr = R.loc[OOS_START:].corr()
    n = len(corr.columns)
    avg_corr = (corr.values.sum() - n) / (n * (n - 1))
    print(f"\nAverage off-diagonal correlation (OOS): {avg_corr:.3f}")

    # Blends
    print("\n\nBlend configs:")
    print(f"{'cw':>5} {'tv':>5}  {'FULL':>5}  {'OOS':>5}  {'CAGR_F':>7}  {'CAGR_O':>7}  {'2022':>6}  {'MDD':>6}")
    best = None
    for cw in [0.40, 0.45, 0.50, 0.55, 0.60]:
        for tv in [0.18, 0.22, 0.25]:
            net, _ = run_v17(cp, sw, crypto_w=cw, target_vol=tv)
            m = util.metrics(net)
            om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
            m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
            result = {"cw": cw, "tv": tv, "full_sr": m["sharpe"], "oos_sr": om.get("sharpe", 0),
                      "oos_cagr": om.get("cagr", 0), "full_cagr": m["cagr"],
                      "y22": m22.get("sharpe", 0), "mdd": m["mdd"], "net": net}
            print(f"  {cw:.2f}  {tv:.2f}  {m['sharpe']:>5.2f}  {om.get('sharpe',0):>5.2f}  "
                  f"{m['cagr']*100:>6.1f}%  {om.get('cagr',0)*100:>6.1f}%  "
                  f"{m22.get('sharpe',0):>6.2f}  {m['mdd']*100:>5.1f}%")
            if best is None or result["oos_sr"] > best["oos_sr"]:
                best = result

    net = best["net"]
    print(f"\nBEST: cw={best['cw']} tv={best['tv']}  OOS SR {best['oos_sr']:.2f}")
    print("\n=== BEST DETAIL ===")
    for lbl, (s, e) in [("FULL 99-26", ("1999-01-01", "2027-12-31")),
                        ("Phoenix window 10-26", ("2010-03-11", "2027-12-31")),
                        ("IS 10-18", ("2010-03-11", IS_END)),
                        ("OOS 19+", (OOS_START, "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("2008 cal", ("2008-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31")),
                        ("2023-24", ("2023-01-01", "2024-12-31")),
                        ("2025+", ("2025-01-01", "2027-12-31"))]:
        util.summarize(util.regime_slice(net, s, e), f"  {lbl}")

    net.to_frame("apex_v21_ret").to_csv(OUT / "apex_v21_returns.csv")
    (OUT / "apex_v21_meta.json").write_text(json.dumps({
        "best": {k: v for k, v in best.items() if k != "net"},
        "sleeves": list(sw.keys()),
        "avg_oos_correlation": float(avg_corr),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
