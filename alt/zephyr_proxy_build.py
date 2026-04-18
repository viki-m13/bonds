"""Build ZEPHYR returns with proxies, extended to 2005-04 (match Sharpe page).

Proxy tiers (most recent wins):
  TIER 1 - Live (2020-10+): real ETFs
  TIER 2 - Near-proxy (2011-06 to 2020-10):
    JAAA -> 60% FLOT + 40% BKLN
    JPST -> MINT
  TIER 3 - Early-proxy (2005-04 to 2011-06):
    JAAA / JPST / MINT / FLOT -> SHY (1-3y Treasuries)
    BKLN / SRLN                -> SHY (no bank-loan ETF pre-2011)
    GLD                         -> GLD (live from 2004-11) or SHY before
  BIL (cash comparator for regime gate): use SHY pre-2007-05

All proxies preserve ZEPHYR's fixed weights. Pre-2011 the backtest is
essentially a short-duration Treasury portfolio with a small gold sleeve
— structurally different from the live strategy. The TIER 3 numbers are
a conservative floor, not a simulation of what JAAA would have returned.

Also computes SPY and AGG on the same date grid for comparison.

Output: data/results/zephyr_proxy_returns.csv
  columns: Close, SPY, AGG, source, tier
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"

WEIGHTS = {
    "JAAA": 0.32, "JPST": 0.28, "MINT": 0.15, "BKLN": 0.10,
    "SRLN": 0.05, "FLOT": 0.05, "GLD": 0.05,
}


def load_etf(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()


def load_fred(s):
    p = FRED / f"{s}.csv"
    if not p.exists(): return None
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    return pd.to_numeric(d, errors="coerce").sort_index()


def pct(s, dates):
    return s.reindex(dates).ffill().pct_change().fillna(0)


def main():
    print("Building ZEPHYR extended to Sharpe-page start...")
    # Align to Sharpe strategy start (first available bond/SPY data here).
    spy = load_etf("SPY")
    start = max(pd.Timestamp("2005-04-25"), spy.index.min())
    dates = spy.loc[start:].index

    # Daily-return series for each constituent, with tiered proxies.
    real = {t: load_etf(t) for t in WEIGHTS}
    flot = load_etf("FLOT"); bkln = load_etf("BKLN"); mint = load_etf("MINT")
    shy = load_etf("SHY")
    gld = load_etf("GLD")

    # Returns
    r_shy = pct(shy, dates)
    r_flot = pct(flot, dates) if flot is not None else r_shy
    r_bkln = pct(bkln, dates) if bkln is not None else r_shy
    r_mint = pct(mint, dates) if mint is not None else r_shy
    r_gld  = pct(gld, dates)  if gld  is not None else pd.Series(0.0, index=dates)

    # Availability masks
    def avail(s):
        return pd.Series(False if s is None else (s.index.min() <= dates), index=dates)

    contribs = pd.Series(0.0, index=dates)
    tier = pd.Series(3, index=dates, dtype=int)  # 3 = early-proxy by default

    def apply(weight, live_ret, live_mask, near_ret, near_mask, early_ret):
        """Return weighted contribution + tier-stamp per date."""
        val = pd.Series(0.0, index=dates)
        val = val.where(~live_mask, live_ret * weight)
        val = val.where(live_mask | ~near_mask, near_ret * weight)
        val = val.where(live_mask | near_mask, early_ret * weight)
        return val

    # JAAA: live JAAA > FLOT+BKLN blend > SHY
    jaaa_live = real["JAAA"]
    jaaa_avail = avail(jaaa_live)
    near_ret = 0.6 * r_flot + 0.4 * r_bkln
    near_mask = avail(flot) & avail(bkln)
    contribs = contribs + apply(WEIGHTS["JAAA"],
                                pct(jaaa_live, dates), jaaa_avail,
                                near_ret, near_mask & ~jaaa_avail,
                                r_shy)

    # JPST: live JPST > MINT > SHY
    jpst = real["JPST"]
    jpst_avail = avail(jpst)
    contribs = contribs + apply(WEIGHTS["JPST"],
                                pct(jpst, dates), jpst_avail,
                                r_mint, avail(mint) & ~jpst_avail,
                                r_shy)

    # MINT: live MINT > SHY
    contribs = contribs + apply(WEIGHTS["MINT"],
                                r_mint, avail(mint),
                                pd.Series(0.0, index=dates), pd.Series(False, index=dates),
                                r_shy)

    # BKLN: live BKLN > SHY
    contribs = contribs + apply(WEIGHTS["BKLN"],
                                r_bkln, avail(bkln),
                                pd.Series(0.0, index=dates), pd.Series(False, index=dates),
                                r_shy)

    # SRLN: live SRLN > BKLN > SHY
    srln = real["SRLN"]
    srln_avail = avail(srln)
    contribs = contribs + apply(WEIGHTS["SRLN"],
                                pct(srln, dates), srln_avail,
                                r_bkln, avail(bkln) & ~srln_avail,
                                r_shy)

    # FLOT: live FLOT > SHY
    contribs = contribs + apply(WEIGHTS["FLOT"],
                                r_flot, avail(flot),
                                pd.Series(0.0, index=dates), pd.Series(False, index=dates),
                                r_shy)

    # GLD: live GLD always (starts 2004-11, pre-start)
    contribs = contribs + WEIGHTS["GLD"] * r_gld

    # Tier tracking
    tier = pd.Series(3, index=dates, dtype=int)
    tier[avail(flot) & avail(bkln) & avail(mint)] = 2
    tier[jaaa_avail & jpst_avail] = 1  # both late ETFs live -> TIER 1

    # Source label: live iff everything real
    all_live = jaaa_avail & jpst_avail & avail(mint) & avail(bkln) & srln_avail & avail(flot) & avail(gld)
    source = np.where(all_live, "live", "proxy")

    # Regime gate (same as main ZEPHYR)
    hy = load_fred("BAMLH0A0HYM2")
    dgs = load_fred("DGS10")
    hy_d = hy.reindex(dates).ffill() if hy is not None else pd.Series(5.0, index=dates)
    dg_d = dgs.reindex(dates).ffill() if dgs is not None else pd.Series(3.0, index=dates)
    g_credit = np.clip((8.0 - hy_d) / (8.0 - 5.0), 0, 1)
    g_rate = ((dg_d - dg_d.shift(63)) < 0.7).astype(float)
    g = (g_credit * g_rate).shift(1).fillna(0)
    bil = load_etf("BIL")
    r_bil = pct(bil, dates) if bil is not None else r_shy
    r_bil = r_bil.where(avail(bil), r_shy)  # SHY as pre-BIL cash
    gated = g * contribs + (1 - g) * r_bil - (0.0002 / 21)  # monthly-drag haircut

    # Benchmarks
    r_spy = pct(spy, dates)
    agg = load_etf("AGG")
    r_agg = pct(agg, dates) if agg is not None else pd.Series(0.0, index=dates)

    out = pd.DataFrame({
        "Close": gated,
        "SPY": r_spy,
        "AGG": r_agg,
        "source": source,
        "tier": tier.values,
    })
    out.index.name = "Date"
    out.to_csv(RESULTS / "zephyr_proxy_returns.csv")

    # Summary
    for label, mask in [("full", slice(None)),
                        ("tier1 live", out["tier"] == 1),
                        ("tier2 near-proxy", out["tier"] == 2),
                        ("tier3 early-proxy", out["tier"] == 3)]:
        r = gated.loc[out.index[mask] if not isinstance(mask, slice) else out.index]
        if len(r) < 2: continue
        ar = r.mean() * 252
        av = r.std() * np.sqrt(252)
        sr = ar / av if av > 0 else 0
        cum = (1 + r).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        d1, d2 = r.index[0].date(), r.index[-1].date()
        print(f"  {label:20s} {d1}..{d2} ({len(r)/252:.1f}y): "
              f"SR={sr:.2f} Ret={ar*100:.2f}% Vol={av*100:.2f}% MDD={mdd*100:.1f}%")


if __name__ == "__main__":
    main()
