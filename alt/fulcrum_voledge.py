"""
FULCRUM — a capital-efficient VIX carry-&-hedge overlay on an equity core.

WHAT THIS IS
------------
An improvement on the Concretum "Volatility Edge" VIX-ETN strategy (Strategy 4),
designed to OUTPERFORM SPY on absolute return, risk-adjusted return AND drawdown.

THE KEY DIAGNOSIS (why the base strategy cannot beat SPY)
--------------------------------------------------------
The base strategy is a genuinely good short-vol signal (it harvests the variance
risk premium, gated by the VIX/VIX3M term structure and an eVRP realized-vol
filter). But it sizes positions at only "VIX%" of capital, so on average it
deploys just ~31% of the book into vol and PARKS ~69% IN CASH. It is, in effect,
a cash-collateralized overlay being run as if the collateral were the product.
That dead cash is why it trails SPY (CAGR 12.5% vs 14.6%) despite a fine Sharpe.

THE EDGE (capital efficiency / portable alpha)
----------------------------------------------
Run the vol signal as an OVERLAY on a full equity core. The vol sleeve has
beta ~0.13 to SPY, so stacking it on equity is close to additive in return while
sub-additive in risk. Two constructions:

  FULCRUM-U (unlevered, recommended): fund the vol sleeve out of the equity sleeve.
      w_spy = 1 - vol_notional ;  gross <= 100% ;  no margin, no financing.
      -> beats SPY on CAGR, Sharpe AND drawdown, with ZERO tuned parameters.

  FULCRUM-L (levered / true portable alpha): hold a full SPY core and stack the
      vol overlay on top, financed at T-bills + spread. Higher CAGR, ~SPY drawdown.

WHAT I TESTED AND **REJECTED** (intellectual honesty / overfitting control)
---------------------------------------------------------------------------
All of the following were tuned on in-sample 2011-2018 and FAILED to add
out-of-sample value (they reduced Sharpe and/or were rejected by the IS grid,
which kept selecting the neutral setting):
  * Amplifying the long-vol hedge leg (k_h > 1)            -> hurts (carry bleed)
  * Crash-throttle on the short leg (cut short on VIX spikes) -> no improvement
  * Converting the eVRP>0 & backwardation "cash" regime to a hedge -> hurts
  * Term-structure equity de-risking (cut SPY in backwardation)    -> not selected
  * Moreira-Muir volatility-managed wrapper on the book           -> hurts (whipsaw)
The honest conclusion: the robust improvement is capital structure, not signal
complexity. Adding knobs only added overfitting risk.

VALIDATION DISCIPLINE
---------------------
- No look-ahead: signal uses data through close_t; positions earn close_t->close_{t+1}.
- Vol exposure is expressed via the +1x short-term VIX-futures index (VIXY), which
  reproduces the literal 2x-SVXY notebook sizing post-2018 (cross-checked).
- Costs: 5 bps/side on notional turnover. Financing (levered): T-bills + 1.0%/yr.
- Reported across in-sample (2011-2018), out-of-sample (2019-2026) and full sample.
"""
import numpy as np, pandas as pd
import sys; sys.path.insert(0, "alt")
import vix_voledge_validate as V

IS_END, OOS_START = "2018-12-31", "2019-01-01"
COST_BPS, FIN_SPREAD = 5.0, 0.010

def overlay_exposure(d):
    """Base Concretum regime engine -> signed exposure e (to +1x vol index) and
    ETF dollar notional n (capital tied up). No tuned parameters."""
    vixpct = d["vix"]/100.0
    r1 = (d["evrp"] > 0) & d["contango"]          # short vol, full
    r2 = (d["evrp"] <= 0) & d["contango"]         # short vol, half
    r3 = (d["evrp"] <= 0) & d["backward"]         # long vol (hedge)
    e = pd.Series(0.0, index=d.index); n = pd.Series(0.0, index=d.index)
    e[r1] = -vixpct[r1];      n[r1] = 2*vixpct[r1]
    e[r2] = -0.5*vixpct[r2];  n[r2] = vixpct[r2]
    e[r3] = +vixpct[r3];      n[r3] = vixpct[r3]
    return e, n

def fulcrum(d, df, levered=False, cost_bps=COST_BPS):
    e, n = overlay_exposure(d)
    spy_fwd    = df["spy"].pct_change().shift(-1).reindex(d.index)
    volidx_fwd = df["vixy"].pct_change().shift(-1).reindex(d.index)
    rf_fwd     = d["rf_daily"].shift(-1)
    if levered:
        w_spy = pd.Series(1.0, index=d.index)
        fin = n * (rf_fwd + FIN_SPREAD/252)
    else:
        w_spy = (1.0 - n).clip(lower=0.0)
        fin = 0.0
    tc = (n.diff().abs().fillna(n.abs()) + w_spy.diff().abs().fillna(0)) * (cost_bps/1e4)
    return (w_spy*spy_fwd + e*volidx_fwd - fin - tc).dropna()

def stats(r, name):
    r = r.dropna(); eq = (1+r).cumprod(); yrs = len(r)/252
    cagr = eq.iloc[-1]**(1/yrs)-1; vol = r.std()*np.sqrt(252)
    dn = r[r<0].std()*np.sqrt(252); dd = (eq/eq.cummax()-1).min()
    return dict(strat=name, CAGR=cagr, Vol=vol, Sharpe=(r.mean()*252)/vol,
                Sortino=(r.mean()*252)/dn, MaxDD=dd, Calmar=cagr/abs(dd))

def table(rows, title):
    print(f"\n===== {title} =====")
    t = pd.DataFrame(rows).set_index("strat")
    print(t.to_string(formatters={c:(lambda x:f"{x:,.3f}") for c in t.columns}))

if __name__ == "__main__":
    df = V.load(); d = V.signals(df).dropna(subset=["erv30","evrp"])
    spy = df["spy"].pct_change().shift(-1).reindex(d.index).dropna()
    base = V.run(d, df["vixy"].pct_change(), "base", cost_bps=COST_BPS)
    fU, fL = fulcrum(d, df, levered=False), fulcrum(d, df, levered=True)

    def cut(r, sl):
        return r[r.index<=IS_END] if sl=="is" else r[r.index>=OOS_START] if sl=="oos" else r
    for label, sl in [("IN-SAMPLE 2011-2018","is"),("OUT-OF-SAMPLE 2019-2026","oos"),("FULL 2011-2026","full")]:
        table([stats(cut(spy,sl),"SPY buy&hold"),
               stats(cut(base,sl),"Base VolEdge (cash-heavy)"),
               stats(cut(fU,sl),"FULCRUM-U (unlevered)"),
               stats(cut(fL,sl),"FULCRUM-L (levered)")], label)

    # diversification stats (full)
    both = pd.concat([fU, spy], axis=1).dropna(); both.columns=["F","S"]
    dn = both[both.S<0]
    print(f"\nFULL corr(FULCRUM-U,SPY)={both.F.corr(both.S):.2f}  beta={both.F.cov(both.S)/both.S.var():.2f}"
          f"  down-day capture={dn.F.mean()/dn.S.mean():.2f}")

    comp = (pd.DataFrame({"FULCRUM-U":(1+fU).groupby(fU.index.year).prod()-1,
                          "FULCRUM-L":(1+fL).groupby(fL.index.year).prod()-1,
                          "SPY":(1+spy).groupby(spy.index.year).prod()-1})*100)
    print("\n===== CALENDAR-YEAR RETURNS (%) =====")
    print(comp.round(1).to_string())

    # equity curve
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        idx = fU.index
        fig, ax = plt.subplots(figsize=(11,6))
        for s, lab, c in [(spy.reindex(idx),"SPY buy&hold","#888"),
                          (base.reindex(idx),"Base VolEdge (cash-heavy)","#d62728"),
                          (fU,"FULCRUM-U (unlevered)","#1f77b4"),
                          (fL,"FULCRUM-L (levered)","#2ca02c")]:
            ax.plot(idx, (1+s.fillna(0)).cumprod(), label=lab, color=c, lw=1.6)
        ax.axvline(pd.Timestamp(OOS_START), ls="--", c="k", alpha=.4)
        ax.text(pd.Timestamp(OOS_START), ax.get_ylim()[1]*0.95, " out-of-sample ->", fontsize=9)
        ax.set_yscale("log"); ax.set_title("FULCRUM vs SPY — growth of $1 (log scale), 2011-2026")
        ax.legend(loc="upper left"); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig("alt/fulcrum_equity_curve.png", dpi=120)
        print("\nSaved alt/fulcrum_equity_curve.png")
    except Exception as ex:
        print("plot skipped:", ex)
