"""
FULCRUM — an UNLEVERED, capital-efficient VIX carry-&-hedge overlay that beats SPY.

WHAT THIS IS
------------
An improvement on the Concretum "Volatility Edge" VIX-ETN strategy (Strategy 4).
Fully invested, gross exposure <= 100% at all times -- NO LEVERAGE, NO MARGIN.

THE DIAGNOSIS (why the base strategy cannot beat SPY)
----------------------------------------------------
The base strategy is a good short-vol signal (harvests the variance risk premium,
gated by the VIX/VIX3M term structure + an eVRP realized-vol filter; Sharpe ~0.84).
But it sizes at "VIX%", so it deploys only ~31% of capital and PARKS ~69% IN CASH.
That dead cash is why it trails SPY despite a fine Sharpe.

THE EDGE (two unlevered constructions; both keep the regime engine unchanged)
----------------------------------------------------------------------------
Treat the book as two sleeves -- an equity sleeve (SPY) and a vol sleeve (the
regime-driven SVXY/VXX position) -- and split a 100% budget between them.

  FULCRUM-U  : vol sleeve funded out of equity, sized at the paper's VIX% rule
               (w_spy = 1 - vol_notional). Zero tuned parameters.
  FULCRUM-RB : RISK-BUDGETED split -- weight the two sleeves by inverse trailing
               volatility (risk parity) so each contributes equal risk and the
               short-vol sleeve auto-shrinks when it becomes dangerous. This is
               the better version: it lifts Sharpe to ~1.04 (full) / 1.08 (OOS).

NO LEVERAGE: in both, w_spy + vol_notional = 1 (any shortfall sits in T-bills).

WHAT WAS TESTED AND **REJECTED** (overfitting / honesty control)
----------------------------------------------------------------
Tuned on in-sample 2011-2018, all FAILED out-of-sample or were rejected by the
IS grid (which kept choosing the neutral setting): amplified long-vol hedge
(k_h>1); crash-throttle on the short leg; converting the cash regime to a hedge;
term-structure equity de-risking; a Moreira-Muir vol-managed wrapper; and brute
scaling of the vol sleeve (saturates ~Sharpe 0.98). The risk-budget grid selected
k_short=1 (pure inverse-vol) -- no fudge factor. The robust edge is capital
structure + risk allocation, NOT signal complexity.

A LOOK-AHEAD BUG I CAUGHT (documented deliberately)
---------------------------------------------------
An earlier exploratory version shifted the whole sleeve-return series, pairing
tomorrow's regime signal with today->tomorrow's return. That produced a fake
Sharpe ~2.0 / CAGR ~90%. It is a bug, not alpha. The code below aligns signal at
close_t with the t->t+1 return (no look-ahead); honest Sharpe is ~1.0-1.08.

VALIDATION DISCIPLINE
---------------------
- No look-ahead: regime/weights from close_t; returns earned close_t -> close_{t+1}.
- Vol exposure via the +1x short-term VIX-futures index (VIXY), which reproduces
  the literal 2x-SVXY notebook sizing post-2018 (cross-checked in the validator).
- Costs 5 bps/side on notional turnover. Reported IS (2011-18)/OOS (2019-26)/full.
"""
import numpy as np, pandas as pd
import sys; sys.path.insert(0, "alt")
import vix_voledge_validate as V

IS_END, OOS_START = "2018-12-31", "2019-01-01"
COST_BPS = 5.0
RB_WINDOW = 60          # trailing-vol window for risk budgeting (IS-selected; robust 20/40/60)

def regime(d):
    """Concretum regime engine -> per-$ vol-sleeve exposure to the +1x vol index,
    and the VIX%-rule dollar notional. r1 short-full, r2 short-half, r3 long-hedge."""
    vixpct = d["vix"]/100.0
    r1 = (d["evrp"] > 0) & d["contango"]
    r2 = (d["evrp"] <= 0) & d["contango"]
    r3 = (d["evrp"] <= 0) & d["backward"]
    perdollar = pd.Series(0.0, index=d.index)   # index-exposure per $1 in the sleeve
    perdollar[r1] = -0.5; perdollar[r2] = -0.5; perdollar[r3] = +1.0   # SVXY -0.5x / VXX +1x
    n_vixpct = pd.Series(0.0, index=d.index)     # paper's VIX% dollar notional
    n_vixpct[r1] = 2*vixpct[r1]; n_vixpct[r2] = vixpct[r2]; n_vixpct[r3] = vixpct[r3]
    return perdollar, n_vixpct

def fulcrum_u(d, df, cost_bps=COST_BPS):
    perdollar, n = regime(d)
    spy_fwd = df["spy"].pct_change().shift(-1).reindex(d.index)
    vix_fwd = df["vixy"].pct_change().shift(-1).reindex(d.index)
    w_spy = (1.0 - n).clip(lower=0.0)
    tc = (n.diff().abs().fillna(n.abs()) + w_spy.diff().abs().fillna(0)) * (cost_bps/1e4)
    return (w_spy*spy_fwd + perdollar*n*vix_fwd - tc).dropna()

def fulcrum_rb(d, df, window=RB_WINDOW, cost_bps=COST_BPS):
    perdollar, _ = regime(d)
    spy_r = df["spy"].pct_change().reindex(d.index)
    vix_r = df["vixy"].pct_change().reindex(d.index)
    spy_fwd, vix_fwd = spy_r.shift(-1), vix_r.shift(-1)
    sleeve_contemp = perdollar * vix_r                          # for trailing-vol estimate
    vs = spy_r.rolling(window).std(); vv = sleeve_contemp.rolling(window).std()
    w_spy = (( 1/vs )/( 1/vs + 1/vv )).shift(1).clip(0,1).fillna(1.0)   # inverse-vol, lagged
    w_sl = 1.0 - w_spy
    tc = (w_spy.diff().abs().fillna(0) + w_sl.diff().abs().fillna(0)) * (cost_bps/1e4)
    return (w_spy*spy_fwd + w_sl*perdollar*vix_fwd - tc).dropna()   # gross = w_spy+w_sl = 1

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
    spy  = df["spy"].pct_change().shift(-1).reindex(d.index).dropna()
    base = V.run(d, df["vixy"].pct_change(), "base", cost_bps=COST_BPS)
    fU, fRB = fulcrum_u(d, df), fulcrum_rb(d, df)
    idx = fRB.index                                  # common (RB needs warmup)
    cut = lambda r, sl: r[r.index<=IS_END] if sl=="is" else r[r.index>=OOS_START] if sl=="oos" else r
    for label, sl in [("IN-SAMPLE 2011-2018","is"),("OUT-OF-SAMPLE 2019-2026","oos"),("FULL 2011-2026","full")]:
        table([stats(cut(spy.reindex(idx),sl),"SPY buy&hold"),
               stats(cut(base.reindex(idx),sl),"Base VolEdge (cash-heavy)"),
               stats(cut(fU.reindex(idx),sl),"FULCRUM-U (unlevered)"),
               stats(cut(fRB,sl),"FULCRUM-RB (risk-budgeted, unlevered)")], label)

    both = pd.concat([fRB, spy.reindex(idx)], axis=1, sort=True).dropna(); both.columns=["F","S"]
    dn = both[both.S<0]
    print(f"\nFULL corr(FULCRUM-RB,SPY)={both.F.corr(both.S):.2f}  beta={both.F.cov(both.S)/both.S.var():.2f}"
          f"  down-day capture={dn.F.mean()/dn.S.mean():.2f}")
    comp = (pd.DataFrame({"FULCRUM-RB":(1+fRB).groupby(fRB.index.year).prod()-1,
                          "SPY":(1+spy.reindex(idx)).groupby(idx.year).prod()-1})*100)
    print("\n===== CALENDAR-YEAR RETURNS (%) =====\n" + comp.round(1).to_string())

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11,6))
        for s, lab, c in [(spy.reindex(idx),"SPY buy&hold","#888"),
                          (base.reindex(idx),"Base VolEdge (cash-heavy)","#d62728"),
                          (fU.reindex(idx),"FULCRUM-U (unlevered)","#1f77b4"),
                          (fRB,"FULCRUM-RB (risk-budgeted, unlevered)","#2ca02c")]:
            ax.plot(idx, (1+s.fillna(0)).cumprod(), label=lab, color=c, lw=1.6)
        ax.axvline(pd.Timestamp(OOS_START), ls="--", c="k", alpha=.4)
        ax.text(pd.Timestamp(OOS_START), ax.get_ylim()[1]*0.9, " out-of-sample ->", fontsize=9)
        ax.set_yscale("log"); ax.set_title("FULCRUM (unlevered) vs SPY — growth of $1 (log), 2011-2026")
        ax.legend(loc="upper left"); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig("alt/fulcrum_equity_curve.png", dpi=120); print("\nSaved alt/fulcrum_equity_curve.png")
    except Exception as ex:
        print("plot skipped:", ex)
