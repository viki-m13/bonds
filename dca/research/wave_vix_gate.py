"""WAVE/ROTATOR x FULCRUM: does the VIX term-structure stress signal beat the
ROTATOR's SPY-210dma bear switch as a crash filter on a PIT individual-stock
strategy?

Same survivorship-controlled PIT harness as SUMMIT/ROTATOR (dca/protocol.py):
244-window grid + named regimes, biweekly DCA, next-open fills, 5 bps, vs SPY/QQQ.
Only the BEAR (de-risk to cash) definition changes:

  baseline   : SPY < 210dma                         (the published ROTATOR)
  vix_repl   : VIX > VIX3M  (term-structure backwardation = FULCRUM stress)
  vix_or     : (SPY<210dma) OR (VIX>VIX3M)          (belt-and-suspenders)
  vix_evrp   : (SPY<210dma) OR (VIX>VIX3M and eVRP<=0)  (full FULCRUM stress gate)

VIX3M (CBOE) starts 2009-09, so the VIX term-structure flag is False before that
(GFC protection in the OR variants still comes from SPY-210dma). Causal: every
signal at date d uses data through close of d; engine fills next open.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data as data_mod, protocol
import strategy_rotator as R

ROOT = data_mod.ROOT

def load_panel_from_summit():
    sp = pd.read_parquet(os.path.join(ROOT, "data", "pit", "summit_panel.parquet"))
    P = {f: sp[f].copy() for f in ["open", "close", "volume", "member"]}
    P["member"] = P["member"].astype(bool)
    return P

# monkeypatch so protocol.get_shared() and strategy_rotator use this panel
_P = load_panel_from_summit()
data_mod.build_panel = lambda force=False: _P

def vix_term_structure(index):
    vix = pd.read_csv(os.path.join(ROOT, "data", "fred", "VIXCLS.csv"), parse_dates=["Date"]).set_index("Date")["VIXCLS"]
    vix = pd.to_numeric(vix, errors="coerce")
    v3m = pd.read_csv(os.path.join(ROOT, "data", "cboe", "VIX3M.csv"), parse_dates=["Date"]).set_index("Date")["VIX3M"]
    spy = data_mod.load_benchmark("SPY")["Close"]
    rv = spy.pct_change().rolling(10).std() * np.sqrt(252) * 100      # 10d realized vol, %
    vix = vix.reindex(index).ffill(); v3m = v3m.reindex(index).ffill(); rv = rv.reindex(index).ffill()
    backward = (vix > v3m).fillna(False)
    evrp_neg = ((vix - rv) <= 0).fillna(False)
    return backward, evrp_neg

def build_variant(P, bear):
    """ROTATOR scores/sell with an arbitrary bear (cash) boolean Series."""
    score = R.leadership(P).where(~R._max_screen(P))
    bear_np = bear.reindex(P["close"].index).fillna(False).to_numpy()
    sc = score.to_numpy(copy=True); sc[bear_np, :] = np.nan
    scores = pd.DataFrame(sc, index=score.index, columns=score.columns)
    member = P["close"].notna() & P["member"]
    rank = R.leadership(P).where(member).rank(axis=1, ascending=False)
    sell = (~(rank <= R.TOP_KEEP).fillna(False)).to_numpy(copy=True)
    sell[bear_np, :] = True
    sell = pd.DataFrame(sell, index=rank.index, columns=rank.columns)
    return scores, sell

if __name__ == "__main__":
    idx = _P["close"].index
    spy = data_mod.load_benchmark("SPY")["Close"].reindex(idx).ffill()
    spy_bear = spy < spy.rolling(R.SPY_MA).mean()
    backward, evrp_neg = vix_term_structure(idx)

    variants = {
        "rotator_baseline_SPY210": spy_bear,
        "rotator_vix_replace":     backward,
        "rotator_vix_OR":          spy_bear | backward,
        "rotator_vix_evrp_OR":     spy_bear | (backward & evrp_neg),
    }
    cards = {}
    for name, bear in variants.items():
        sc, sl = build_variant(_P, bear)
        cards[name] = protocol.evaluate_signal(sc, name, k=3, sell=sl)

    print("\n==== SUMMARY (k=3, biweekly, 5bps; PIT grid) ====")
    hdr = f"{'variant':28s} {'win_spy':>8s} {'win_qqq':>8s} {'med_vs_spy':>11s} {'worst_vs_spy':>13s} {'full_mult':>10s}"
    print(hdr)
    for n, c in cards.items():
        print(f"{n:28s} {c['win_spy']:>8.0%} {c['win_qqq']:>8.0%} {c['med_vs_spy']:>+11.1%} {c['worst_vs_spy']:>+13.1%} {c['full_mult']:>10.2f}")
    print("\n==== KEY REGIME WINDOWS (vs_spy) ====")
    regs = ["vol_2018", "covid_2020", "bear_2022", "ai_bull_2023_2026", "GFC_2007_2009"]
    print(f"{'variant':28s} " + " ".join(f"{r[:10]:>12s}" for r in regs))
    for n, c in cards.items():
        print(f"{n:28s} " + " ".join(f"{c['regimes'].get(r,{}).get('vs_spy',float('nan')):>+12.1%}" for r in regs))
