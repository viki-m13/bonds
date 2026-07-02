"""Sleeve A — Connors-style NDX-100 mean-reversion on TRUE PIT membership
(2015-01 .. 2026-06), adjusted OHLC, with an execution-timing study and a
full cost model.

Execution modes (the question this script answers):
  moc        signal ~3:55pm, enter Market-On-Close at close t   (fastest)
  moo        run model after close t, enter Market-On-Open at open t+1
  nextclose  run after close t, enter MOC at close t+1           (laziest)
  limit      original spec: run after close t, work a limit at
             close - 0.9*ATR(5) during day t+1 (passive fill)

All modes share the same exits: profit-target limit at entry-signal close
+ 0.5*ATR (working intraday, passive); close > previous high; 9-bar time
stop. Discretionary (close-based) exits execute per-mode: MOC same close
(moc), next open (moo/limit), next close (nextclose).

Costs per fill via lib.fill_cost_bps: commission $0.005/sh, auction fills
+1bp+impact, passive fills commission-only, plus a fill-through
requirement on limit fills (price must trade N bps through the limit) to
kill optimistic-fill bias. Leverage (if any) is financed at FEDFUNDS+150bp.
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import (load_n100, load_etf, wilder_rsi, atr_simple, fill_cost_bps,
                 financing_daily, riskfree_daily, stats, fmt, OUT)

t0 = time.time()
P = load_n100()
O, H, L, C, V, MEM = (P[k] for k in ["open", "high", "low", "close",
                                     "volume", "member"])
dates = C.index
prevC, prevH = C.shift(1), H.shift(1)
atr = atr_simple(H, L, C, 5)
atrp = atr / C
ema200 = C.ewm(span=200, min_periods=200).mean()
vol20 = V.rolling(20, min_periods=20).mean()
dadv = (C * V).rolling(20, min_periods=20).mean()          # dollar ADV
rsi2 = wilder_rsi(C, 2)

TRADE_START = pd.Timestamp("2015-01-02")   # first day with PIT membership
uni = MEM & (C > 5) & (vol20 > 100_000) & (C > ema200) & (atrp > 0.03)
trig = (C < prevC * 0.97) & uni

tk = list(C.columns)
col = {t: j for j, t in enumerate(tk)}
Ov, Hv, Lv, Cv = O.values, H.values, L.values, C.values
PHv, ATRv, ATRPv = prevH.values, atr.values, atrp.values
DADVv, RSIv, TRIGv = dadv.values, rsi2.values, trig.values
i0 = dates.searchsorted(TRADE_START)
fin = financing_daily(dates).values
rf = riskfree_daily(dates)


def run(mode, rsi_max=15.0, pos_frac=0.20, maxpos=10, cash_capped=True,
        fill_through_bps=5.0, costs=True, E0=1e5):
    """Event-loop backtest. Returns (equity Series, trades DataFrame)."""
    cash, pos, eq, trades = E0, {}, [], []
    pend = []            # entry orders generated at close t, live on day t+1
    thr = fill_through_bps / 1e4

    def cost(style, px, notional, adv):
        return fill_cost_bps(style, px, notional, adv) / 1e4 if costs else 0.0

    for i in range(i0, len(dates)):
        # ---- 1. exits on today's bar (positions opened before today) -------
        for t in list(pos):
            p, j = pos[t], col[t]
            hi, lo, cl, ph = Hv[i, j], Lv[i, j], Cv[i, j], PHv[i, j]
            if not np.isfinite(cl):
                p["bars"] += 1
                if p["bars"] > 15:      # halted/delisted: liquidate at last mark
                    cash += p["sh"] * p["last"] * (1 - cost("auction", p["last"],
                            p["sh"] * p["last"], p["adv"]))
                    trades.append((t, p["ep"], p["last"], "delist", p["bars"]))
                    del pos[t]
                continue
            p["last"] = cl
            ex = px = style = None
            if p.get("exit_pending"):          # discretionary exit queued
                if mode == "moo" or mode == "limit":
                    px, style, ex = Ov[i, j], "auction", p["exit_pending"]
                elif mode == "nextclose":
                    px, style, ex = cl, "auction", p["exit_pending"]
                if px is not None and np.isfinite(px):
                    notional = p["sh"] * px
                    cash += notional * (1 - cost(style, px, notional, p["adv"]))
                    trades.append((t, p["ep"], px, ex, p["bars"]))
                    del pos[t]
                    continue
                p["exit_pending"] = None       # open missing: retry via signals
            if np.isfinite(hi) and hi >= p["tgt"] * (1 + thr):
                ex, px, style = "target", p["tgt"], "passive"
            elif np.isfinite(ph) and cl > ph:
                ex = "close>prevH"
            elif p["bars"] >= 9:
                ex = "9bars"
            if ex and px is None:              # close-based discretionary exit
                if mode == "moc":
                    px, style = cl, "auction"
                else:                          # execute at next open/close
                    p["exit_pending"] = ex
                    p["bars"] += 1
                    continue
            if ex and np.isfinite(px):
                notional = p["sh"] * px
                cash += notional * (1 - cost(style, px, notional, p["adv"]))
                trades.append((t, p["ep"], px, ex, p["bars"]))
                del pos[t]
                continue
            p["bars"] += 1

        # ---- 2. entries ----------------------------------------------------
        def buy(t, fill, tgt, adv, style):
            nonlocal cash
            mv = sum(pp["sh"] * Cv[i, col[tt]] for tt, pp in pos.items()
                     if np.isfinite(Cv[i, col[tt]]))
            equity = cash + mv
            sh = int((pos_frac * equity) // fill)
            if cash_capped:
                avail = cash - 1
                sh = min(sh, int(avail // (fill * 1.001)))
            if sh <= 0:
                return
            notional = sh * fill
            cash -= notional * (1 + cost(style, fill, notional, adv))
            pos[t] = {"sh": sh, "ep": fill, "tgt": tgt, "bars": 0,
                      "adv": adv, "last": fill, "exit_pending": None}

        for t, lim, tgt, adv in pend:
            if t in pos or len(pos) >= maxpos:
                continue
            j = col[t]
            if mode == "limit":
                op, lo = Ov[i, j], Lv[i, j]
                if not np.isfinite(lo) or lo > lim * (1 - thr):
                    continue                    # never traded through the limit
                fill = min(op, lim) if np.isfinite(op) else lim
                style = "auction" if (np.isfinite(op) and op < lim) else "passive"
                buy(t, fill, tgt, adv, style)
            elif mode == "moo":
                op = Ov[i, j]
                if np.isfinite(op):
                    buy(t, op, tgt, adv, "auction")
            elif mode == "nextclose":
                cl = Cv[i, j]
                if np.isfinite(cl):
                    buy(t, cl, tgt, adv, "auction")
        pend = []

        # ---- 3. today's signals -> orders ---------------------------------
        cand = [(ATRPv[i, col[t]], t) for t in tk
                if TRIGv[i, col[t]] and RSIv[i, col[t]] < rsi_max
                and np.isfinite(ATRPv[i, col[t]])]
        cand.sort(reverse=True)
        orders = [(t, Cv[i, col[t]] - 0.9 * ATRv[i, col[t]],
                   Cv[i, col[t]] + 0.5 * ATRv[i, col[t]], DADVv[i, col[t]])
                  for _, t in cand[:20]]
        if mode == "moc":                       # enter at THIS close
            for t, lim, tgt, adv in orders:
                if t in pos or len(pos) >= maxpos:
                    continue
                cl = Cv[i, col[t]]
                if np.isfinite(cl):
                    buy(t, cl, tgt, adv, "auction")
        else:
            pend = orders

        # ---- 4. financing on any margin debit + mark ----------------------
        if cash < 0:
            cash += cash * fin[i]               # cash negative -> pays interest
        mv = sum(pp["sh"] * Cv[i, col[tt]] for tt, pp in pos.items()
                 if np.isfinite(Cv[i, col[tt]]))
        eq.append((cash + mv, mv / (cash + mv) if cash + mv > 0 else 0.0))

    eq, expo = (pd.Series([e[k] for e in eq], index=dates[i0:])
                for k in (0, 1))
    tr = pd.DataFrame(trades, columns=["tk", "entry", "exit", "reason", "bars"])
    tr["ret"] = tr["exit"] / tr["entry"] - 1
    return eq, tr, expo


if __name__ == "__main__":
    qqq = load_etf("QQQ")["Close"]
    qqq_r = qqq.pct_change().loc[TRADE_START:"2026-06-12"].dropna()
    print(f"PIT window {TRADE_START.date()} -> {dates[-1].date()}   "
          f"(setup {time.time()-t0:.0f}s)")
    print(fmt(stats(qqq_r, rf, "QQQ benchmark")))
    print()
    rows, expos = {}, {}
    for mode in ["limit", "moc", "moo", "nextclose"]:
        for label, kw in [
            ("rsi15 pf20 cash-capped costed", dict(rsi_max=15, pos_frac=0.20)),
            ("rsi15 pf20 cash-capped FREE", dict(rsi_max=15, pos_frac=0.20,
                                                 costs=False)),
            ("no-rsi pf20 cash-capped costed", dict(rsi_max=101, pos_frac=0.20)),
        ]:
            eq, tr, expo = run(mode, **kw)
            r = eq.pct_change().dropna()
            st = stats(r, rf, f"{mode:9s} {label}")
            n = len(tr)
            wr = (tr["ret"] > 0).mean() if n else np.nan
            print(fmt(st) + f"  win {wr*100:4.0f}%  n={n:4d} "
                  f"avg {tr['ret'].mean()*100:+.2f}%  expo {expo.mean()*100:.0f}%")
            rows[(mode, label)] = r
            expos[(mode, label)] = expo
        print()
    # persist the headline configs for the ensemble stage
    keep = {m: rows[(m, "rsi15 pf20 cash-capped costed")]
            for m in ["limit", "moc", "moo", "nextclose"]}
    pd.DataFrame(keep).to_parquet(os.path.join(OUT, "sleeveA_mr_modes.parquet"))
    pd.DataFrame({m: expos[(m, "rsi15 pf20 cash-capped costed")]
                  for m in ["limit", "moc", "moo", "nextclose"]}) \
        .to_parquet(os.path.join(OUT, "sleeveA_mr_expo.parquet"))
    print(f"saved -> out/sleeveA_mr_modes.parquet  t={time.time()-t0:.0f}s")
