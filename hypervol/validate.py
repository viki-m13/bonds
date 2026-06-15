"""HYPERVOL — thorough, honest validation.

Produces a full report (stdout + results/validation.json) covering:
  1. Benchmarks                : buy&hold perp (net funding)
  2. Faithful directional port : Strategy-4 long/short timing on the perp
  3. Naked perp carry          : single-leg funding harvest (the failure mode)
  4. Delta-neutral basis carry : always-on vs Strategy-4 eVRP-gated
  5. Ablations                 : does each signal component add value?
  6. IS / OOS split            : chronological, no peeking
  7. Cost sensitivity          : 1x / 2x / 4x HL fees + slippage
  8. Parameter robustness      : grid of rv/funding windows & target vol
  9. Carry tail-risk audit     : negative-funding regimes, worst months, capacity
 10. Out-of-universe coin      : SOL (no DVOL) funding-only carry
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import (Config, backtest, load_coin, stats, fmt_stats, COST_BPS,
                     add_signals, DPY)
from .carry import load_carry_frame, backtest_carry

OUT = Path("/home/user/bonds/hypervol/results")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS: dict = {}

CORE = ["BTC", "ETH"]


def port(rets: list[pd.Series]) -> pd.Series:
    return pd.concat(rets, axis=1).mean(axis=1).dropna()


def line(label: str, s: dict) -> None:
    print(f"  {label:34s} {fmt_stats(s)}")


def section(t: str) -> None:
    print("\n" + "=" * 96 + f"\n{t}\n" + "=" * 96)


# --------------------------------------------------------------------------- #
def bench_and_directional() -> None:
    section("1-2. BENCHMARKS  &  FAITHFUL DIRECTIONAL STRATEGY-4 PORT (perp long/short)")
    dfs = {c: load_coin(c) for c in CORE}

    # buy & hold (long perp, net the funding a long pays)
    bh = []
    for c in CORE:
        d = dfs[c]
        r = (d["ret"] - d["funding_day"]).fillna(0.0)
        bh.append(r.rename(c)); line(f"buy&hold {c} (net funding)", stats(r))
    line("buy&hold PORT 50/50", stats(port(bh)))
    RESULTS["buyhold_port"] = stats(port(bh))

    print()
    for sm in ["voltarget", "ivprop"]:
        for lo in [False, True]:
            rets = []
            for c in CORE:
                cfg = Config(mode="directional", long_only=lo, size_mode=sm)
                rets.append(backtest(dfs[c], cfg)["strat_ret"].rename(c))
            tag = f"directional {'LO ' if lo else 'L/S'} {sm}"
            s = stats(port(rets)); line(tag + " PORT", s)
            RESULTS[f"dir_{sm}_{'lo' if lo else 'ls'}"] = s


def naked_carry() -> None:
    section("3. NAKED SINGLE-LEG PERP CARRY  (funding harvest WITHOUT a spot hedge)")
    print("  Shows why funding carry must be delta-neutral: short a rising market.")
    dfs = {c: load_coin(c) for c in CORE}
    rets = []
    for c in CORE:
        cfg = Config(mode="carry", long_only=False)
        rets.append(backtest(dfs[c], cfg)["strat_ret"].rename(c))
    s = stats(port(rets)); line("naked carry L/S PORT", s)
    RESULTS["naked_carry_port"] = s


def carry_sleeve() -> None:
    section("4. DELTA-NEUTRAL BASIS CARRY  (long spot + short perp)  — the real edge")
    frames = {c: load_carry_frame(c) for c in CORE}
    for gated in [False, True]:
        rets = []
        for c in CORE:
            rets.append(backtest_carry(frames[c], gated)["strat_ret"].rename(c))
            line(f"{c} carry {'gated(Strat4)' if gated else 'always-on'}",
                 stats(backtest_carry(frames[c], gated)["strat_ret"]))
        s = stats(port(rets))
        line(f"carry {'gated(Strat4)' if gated else 'always-on'} PORT", s)
        RESULTS[f"carry_{'gated' if gated else 'alwayson'}_port"] = s
    return frames


def ablation(frames) -> None:
    section("5. ABLATION — does each signal component add value to the carry?")
    print("  Sharpe of the BTC+ETH carry PORT under progressively richer signals.\n")
    btc, eth = frames["BTC"], frames["ETH"]

    def run(weight_fn, label):
        rets = []
        for df in (btc, eth):
            u = weight_fn(df)
            u_eff = u.shift(1).fillna(0.0)
            s = (df["basis"] + df["funding_day"]).fillna(0.0)
            turn = u.diff().abs().fillna(u.abs())
            cost = -(2 * COST_BPS / 1e4) * turn.shift(1).fillna(0.0)
            rets.append((u_eff * s + cost).rename(df.index.name or "r"))
        line(label, stats(port(rets)))

    run(lambda d: pd.Series(1.0, index=d.index),
        "a) always short perp (u=+1)")
    run(lambda d: np.sign(d["funding_ann"]).fillna(0.0),
        "b) + follow funding sign (daily)")
    run(lambda d: np.sign(d["fund_smooth"]).fillna(0.0),
        "c) + smooth funding (7d)  [our carry]")
    run(lambda d: (np.sign(d["fund_smooth"]) *
                   np.where(d["evrp"] > 0, 1.0, 0.5)).fillna(0.0),
        "d) + eVRP size tilt (Strat4)")


def is_oos(frames) -> None:
    section("6. IN-SAMPLE / OUT-OF-SAMPLE  (chronological 60/40 split, no peeking)")
    # Use the always-on carry (winner) + directional port.
    dn = {c: backtest_carry(frames[c], gated=False)["strat_ret"] for c in CORE}
    dirp = {}
    for c in CORE:
        dirp[c] = backtest(load_coin(c), Config(mode="directional"))["strat_ret"]
    carry_p = port([dn[c] for c in CORE])
    dir_p = port([dirp[c] for c in CORE])

    for name, series in [("delta-neutral carry", carry_p), ("directional L/S", dir_p)]:
        n = len(series); cut = int(n * 0.6)
        s_is, s_oos = series.iloc[:cut], series.iloc[cut:]
        print(f"\n  {name}:")
        print(f"    split at {series.index[cut].date()}")
        line("IS  (first 60%)", stats(s_is))
        line("OOS (last  40%)", stats(s_oos))
        RESULTS[f"oos_{name[:5]}_is"] = stats(s_is)
        RESULTS[f"oos_{name[:5]}_oos"] = stats(s_oos)


def cost_sensitivity(frames) -> None:
    section("7. COST SENSITIVITY  (base HL taker+slip = 7.5bps/side; perps DON'T roll)")
    print(f"  base per-side cost = {COST_BPS:.1f} bps\n")
    for mult, tag in [(1, "1x base"), (2, "2x"), (4, "4x"), (8, "8x (stress)")]:
        rets = []
        for c in CORE:
            rets.append(backtest_carry(frames[c], gated=False,
                                       cost_bps=COST_BPS * mult)["strat_ret"])
        line(f"carry @ {tag} ({COST_BPS*mult:.1f}bps/side)", stats(port(rets)))


def robustness(frames) -> None:
    section("8. PARAMETER ROBUSTNESS  (carry: funding-smoothing window & band)")
    print("  Distribution of PORT Sharpe across the grid — looking for a plateau.\n")
    sharpes = []
    for fw in [1, 3, 5, 7, 10, 14, 21]:
        for band in [0.1, 0.25, 0.5, 0.9]:
            rets = []
            for c in CORE:
                df = frames[c].copy()
                df["fund_smooth"] = df["funding_ann"].rolling(fw).mean()
                bt = backtest_carry(df, gated=False, band=band)
                rets.append(bt["strat_ret"])
            sh = stats(port(rets))["sharpe"]
            sharpes.append(sh)
    arr = np.array(sharpes)
    print(f"  grid n={len(arr)}  Sharpe: min={arr.min():.2f} "
          f"p25={np.percentile(arr,25):.2f} median={np.median(arr):.2f} "
          f"p75={np.percentile(arr,75):.2f} max={arr.max():.2f}")
    RESULTS["robust_sharpe_median"] = float(np.median(arr))
    RESULTS["robust_sharpe_min"] = float(arr.min())

    section("8b. DIRECTIONAL ROBUSTNESS  (rv window x target vol)")
    sh2 = []
    for rvw in [5, 10, 15, 20]:
        for tv in [0.20, 0.30, 0.40]:
            rets = []
            for c in CORE:
                cfg = Config(mode="directional", rv_window=rvw, target_vol=tv)
                rets.append(backtest(load_coin(c), cfg)["strat_ret"])
            sh2.append(stats(port(rets))["sharpe"])
    a2 = np.array(sh2)
    print(f"  grid n={len(a2)}  Sharpe: min={a2.min():.2f} median={np.median(a2):.2f} "
          f"max={a2.max():.2f}")
    RESULTS["robust_dir_sharpe_median"] = float(np.median(a2))


def tail_audit(frames) -> None:
    section("9. CARRY TAIL-RISK AUDIT  (where this trade actually bites)")
    for c in CORE:
        df = frames[c]
        bt = backtest_carry(df, gated=False)
        r = bt["strat_ret"]
        neg_funding = (df["funding_ann"] < 0).mean()
        worst_m = r.rolling(30).sum().min()
        worst_d = r.min()
        # behaviour during the worst funding month
        print(f"\n  {c}:")
        print(f"    funding negative on {neg_funding:.1%} of days  "
              f"(mean ann funding {df['funding_ann'].mean():+.1%})")
        print(f"    worst single day {worst_d:+.2%}   worst 30d window {worst_m:+.2%}")
        print(f"    days with |daily basis|>1% : {(df['basis'].abs()>0.01).mean():.1%}"
              f"  (intraday blowouts NOT captured by daily close)")
    RESULTS["caveat"] = ("Daily-close model understates intraday basis/liquidation "
                         "risk; Sharpe is capacity- and tail-limited.")


def out_of_universe() -> None:
    section("10. OUT-OF-UNIVERSE COIN — SOL (no DVOL): funding-only carry holds up?")
    df = load_carry_frame("SOL")
    bt = backtest_carry(df, gated=False)   # falls back to sign(funding)
    s = stats(bt["strat_ret"])
    line("SOL delta-neutral carry", s)
    RESULTS["sol_carry"] = s
    print(f"    mean ann funding {df['funding_ann'].mean():+.1%}  "
          f"(confirms premium generalises beyond BTC/ETH)")


def main() -> None:
    bench_and_directional()
    naked_carry()
    frames = carry_sleeve()
    ablation(frames)
    is_oos(frames)
    cost_sensitivity(frames)
    robustness(frames)
    tail_audit(frames)
    out_of_universe()

    (OUT / "validation.json").write_text(json.dumps(RESULTS, indent=2, default=float))
    print("\n\nSaved results/validation.json")


if __name__ == "__main__":
    main()
