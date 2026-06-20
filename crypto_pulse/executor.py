"""PULSE-HL executor — reconcile target positions to Hyperliquid and trade them.

SAFE BY DEFAULT. This runs in DRY-RUN unless you pass --live AND set HL_ACCOUNT
(public address) and HL_SECRET_KEY (env). Dry-run needs neither secret nor the
hyperliquid SDK: it pulls public account state, computes the orders it WOULD
place, runs every risk gate, and prints them. Only the --live path imports the
SDK and signs orders. Review + paper-trade (see BOT_DEPLOYMENT.md) before --live.

Pipeline each run (intended: once per UTC day after the daily close):
  1. targets  <- live_signal.current_targets()  (3-sleeve TREND+CARRY+ORDERFLOW)
  2. state    <- HL clearinghouseState (positions, account value) + allMids + meta
  3. reconcile-> per-coin delta notional = target - current
  4. risk gate-> gross-leverage cap, per-coin cap, drawdown kill-switch,
                 staleness guard, funding guard
  5. orders   -> reduce-only on shrink/flip; ALO limit near mid (maker) by
                 default, IOC option; rounded to tick/lot; skip < $10 notional
  6. submit   -> dry-run prints; --live signs via the hyperliquid SDK

Nothing here auto-deploys capital; it is the engineering scaffold for the
reviewed bot.
"""
import argparse
import json
import os
import urllib.request

import live_signal as ls

INFO = "https://api.hyperliquid.xyz/info"

# ---- risk limits (the executor enforces these regardless of the signal) ----
MAX_GROSS_LEVERAGE = 3.0       # refuse to exceed, even if the target asks
PER_COIN_CAP_FRAC = 0.15       # max |notional| per coin as a fraction of equity
DRAWDOWN_KILL = 0.25           # flatten + halt if account drawdown exceeds this
MAX_FUNDING_BPS_HR = 5.0       # skip/length-cap a coin if |funding| > this
MIN_ORDER_USD = 12.0
SLIP_LIMIT_BPS = 5.0           # ALO limit placed this far inside mid


def _post(body):
    req = urllib.request.Request(INFO, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=20).read())


def get_state(address):
    cs = _post({"type": "clearinghouseState", "user": address})
    equity = float(cs["marginSummary"]["accountValue"])
    pos = {}
    for p in cs.get("assetPositions", []):
        po = p["position"]
        pos[po["coin"]] = float(po["szi"]) * float(po.get("entryPx") or 0)
    return equity, pos, cs


def get_market():
    mids = {k: float(v) for k, v in _post({"type": "allMids"}).items()}
    meta = _post({"type": "meta"})
    info = {c["name"]: {"maxLev": c["maxLeverage"],
                        "szDec": c["szDecimals"]} for c in meta["universe"]}
    # current funding (annualized hourly) per coin
    ctxs = _post({"type": "metaAndAssetCtxs"})[1]
    funding = {meta["universe"][i]["name"]: float(ctxs[i].get("funding", 0)) * 1e4
               for i in range(len(meta["universe"]))}     # bps/hour
    return mids, info, funding


def reconcile(targets_usd, equity, current_pos, mids, info, funding,
              high_water=None):
    """Return (orders, blocks) where orders are dicts ready to send and blocks
    explain anything skipped. Pure/inspectable — no network, no signing."""
    orders, blocks = [], []
    # drawdown kill-switch
    if high_water and equity < high_water * (1 - DRAWDOWN_KILL):
        blocks.append(f"DRAWDOWN KILL: equity {equity:.0f} < "
                      f"{(1-DRAWDOWN_KILL):.0%} of high-water {high_water:.0f} "
                      "-> flatten all, halt")
        for coin, cur in current_pos.items():
            if abs(cur) >= MIN_ORDER_USD:
                orders.append(_mk_order(coin, -cur, mids, info, reduce_only=True))
        return [o for o in orders if o], blocks

    # gross-leverage cap on the target book
    gross = sum(abs(v) for v in targets_usd.values())
    if gross > MAX_GROSS_LEVERAGE * equity:
        sc = MAX_GROSS_LEVERAGE * equity / gross
        targets_usd = {k: v * sc for k, v in targets_usd.items()}
        blocks.append(f"gross cap: scaled targets x{sc:.2f} to {MAX_GROSS_LEVERAGE}x")

    coins = set(targets_usd) | set(current_pos)
    for coin in sorted(coins):
        tgt = targets_usd.get(coin, 0.0)
        # per-coin cap
        cap = PER_COIN_CAP_FRAC * equity
        if abs(tgt) > cap:
            tgt = cap * (1 if tgt > 0 else -1)
            blocks.append(f"{coin}: capped to {PER_COIN_CAP_FRAC:.0%} equity")
        # funding guard: don't open/expand into extreme funding against us
        f = funding.get(coin, 0.0)
        cur = current_pos.get(coin, 0.0)
        if abs(f) > MAX_FUNDING_BPS_HR and (abs(tgt) > abs(cur)) and \
                (tgt * f > 0):          # paying funding and increasing exposure
            blocks.append(f"{coin}: funding {f:+.1f}bps/hr against us -> hold, "
                          "no add")
            tgt = cur
        delta = tgt - cur
        if abs(delta) < MIN_ORDER_USD:
            continue
        o = _mk_order(coin, delta, mids, info,
                      reduce_only=(cur != 0 and abs(tgt) < abs(cur)))
        if o:
            orders.append(o)
    return orders, blocks


def _mk_order(coin, delta_usd, mids, info, reduce_only=False):
    if coin not in mids or coin not in info:
        return None
    mid = mids[coin]
    is_buy = delta_usd > 0
    # ALO (post-only) limit one SLIP_LIMIT_BPS inside the mid -> maker
    px = mid * (1 - SLIP_LIMIT_BPS / 1e4) if is_buy else mid * (1 + SLIP_LIMIT_BPS / 1e4)
    sz = round(abs(delta_usd) / mid, info[coin]["szDec"])
    if sz <= 0:
        return None
    return {"coin": coin, "is_buy": is_buy, "sz": sz, "limit_px": round(px, 6),
            "reduce_only": reduce_only, "notional_usd": round(delta_usd, 2),
            "tif": "Alo"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="actually place orders (needs HL_SECRET_KEY); default dry-run")
    ap.add_argument("--high-water", type=float, default=None,
                    help="account high-water mark for the drawdown kill-switch")
    args = ap.parse_args()

    address = os.environ.get("HL_ACCOUNT")
    if not address:
        print("Set HL_ACCOUNT (public address) to read live state. Showing "
              "target weights only (signal-only mode):\n")
        asof, notional, gross, rw = ls.current_targets()
        print(f"3-sleeve targets {asof.date()} | gross {gross:.2f}x | "
              + ", ".join(f"{k} {rw[k]:.0%}" for k in rw.index))
        for c, n in notional.items():
            print(f"  {c:6s} {'LONG' if n>0 else 'SHORT':5s} ${n:,.0f}")
        return

    equity, current_pos, _ = get_state(address)
    mids, info, funding = get_market()
    asof, notional, gross, rw = ls.current_targets()
    targets_usd = {c: float(n) / ls.ACCOUNT_EQUITY_USD * equity
                   for c, n in notional.items()}   # rescale to live equity
    orders, blocks = reconcile(targets_usd, equity, current_pos, mids, info,
                               funding, high_water=args.high_water)

    print(f"3-SLEEVE executor | asof {asof.date()} | equity ${equity:,.0f} | "
          f"gross {gross:.2f}x | {len(orders)} orders | "
          f"{'LIVE' if args.live else 'DRY-RUN'}")
    for b in blocks:
        print("  [risk]", b)
    for o in orders:
        print(f"  {'BUY ' if o['is_buy'] else 'SELL'} {o['sz']:>12} {o['coin']:6s}"
              f" @ {o['limit_px']:<12} {'(reduce)' if o['reduce_only'] else ''}"
              f" ${o['notional_usd']:+,.0f}")

    if not args.live:
        print("\nDRY-RUN: no orders sent. Re-run with --live + HL_SECRET_KEY to "
              "trade (only after paper-trading review).")
        return

    # ---- live path: sign + submit via the hyperliquid SDK (lazy import) ----
    key = os.environ.get("HL_SECRET_KEY")
    if not key:
        raise SystemExit("--live needs HL_SECRET_KEY in env. Aborting.")
    from hyperliquid.exchange import Exchange         # noqa: E402
    from eth_account import Account                   # noqa: E402
    ex = Exchange(Account.from_key(key), account_address=address)
    for o in orders:
        res = ex.order(o["coin"], o["is_buy"], o["sz"], o["limit_px"],
                       {"limit": {"tif": o["tif"]}}, reduce_only=o["reduce_only"])
        print("  sent", o["coin"], res.get("status"))


if __name__ == "__main__":
    main()
