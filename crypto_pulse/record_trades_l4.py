"""Lean per-account trade-tape recorder (L4-grade attribution, FREE via WS).

The public Hyperliquid `trades` channel carries every fill with BOTH counterparty
addresses: {coin, side, px, sz, time, hash, tid, users:[buyer, seller]}. side 'B'
means the aggressor lifted the ask (taker buy); 'A' means the aggressor hit the bid
(taker sell). Recording this forward gives us per-account order flow — who is
aggressing, net flow concentration, whale participation — which is a genuinely
different information axis from price-based factors (STRATA) and the vol channel (VOL).

This writes a compact Parquet shard per UTC hour (far smaller than dumping raw L2),
so it is cheap to run continuously on a small always-on box for weeks. Then point
flow_l4.py at the shard directory to build features and backtest.

History cannot be backfilled from the WS; for a historical backtest pull the
node_trades archive from S3 instead (see fetch_hl_l4.py) — same schema, plus start_pos.

Run from crypto_pulse/:
    python record_trades_l4.py --secs 3600 --coins BTC,ETH,SOL,DOGE,AVAX,XRP,LINK
"""
import argparse
import json
import os
import time

import pandas as pd
import websocket  # websocket-client

WS = "wss://api.hyperliquid.xyz/ws"
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "hl_trades_l4")


def hour_key(ms):
    return time.strftime("%Y%m%d_%H", time.gmtime(ms / 1000))


def flush(rows, hk):
    if not rows:
        return
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"trades_{hk}.parquet")
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df = df.drop_duplicates(subset=["tid"])      # WS resends trades; tid is unique
    df.to_parquet(path, index=False)
    print(f"[{hk}] {len(df)} trades -> {os.path.basename(path)}", flush=True)


def subscribe(ws, coins):
    for c in coins:
        ws.send(json.dumps({"method": "subscribe",
                            "subscription": {"type": "trades", "coin": c}}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=int, default=3600)
    ap.add_argument("--coins", default="BTC,ETH,SOL,DOGE,AVAX,XRP,LINK")
    args = ap.parse_args()
    coins = args.coins.split(",")

    ws = websocket.create_connection(WS, timeout=20)
    subscribe(ws, coins)
    rows, cur_hk = [], None
    t0 = time.time()
    n = 0
    while time.time() - t0 < args.secs:
        try:
            m = json.loads(ws.recv())
        except Exception as e:
            try:
                ws = websocket.create_connection(WS, timeout=20); subscribe(ws, coins); continue
            except Exception:
                print("reconnect failed:", e); break
        if m.get("channel") != "trades":
            continue
        for d in m["data"]:
            u = d.get("users") or [None, None]
            hk = hour_key(d["time"])
            if cur_hk is None:
                cur_hk = hk
            if hk != cur_hk:
                flush(rows, cur_hk); rows, cur_hk = [], hk
            rows.append(dict(
                coin=d["coin"], time=int(d["time"]), side=d["side"],
                px=float(d["px"]), sz=float(d["sz"]), tid=int(d["tid"]),
                buyer=u[0], seller=u[1], hash=d.get("hash")))
            n += 1
    flush(rows, cur_hk)
    print(f"done: {n} trades recorded over {int(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    main()
