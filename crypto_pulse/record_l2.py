"""Hyperliquid L2 order-book + trade recorder (the maker-path data collector).

Subscribes to the HL WebSocket l2Book + trades channels for a set of liquid
coins and writes every message (with a local receive timestamp) to a JSONL file.
This is FORWARD data collection — L2 history cannot be backfilled, so a real
maker validation needs this run for hours/days. Even a short run yields thousands
of book updates + trades, enough to exercise the queue-fill simulator
(maker_sim.py) and measure spread capture vs adverse selection honestly.

Run from crypto_pulse/:
    python record_l2.py --secs 900 --coins BTC,ETH,SOL,DOGE,AVAX
"""
import argparse
import json
import os
import time

import websocket  # websocket-client

WS = "wss://api.hyperliquid.xyz/ws"
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "hl_l2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=int, default=900)
    ap.add_argument("--coins", default="BTC,ETH,SOL,DOGE,AVAX")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    coins = args.coins.split(",")
    os.makedirs(OUTDIR, exist_ok=True)
    path = args.out or os.path.join(
        OUTDIR, f"rec_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")

    ws = websocket.create_connection(WS, timeout=20)
    for c in coins:
        for ch in ("l2Book", "trades"):
            ws.send(json.dumps({"method": "subscribe",
                                "subscription": {"type": ch, "coin": c}}))
    n_l2 = n_tr = 0
    t0 = time.time()
    with open(path, "w") as fh:
        while time.time() - t0 < args.secs:
            try:
                raw = ws.recv()
            except Exception as e:
                fh.flush()
                try:
                    ws = websocket.create_connection(WS, timeout=20)
                    for c in coins:
                        for ch in ("l2Book", "trades"):
                            ws.send(json.dumps({"method": "subscribe",
                                    "subscription": {"type": ch, "coin": c}}))
                    continue
                except Exception:
                    print("reconnect failed:", e); break
            m = json.loads(raw)
            ch = m.get("channel")
            if ch not in ("l2Book", "trades"):
                continue
            rec = {"r": time.time(), "c": ch, "d": m["data"]}
            fh.write(json.dumps(rec) + "\n")
            if ch == "l2Book":
                n_l2 += 1
            else:
                n_tr += len(m["data"]) if isinstance(m["data"], list) else 1
            if (n_l2 + n_tr) % 2000 == 0:
                print(f"  {int(time.time()-t0)}s  l2={n_l2} trades={n_tr}", flush=True)
    ws.close()
    print(f"DONE {path}  l2={n_l2} trades={n_tr} secs={int(time.time()-t0)}")


if __name__ == "__main__":
    main()
