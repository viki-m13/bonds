"""Forward order-flow recorder — clean signed taker volume from the HL trades feed.

Subscribes to the HL WebSocket trades channel for a set of coins and accumulates,
per UTC day per coin, the aggressive BUY vs SELL dollar volume (HL trade side
'B' = aggressor lifted the ask = buy; 'A' = aggressor hit the bid = sell). On each
UTC day rollover it appends a finalized row to data/hl_orderflow/orderflow.csv:

    date, coin, buy_qvol_usd, sell_qvol_usd, n_trades

Order flow = buy - sell; imbalance = (buy - sell)/(buy + sell). This is the clean,
liquid-venue version of the signal three_sleeve.py proxied from OHLC. Run it
continuously on a server for a few weeks, then plug the real series into the
ORDERFLOW sleeve. Run from crypto_pulse/:
    python record_orderflow.py --coins BTC,ETH,SOL,DOGE,AVAX,XRP,LINK,LTC,...
"""
import argparse
import csv
import json
import os
import time
from collections import defaultdict

import websocket  # websocket-client

WS = "wss://api.hyperliquid.xyz/ws"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "hl_orderflow")


def utc_day(ms):
    return time.strftime("%Y-%m-%d", time.gmtime(ms / 1000))


def flush(path, day, agg):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "coin", "buy_qvol_usd", "sell_qvol_usd", "n_trades"])
        for coin, (b, s, n) in sorted(agg.items()):
            w.writerow([day, coin, round(b, 2), round(s, 2), n])
    print(f"[{day}] flushed {len(agg)} coins", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", default="BTC,ETH,SOL,DOGE,AVAX,XRP,LINK,LTC,DOT,"
                    "ATOM,UNI,ETC,BCH,AAVE,NEAR,APT,ARB,INJ,SUI,TIA")
    args = ap.parse_args()
    coins = args.coins.split(",")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "orderflow.csv")

    def connect():
        ws = websocket.create_connection(WS, timeout=20)
        for c in coins:
            ws.send(json.dumps({"method": "subscribe",
                                "subscription": {"type": "trades", "coin": c}}))
        return ws

    ws = connect()
    cur_day = None
    agg = defaultdict(lambda: [0.0, 0.0, 0])     # coin -> [buy_usd, sell_usd, n]
    while True:
        try:
            m = json.loads(ws.recv())
        except Exception as e:
            print("reconnect:", e, flush=True)
            time.sleep(2)
            try:
                ws = connect()
            except Exception:
                time.sleep(10)
            continue
        if m.get("channel") != "trades":
            continue
        for t in m["data"]:
            d = utc_day(t["time"])
            if cur_day is None:
                cur_day = d
            if d != cur_day:                      # UTC rollover -> finalize day
                flush(path, cur_day, agg)
                agg = defaultdict(lambda: [0.0, 0.0, 0])
                cur_day = d
            usd = float(t["px"]) * float(t["sz"])
            rec = agg[t["coin"]]
            if t["side"] == "B":
                rec[0] += usd
            else:
                rec[1] += usd
            rec[2] += 1


if __name__ == "__main__":
    main()
