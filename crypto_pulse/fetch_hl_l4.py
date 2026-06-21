"""Historical per-account trade-tape puller from the Hyperliquid S3 archive.

The node_trades archive holds every fill with both counterparty addresses AND each
side's starting position (start_pos) and order id — i.e. L4-grade attribution, with
HISTORY (unlike the forward-only WS recorder). Default layout (requester-pays):

    s3://hl-mainnet-node-data/node_trades/hourly/{YYYYMMDD}/{H}      (lz4 JSON lines)

Each record (per docs): {coin, side('B'/'A'), time, px, sz, hash, trade_dir_override,
side_info:[{user,start_pos,oid,...}(buyer), {...}(seller)]}. We flatten to the SAME
tidy schema as record_trades_l4.py so flow_l4.py is source-agnostic:
    coin, time(ms), side, px, sz, tid?, buyer, seller, buyer_start_pos, seller_start_pos

Requires AWS credentials (requester-pays => you pay egress). Set them in the env
(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) or via an attached role, then:

    python fetch_hl_l4.py --start 2025-01-01 --end 2025-03-31 \
        --coins BTC,ETH,SOL,DOGE,AVAX,XRP,LINK  (-> data/hl_trades_l4/hist/*.parquet)
"""
import argparse
import datetime as dt
import io
import json
import os

import pandas as pd

BUCKET = "hl-mainnet-node-data"
PREFIX = "node_trades/hourly"
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "hl_trades_l4", "hist")


def _iso_to_ms(t):
    if isinstance(t, (int, float)):
        return int(t)
    try:
        return int(pd.Timestamp(t).value // 1_000_000)
    except Exception:
        return None


def _flatten(rec):
    si = rec.get("side_info") or [{}, {}]
    b, s = (si + [{}, {}])[:2]
    return dict(
        coin=rec.get("coin"), time=_iso_to_ms(rec.get("time")), side=rec.get("side"),
        px=float(rec["px"]), sz=float(rec["sz"]),
        buyer=b.get("user"), seller=s.get("user"),
        buyer_start_pos=pd.to_numeric(b.get("start_pos"), errors="coerce"),
        seller_start_pos=pd.to_numeric(s.get("start_pos"), errors="coerce"),
        hash=rec.get("hash"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    ap.add_argument("--coins", default="BTC,ETH,SOL,DOGE,AVAX,XRP,LINK")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--prefix", default=PREFIX)
    args = ap.parse_args()
    coins = set(args.coins.split(","))

    try:
        import boto3
        import lz4.frame as lz4f
    except ImportError as e:
        raise SystemExit(f"need boto3 + lz4: pip install boto3 lz4  ({e})")
    if not (os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")):
        raise SystemExit("no AWS credentials in env — this archive is requester-pays. "
                         "Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (you pay egress).")
    s3 = boto3.client("s3")
    os.makedirs(OUTDIR, exist_ok=True)

    d0 = dt.date.fromisoformat(args.start); d1 = dt.date.fromisoformat(args.end)
    day = d0
    while day <= d1:
        ds = day.strftime("%Y%m%d")
        rows = []
        for h in range(24):
            key = f"{args.prefix}/{ds}/{h}"
            try:
                obj = s3.get_object(Bucket=args.bucket, Key=key, RequestPayer="requester")
                raw = obj["Body"].read()
            except Exception:
                continue
            try:
                raw = lz4f.decompress(raw)
            except Exception:
                pass  # some shards may be plain JSONL
            for line in io.BytesIO(raw).read().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("coin") in coins:
                    rows.append(_flatten(rec))
        if rows:
            out = os.path.join(OUTDIR, f"trades_{ds}.parquet")
            pd.DataFrame(rows).to_parquet(out, index=False)
            print(f"[{ds}] {len(rows)} trades -> {os.path.basename(out)}", flush=True)
        else:
            print(f"[{ds}] no data (check bucket/prefix/coins)", flush=True)
        day += dt.timedelta(days=1)


if __name__ == "__main__":
    main()
