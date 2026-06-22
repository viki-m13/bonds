"""Merge trade-tape Parquet shards into a destination dir, deduping on tid.

Used by the record_flow workflow to fold each job's freshly recorded shards into the
flow-data branch without losing prior trades when hour-buckets overlap at job handoff.

    python merge_shards.py <src_dir> <dst_dir>
"""
import glob
import os
import sys

import pandas as pd


def main(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for src in sorted(glob.glob(os.path.join(src_dir, "*.parquet"))):
        dst = os.path.join(dst_dir, os.path.basename(src))
        new = pd.read_parquet(src)
        if os.path.exists(dst):
            new = pd.concat([pd.read_parquet(dst), new], ignore_index=True)
        if "tid" in new.columns:
            new = new.drop_duplicates(subset=["tid"])
        new.to_parquet(dst, index=False)
        print(f"{os.path.basename(dst)}: {len(new)} trades", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
