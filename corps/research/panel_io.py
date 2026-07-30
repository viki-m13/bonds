"""Load the OSBAP corporate panel from the repo's committed year-partitioned
files (corps/data/panel/*.parquet), or the single local file if present."""
from pathlib import Path
import glob
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SINGLE = ROOT / "data" / "panel_osbap_full.parquet"
PARTS = ROOT / "data" / "panel"


def load_full(columns=None) -> pd.DataFrame:
    if SINGLE.exists():
        return pd.read_parquet(SINGLE, columns=columns)
    files = sorted(glob.glob(str(PARTS / "osbap_*.parquet")))
    if not files:
        raise FileNotFoundError(
            "no panel found — run corps/scripts/build_osbap_panel.py or "
            "unzip the committed corps/data/panel/ files")
    return pd.concat((pd.read_parquet(f, columns=columns) for f in files),
                     ignore_index=True)
