"""Configuration for the winrate30 tool."""
from pathlib import Path

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "data_cache"
REPORTS_DIR = ROOT / "reports"
RULES_FILE = ROOT / "selected_rules.json"

DATA_START = "2000-01-01"

# "Positive 30 calendar days later" ~= 21 trading days.
HORIZON = 21

# Minimum price for a signal (avoids penny-stock artifacts in old data)
MIN_PRICE = 5.0

# Walk-forward validation: train on everything before the test year,
# test on the test year. First test year chosen so the first training
# window is ~15 years.
FIRST_TEST_YEAR = 2016

# Rule screening / selection
MIN_RAW_SIGNALS = 250        # raw (overlapping) signal count to keep a rule
SCREEN_TOP = 150             # rules passed from raw screen to dedup evaluation
MIN_DEDUP_SIGNALS = 30       # non-overlapping training signals required
# Hit-rate tiers searched in order; no low-rate fallback tier on purpose:
# the ensemble may end up smaller than N_SELECT rather than admit a rule
# with a mediocre training hit rate.
RATE_TIERS = (0.90, 0.85, 0.80)
OVERLAP_MAX = 0.6            # max signal overlap between ensemble rules
N_SELECT = 2                 # max rules in the final ensemble (union)
