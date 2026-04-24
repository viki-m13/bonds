"""Expand the crypto universe with more coins, emphasising dead/collapsed
ones to stress-test survivorship handling.

Adds ~30 coins beyond the original 20, focusing on:
  - Majors we're missing: BNB, XMR, DASH, EOS, NEO, IOTA
  - Survivor alts: ETC, ZEC, QTUM, BAT, ZRX, OMG, HBAR, VET, WAVES, THETA, FIL, GRT
  - Dead/collapsed / abandoned / depegged:
      XEM (NEM — largely dead), STEEM (near zero), BTG (Bitcoin Gold — collapsed),
      BCN (Bytecoin — delisted), SC (Siacoin — abandoned), DCR (Decred — fading)
"""
from pathlib import Path
import sys
import time
import yfinance as yf
import pandas as pd

OUT = Path("/home/user/bonds/data/crypto")
OUT.mkdir(parents=True, exist_ok=True)

# (yf_ticker, local_symbol, notes)
NEW_COINS = [
    # Top majors we didn't have
    ("BNB-USD",  "BNB",   "Binance Coin — alive, top-3"),
    ("XMR-USD",  "XMR",   "Monero — alive, privacy"),
    ("DASH-USD", "DASH",  "Dash — alive but fading"),
    ("EOS-USD",  "EOS",   "EOS — alive but largely abandoned"),
    ("NEO-USD",  "NEO",   "NEO — alive but weak"),
    ("IOTA-USD", "IOTA",  "IOTA — alive"),
    ("ETC-USD",  "ETC",   "Ethereum Classic — alive"),
    ("ZEC-USD",  "ZEC",   "Zcash — alive"),
    ("HBAR-USD", "HBAR",  "Hedera — alive"),
    ("VET-USD",  "VET",   "VeChain — alive"),
    ("FIL-USD",  "FIL",   "Filecoin — alive but -95% from ATH"),
    ("THETA-USD","THETA", "Theta — alive"),
    ("XTZ-USD",  "XTZ",   "Tezos — alive"),
    ("QTUM-USD", "QTUM",  "Qtum — alive but weak"),
    ("MKR-USD",  "MKR",   "Maker — alive"),
    ("GRT-USD",  "GRT",   "The Graph — alive"),
    # Coins that have largely FAILED or been delisted/abandoned:
    ("XEM-USD",  "XEM",   "NEM — largely dead, -99% from ATH"),
    ("BTG-USD",  "BTG",   "Bitcoin Gold — collapsed, attacked"),
    ("STEEM-USD","STEEM", "Steem — near zero"),
    ("SC-USD",   "SC",    "Siacoin — abandoned"),
    ("DCR-USD",  "DCR",   "Decred — -95%"),
    ("BAT-USD",  "BAT",   "Basic Attention — delisted some venues"),
    ("ZRX-USD",  "ZRX",   "0x — weak, delisted partially"),
    ("OMG-USD",  "OMG",   "OMG Network — delisted, near dead"),
    ("LRC-USD",  "LRC",   "Loopring — faded"),
    ("KNC-USD",  "KNC",   "Kyber — faded"),
    ("ICX-USD",  "ICX",   "Icon — faded"),
    ("WAVES-USD","WAVES", "Waves — USDN collapse 2022, -99%"),
    ("CELR-USD", "CELR",  "Celer — delisted from some"),
    ("ANT-USD",  "ANT",   "Aragon — dissolved"),
]


def download(yf_t, local, start="2014-09-17"):
    fp = OUT / f"{local}_USD.csv"
    if fp.exists():
        return "skipped"
    try:
        df = yf.download(yf_t, start=start, progress=False, auto_adjust=False)
        if df.empty or len(df) < 60:
            return f"too short ({len(df)})"
        # Flatten multi-level columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].reset_index()
        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df.to_csv(fp, index=False)
        return f"ok n={len(df)} {df['Date'].min().date()}→{df['Date'].max().date()}"
    except Exception as e:
        return f"fail: {e}"


def main():
    for yf_t, local, note in NEW_COINS:
        r = download(yf_t, local)
        print(f"  {local:8s} {yf_t:12s} {r}  [{note}]")
        time.sleep(0.5)  # be polite to Yahoo


if __name__ == "__main__":
    main()
