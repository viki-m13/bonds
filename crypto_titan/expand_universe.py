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
    # (First wave — already in repo: BNB, XMR, DASH, EOS, NEO, IOTA, ETC, ZEC,
    #  HBAR, VET, FIL, THETA, XTZ, QTUM, MKR, GRT, XEM, BTG, STEEM, SC, DCR,
    #  BAT, ZRX, OMG, LRC, KNC, ICX, WAVES, CELR, ANT)

    # More majors / surviving alts
    ("SHIB-USD", "SHIB",  "Shiba — meme, alive but -85% ATH"),
    ("NEAR-USD", "NEAR",  "Near Protocol — L1, alive"),
    ("APT-USD",  "APT",   "Aptos — L1, alive"),
    ("ARB-USD",  "ARB",   "Arbitrum — L2, alive"),
    ("OP-USD",   "OP",    "Optimism — L2, alive"),
    ("SUI-USD",  "SUI",   "Sui — L1, alive"),
    ("INJ-USD",  "INJ",   "Injective — alive"),
    ("TIA-USD",  "TIA",   "Celestia — alive"),
    ("PEPE-USD", "PEPE",  "Pepe — meme, alive"),
    ("STX-USD",  "STX",   "Stacks — alive"),
    ("AR-USD",   "AR",    "Arweave — alive, -90% ATH"),
    ("RNDR-USD", "RNDR",  "Render — alive"),
    ("SEI-USD",  "SEI",   "Sei — alive"),
    ("KAS-USD",  "KAS",   "Kaspa — alive"),
    ("PYTH-USD", "PYTH",  "Pyth Network — alive"),
    ("JUP-USD",  "JUP",   "Jupiter — alive"),

    # DeFi tokens (many -90%+ from ATH)
    ("AAVE-USD", "AAVE",  "Aave — alive but -85% ATH"),
    ("CRV-USD",  "CRV",   "Curve — alive, -95% ATH"),
    ("COMP-USD", "COMP",  "Compound — alive, -95% ATH"),
    ("YFI-USD",  "YFI",   "Yearn — alive, -97% ATH"),
    ("SNX-USD",  "SNX",   "Synthetix — -95% ATH"),
    ("1INCH-USD","1INCH", "1inch — -95% ATH"),
    ("BAL-USD",  "BAL",   "Balancer — -95% ATH"),
    ("SUSHI-USD","SUSHI", "SushiSwap — -99% ATH"),
    ("DYDX-USD", "DYDX",  "dYdX — -95% ATH"),

    # Gaming / metaverse (all -90%+)
    ("MANA-USD", "MANA",  "Decentraland — -95% ATH"),
    ("SAND-USD", "SAND",  "Sandbox — -95% ATH"),
    ("AXS-USD",  "AXS",   "Axie — collapsed after Ronin hack, -98%"),
    ("GALA-USD", "GALA",  "Gala — -99% ATH"),
    ("ENJ-USD",  "ENJ",   "Enjin — -95% ATH"),
    ("ILV-USD",  "ILV",   "Illuvium — -98% ATH"),

    # Old ICOs — many failed / barely alive
    ("BAT-USD",  "BAT",   "ALREADY HAVE"),  # skip
    ("BNT-USD",  "BNT",   "Bancor — -99% ATH"),
    ("POWR-USD", "POWR",  "Power Ledger — faded"),
    ("STORJ-USD","STORJ", "Storj — faded, -95% ATH"),
    ("REP-USD",  "REP",   "Augur — DEAD, near zero"),
    ("GNT-USD",  "GNT",   "Golem — symbol changed (delisted)"),
    ("CDT-USD",  "CDT",   "Blox — dead"),
    ("ZIL-USD",  "ZIL",   "Zilliqa — faded"),
    ("ONT-USD",  "ONT",   "Ontology — faded"),
    ("WTC-USD",  "WTC",   "Waltonchain — DEAD"),
    ("LSK-USD",  "LSK",   "Lisk — mostly abandoned"),
    ("ARK-USD",  "ARK",   "Ark — faded"),
    ("STRAT-USD","STRAT", "Stratis — faded"),

    # Failed privacy / payment coins
    ("PIVX-USD", "PIVX",  "PIVX — fading"),
    ("DGB-USD",  "DGB",   "DigiByte — faded, -95%"),
    ("XVG-USD",  "XVG",   "Verge — near dead"),
    ("NANO-USD", "NANO",  "Nano (was XRB) — faded"),
    ("VTC-USD",  "VTC",   "Vertcoin — near dead"),
    ("BCD-USD",  "BCD",   "Bitcoin Diamond — DEAD"),
    ("BCN-USD",  "BCN",   "Bytecoin — delisted many venues"),

    # Rugged / catastrophic failures
    ("HEX-USD",  "HEX",   "HEX — ponzi-like, -95%"),
    ("KIN-USD",  "KIN",   "Kin — essentially dead"),
    ("REN-USD",  "REN",   "REN — protocol shutdown 2023"),
    ("MIR-USD",  "MIR",   "Mirror — Terra ecosystem collapse"),
    ("ANC-USD",  "ANC",   "Anchor — Terra ecosystem, -99.99%"),
    ("RUNE-USD", "RUNE",  "THORChain — hack, -95% recovery"),

    # Chain tokens that faded
    ("ONE-USD",  "ONE",   "Harmony — -99% after Horizon Bridge hack"),
    ("FTM-USD",  "FTM",   "Fantom — rebranded to S, -90%"),
    ("KSM-USD",  "KSM",   "Kusama — faded"),
    ("KAVA-USD", "KAVA",  "Kava — faded"),
    ("IOST-USD", "IOST",  "IOST — faded"),
    ("CHZ-USD",  "CHZ",   "Chiliz — -90% ATH"),
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
