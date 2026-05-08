"""Breakdown: which slice of the extended universe (small-mid / intl / small-tech)
hurts or helps PURE and COMPOSITE? Also test top-K sensitivity (3/5/7/10).
"""
from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd
from alt.meridian_extended_stocks_test import (
    get_stock_universe_base, panel, topk_sleeve, overlay, metrics,
    DD_FLOOR_PURE, DD_FLOOR_COMP, IS_START, IS_END, OOS_START, ETF_UNIVERSE,
)

ROOT = Path(__file__).resolve().parent.parent
STOCK_EXT = ROOT / "data" / "stocks_extended"

# Re-derive the three category lists from the download script's intent
SMALL_MID_HINTS = {
    "AAON","AIT","ALLE","AMCR","AOS","APH","ATR","AVNT","AVT","AYI","BCO",
    "BDX","BR","BRO","BURL","CASY","CCK","CFR","CGNX","CHE","CIEN","CLH",
    "CMS","CNC","COLM","CR","CSL","CW","DAR","DKL","DLB","DOX","DPZ","EHC",
    "EME","ENS","EXLS","FAF","FCN","FELE","FLO","FMC","FRT","FTI","FTV",
    "GGG","GPC","HALO","HRC","HSY","IDXX","IEX","INGR","ITT","JEF","JKHY",
    "KEX","KMT","KNX","LAD","LAMR","LAZ","LFUS","LHX","LITE","LNC","LPLA",
    "LULU","MAS","MGEE","MORN","MPWR","MSA","MSCI","MTH","MTN","MTRN","NHI",
    "NJR","NLY","NNN","NRG","NUE","OFC","OGE","OHI","OLN","OMI","OSK","OZK",
    "PB","PEG","PEN","PFGC","PINC","PNW","POR","POWI","PSN","PTC","PVH",
    "PWR","QGEN","RBC","REG","REXR","RGA","RHI","RJF","RL","RLI","RNR",
    "RPM","RS","SAFM","SAH","SAIC","SBNY","SBRA","SCHL","SCI","SE","SEDG",
    "SEE","SEM","SF","SFM","SGEN","SHO","SHOO","SIGI","SITE","SJM","SLM",
    "SMG","SNA","SNV","SON","SPGI","SPR","SR","SRE","STE","STLD","STZ",
    "SU","SUI","SUN","SWAV","SWK","SWKS","SYY","TBI","TDG","TDY","TEX",
    "TFII","TFX","THC","THG","THO","TJX","TKR","TMHC","TOL","TPR","TPX",
    "TRGP","TRI","TRMB","TRN","TROW","TRP","TRTN","TRU","TRV","TS","TSCO",
    "TT","TTC","TTEK","TTM","TTMI","TTWO","U","UA","UAA","UBSI","UDR","UFS",
    "UGI","UHS","UI","ULBI","UMBF","UNF","UNM","URBN","URI","VFC","VICI",
    "VLO","VMC","VMI","VNO","VOYA","VRSK","VRSN","VRT","VRTX","VSAT","VST",
    "VTR","VTRS","VVV","WAFD","WAL","WAT","WBA","WBC","WBS","WBD","WBT",
    "WCN","WDC","WDFC","WEC","WELL","WERN","WEX","WGO","WH","WHR","WIRE",
    "WK","WLK","WPC","WRB","WRK","WSBC","WSC","WSFS","WSM","WSO","WST",
    "WTRG","WTS","WU","WWD","WWE","WWW","WY","WYND","WYNN","X","XEL","XPO",
    "XRAY","XYL","Y","YETI","YUM","ZBH","ZION","ZTS","ARW","ATO","AVY","BC",
    "BWA","CACI","CAH","CARR","CCL","CHRW","CINF","CNDT","CNX","COTY","CPB",
    "CRL","DPZ","DRI","EQT","EVR","EXR","ANET","ALGN","BLDR","CDW","CHRW",
    "DECK","DKS","DXCM","EXEL","FDS","FFIV","HEI","HUBB","JBL","KEYS","LECO",
    "LSCC","MASI","MIDD","MKSI","MKTX","MOH","ODFL","ON","PAYC","RGEN","ROL",
    "SAIA","SLAB","SNX","TREX","TYL","ULTA","WAB","WST","ZBRA","POOL","MANH",
    "EXP","NDSN","MEDP","FIVE","MTZ",
}

SMALL_TECH_HINTS = {
    "EPAM","WIX","SPSC","BL","QLYS","PCTY","PAYC","NTNX","RNG","FIVN","ESTC",
    "EVR","CACI","LOPE","GWRE","VEEV","TWLO","OKTA","ZS","DDOG","MDB","SNOW",
    "NET","CRWD","TEAM","DOCU","ZM","HUBS","PLTR","AI","SMCI","IOT","BSY",
    "PATH","S","GTLB","FROG","BILL","DT","ENV","BMI","CGNX","PRGS","QTWO",
    "ALRM","BAND","BRZE","ASGN","CIEN","COHU","CYBR","VRNS","TENB","PFPT",
    "JNPR","SUMO","DOMO","ZUO","EVER","PD","CSGS","AMSC","ANSS","PEGA",
    "TRMB","INTU","ADSK","FLT","GLOB","OTEX","DXC",
}

INTL_HINTS = {
    "TSM","ASML","NVO","SAP","TM","HMC","BTI","DEO","UL","BHP","RIO","TTE",
    "SHEL","BCS","HSBC","MFG","AZN","GSK","NVS","RY","TD","BNS","BP","STM",
    "BABA","BIDU","JD","PDD","NIO","TCEHY","NTES","ICLR","ABEV","VALE","PBR",
    "ITUB","TEVA","SNY","NMR","MUFG","SMFG","RELX","FMS","PHG","VOD","ORAN",
    "E","ENB","TRP","CNQ","SU","CNI","WCN","CP","MFC","BMO","WIT","INFY",
    "IBN","HDB","NTDOY","SONY","NICE","CHKP","TLK","PHI","VIV","ATI","AGCO",
    "LH","ESS",
}


def avail(s):
    return [t for t in sorted(s) if (STOCK_EXT / f"{t}.csv").exists()]


def with_history(univ, min_start=IS_START):
    """Filter to names with ≥min_start data."""
    out = []
    for t in univ:
        p = STOCK_EXT / f"{t}.csv"
        if not p.exists(): continue
        df = pd.read_csv(p, parse_dates=["Date"], usecols=["Date"])
        if df["Date"].min() <= min_start:
            out.append(t)
    return out


def run_pure_universe(univ, k=3, lookback=126, dd_floor=DD_FLOOR_PURE):
    opens, closes = panel(univ, ["BIL"])
    raw, _ = topk_sleeve(univ, opens, closes, k, lookback, "W")
    net = overlay(raw, dd_floor=dd_floor)
    return {
        "n": len(univ),
        "raw": metrics(raw.loc[IS_START:], "RAW"),
        "full": metrics(net.loc[IS_START:], "FULL"),
        "is": metrics(net.loc[IS_START:IS_END], "IS"),
        "oos": metrics(net.loc[OOS_START:], "OOS"),
    }


def fmt(label, m):
    return (f"  {label:32s} n={m['n']:3d}  FULL Sh={m['full']['sharpe']:5.2f} "
            f"CAGR={m['full']['cagr']*100:5.1f}% MDD={m['full']['mdd']*100:6.1f}% | "
            f"IS Sh={m['is']['sharpe']:.2f} OOS Sh={m['oos']['sharpe']:.2f}")


def main():
    base = get_stock_universe_base()
    sm = with_history(avail(SMALL_MID_HINTS))
    st = with_history(avail(SMALL_TECH_HINTS))
    intl = with_history(avail(INTL_HINTS))
    sm = [t for t in sm if t not in base]
    st = [t for t in st if t not in base]
    intl = [t for t in intl if t not in base]
    print(f"BASE 90 + small_mid={len(sm)} small_tech={len(st)} intl={len(intl)}")
    print()

    print("=== PURE k=3 lookback=126d, weekly ===")
    print(fmt("BASE only (90)", run_pure_universe(base)))
    print(fmt("BASE + small_mid", run_pure_universe(base + sm)))
    print(fmt("BASE + small_tech", run_pure_universe(base + st)))
    print(fmt("BASE + intl ADR", run_pure_universe(base + intl)))
    print(fmt("BASE + ALL extended", run_pure_universe(base + sm + st + intl)))

    print("\n=== Top-K sensitivity on EXTENDED (BASE+ALL) ===")
    full = base + sm + st + intl
    for k in [3, 5, 7, 10, 15]:
        print(fmt(f"k={k}", run_pure_universe(full, k=k)))

    print("\n=== Top-K sensitivity on BASE ===")
    for k in [3, 5, 7, 10]:
        print(fmt(f"k={k}", run_pure_universe(base, k=k)))

    print("\n=== Lookback sensitivity on BASE+ALL, k=10 ===")
    for lb in [63, 126, 189, 252]:
        print(fmt(f"lookback={lb}", run_pure_universe(full, k=10, lookback=lb)))


if __name__ == "__main__":
    main()
