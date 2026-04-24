"""Generate factsheet data JSON for the crypto-APEX webpage."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
import numpy as np
import pandas as pd
import util
from util import metrics, regime_slice, load_prices, load_macro, SURVIVORS, DEAD, ALL_COINS, OUT, DPY
import sleeves as SV
from strategy import build_portfolio, FINAL_SLEEVES, TARGET_VOL, DD_FLOOR

OOS_START = "2022-07-01"


def _yearly(net):
    y = net.groupby(net.index.year).apply(lambda r: (1 + r).prod() - 1)
    return {str(int(k)): float(v) for k, v in y.items()}


def _monthly(net):
    m = net.groupby([net.index.year, net.index.month]).apply(lambda r: (1 + r).prod() - 1)
    out = {}
    for (yr, mo), v in m.items():
        out.setdefault(str(int(yr)), {})[str(int(mo))] = float(v)
    return out


def _drawdown_series(net):
    c = (1 + net).cumprod()
    hwm = c.cummax()
    dd = c / hwm - 1
    out = [(str(d.date()), float(v)) for d, v in dd.items() if pd.notna(v)]
    # Resample to weekly to reduce size
    return out[::7]


def _equity_series(net):
    c = (1 + net).cumprod() * 10000
    out = [(str(d.date()), float(v)) for d, v in c.items() if pd.notna(v)]
    return out[::3]  # every 3rd day


def _rolling_sr_series(net, window=365):
    sr = net.rolling(window).mean() / net.rolling(window).std() * np.sqrt(util.DPY)
    out = [(str(d.date()), float(v)) for d, v in sr.items() if pd.notna(v)]
    return out[::7]


def main():
    cp = load_prices()
    macro = load_macro(cp.index)
    all_sw = SV.build_all(cp, macro)
    sw = {k: all_sw[k] for k in FINAL_SLEEVES}

    net = build_portfolio(cp, sw, target_vol=TARGET_VOL, dd_floor=DD_FLOOR).fillna(0.0)
    net.to_frame("crypto_apex_ret").to_csv(OUT / "crypto_apex_returns.csv")

    # Benchmarks
    btc_r = cp["BTC"].pct_change().fillna(0.0).clip(-0.30, 0.30)
    eq_r = cp[SURVIVORS].pct_change().mean(axis=1).fillna(0.0).clip(-0.30, 0.30)

    # Survivors-only for bias comparison
    cp_surv = load_prices(coins=SURVIVORS)
    sw_surv = {k: v for k, v in SV.build_all(cp_surv, macro).items() if k in FINAL_SLEEVES}
    net_surv = build_portfolio(cp_surv, sw_surv, target_vol=TARGET_VOL, dd_floor=DD_FLOOR).fillna(0.0)

    # Build data blob
    data = {
        "name": "CRYPTO-APEX",
        "tagline": "APEX Methodology on 20-Coin Crypto Universe with Survivorship-Bias Accounting",
        "description": "Three uncorrelated sleeves (ACCEL+HURST+DOMINANCE) + BTC-regime master kill-switch + dead-coin catastrophe filter. 15 survivors + 5 delisted coins (LUNA, USTC, FTT, MATIC, UNI).",
        "as_of": str(net.index[-1].date()),
        "universe": {
            "full": ALL_COINS,
            "survivors": SURVIVORS,
            "dead": DEAD,
        },
        "config": {
            "sleeves": FINAL_SLEEVES,
            "target_vol": TARGET_VOL,
            "dd_floor": DD_FLOOR,
            "tc_bps": 30.0,
            "ret_cap": 0.30,
            "catastrophe_dd": -0.50,
        },
        "metrics": {
            "full": metrics(net),
            "is": metrics(regime_slice(net, str(net.index[0].date()), "2022-06-30")),
            "oos": metrics(regime_slice(net, OOS_START, "2027-12-31")),
            "y2018": metrics(regime_slice(net, "2018-01-01", "2018-12-31")),
            "y2019": metrics(regime_slice(net, "2019-01-01", "2019-12-31")),
            "y2020": metrics(regime_slice(net, "2020-01-01", "2020-12-31")),
            "y2021": metrics(regime_slice(net, "2021-01-01", "2021-12-31")),
            "y2022": metrics(regime_slice(net, "2022-01-01", "2022-12-31")),
            "y2324": metrics(regime_slice(net, "2023-01-01", "2024-12-31")),
            "y2025plus": metrics(regime_slice(net, "2025-01-01", "2027-12-31")),
        },
        "benchmarks": {
            "BTC_HOLD": {
                "full": metrics(btc_r),
                "oos": metrics(regime_slice(btc_r, OOS_START, "2027-12-31")),
            },
            "EQ_WEIGHT": {
                "full": metrics(eq_r),
                "oos": metrics(regime_slice(eq_r, OOS_START, "2027-12-31")),
            },
        },
        "survivorship": {
            "full_universe": metrics(net),
            "survivors_only": metrics(net_surv),
            "bias_sr": metrics(net_surv)["sharpe"] - metrics(net)["sharpe"],
            "bias_cagr": metrics(net_surv)["cagr"] - metrics(net)["cagr"],
        },
        "yearly": _yearly(net),
        "monthly": _monthly(net),
        "equity": _equity_series(net),
        "drawdown": _drawdown_series(net),
        "rolling_sr": _rolling_sr_series(net),
    }

    (OUT / "crypto_apex_factsheet.json").write_text(json.dumps(data, default=str))
    print(f"Saved factsheet: {OUT / 'crypto_apex_factsheet.json'}")
    print(f"  Full SR: {data['metrics']['full']['sharpe']}")
    print(f"  OOS SR:  {data['metrics']['oos']['sharpe']}")
    print(f"  CAGR:    {data['metrics']['full']['cagr']*100:.1f}%")
    print(f"  MDD:     {data['metrics']['full']['mdd']*100:.1f}%")


if __name__ == "__main__":
    main()
