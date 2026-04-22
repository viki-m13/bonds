"""PHOENIX full refresh — runs the entire pipeline on fresh data.

Steps (all must complete for the webapp to show fresh numbers):
  1. fetch_latest: pull new ETF prices via yfinance (handles already-current CSVs)
  2. fetch_fred_latest: pull fresh macro series (VIX, HY OAS, rates)
  3. re-run each sleeve strategy to extend sleeve returns CSVs through latest data
  4. re-run phoenix_production.py to produce fresh net_ret
  5. regenerate phoenix_factsheet.json
  6. regenerate audit bundle (yearly contrib, last-30-days, positions)
  7. run live_signal.py to get today's orders
  8. inject fresh F / A / L / LIVE into docs/phoenix.html

Outputs:
  - All *_returns.csv updated through last available market date
  - data/results/phoenix_factsheet.json        (drives const F)
  - data/results/phoenix_v2_live.json          (drives const L)
  - data/results/phoenix_v2_audit.json         (drives const A)
  - data/results/live_signal.json              (drives const LIVE)
  - docs/phoenix.html                          (injected data)
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
ALT = ROOT / "alt"
R = ROOT / "data/results"


def run(cmd: list, desc: str):
    print(f"\n{'='*70}\n{desc}\n{'='*70}")
    proc = subprocess.run(cmd, cwd=ROOT, text=True)
    if proc.returncode != 0:
        print(f"[WARN] {desc} exited with code {proc.returncode}; continuing...")
        return False
    return True


def fetch_latest_prices():
    """Delegate to live_signal.py's fetch methods."""
    sys.path.insert(0, str(ALT))
    import live_signal as ls
    universe = list(set(ls.UNIVERSE + ["SPY", "BIL", "QQQ", "TLT", "IEF",
                                         "GLD", "USO", "XLK", "XLE", "XLF", "SMH",
                                         "VNQ", "EEM", "FXI"]))
    print("Fetching latest ETF prices via yfinance...")
    ls.fetch_latest(universe)
    print("\nFetching latest FRED macro series...")
    ls.fetch_fred_latest()


def regenerate_factsheet():
    """Regenerate phoenix_factsheet.json from latest phoenix_production_returns.csv."""
    import pandas as pd
    import numpy as np

    prod = pd.read_csv(R/"phoenix_production_returns.csv", parse_dates=["Date"]).set_index("Date")
    ret = prod["net_ret"]

    def load_etf(t):
        p = ROOT / "data/etfs" / f"{t}.csv"
        if not p.exists(): return None
        df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
        return df[~df.index.duplicated(keep="first")]

    spy = load_etf("SPY"); tlt = load_etf("TLT")
    spy_ret = spy["Open"].pct_change().fillna(0) if spy is not None else pd.Series(dtype=float)
    tlt_ret = tlt["Open"].pct_change().fillna(0) if tlt is not None else pd.Series(dtype=float)
    s6040 = 0.6 * spy_ret + 0.4 * tlt_ret
    dates = ret.index
    spy_r = spy_ret.reindex(dates).fillna(0)
    s6040_r = s6040.reindex(dates).fillna(0)

    def gr(r, base=10000): return base * (1 + r).cumprod()
    eq_p = gr(ret); eq_spy = gr(spy_r); eq_6040 = gr(s6040_r)

    def metrics(r):
        r = r.dropna()
        if len(r) == 0: return {}
        mu = r.mean() * 252; sd = r.std() * np.sqrt(252)
        sr = mu / sd if sd > 0 else 0
        c = (1 + r).cumprod(); dd = (c / c.cummax() - 1).min()
        yrs = len(r) / 252
        cagr = c.iloc[-1] ** (1 / yrs) - 1 if c.iloc[-1] > 0 else -1
        neg = r[r < 0]
        sortino = mu / (neg.std() * np.sqrt(252)) if len(neg) > 0 and neg.std() > 0 else 0
        return {"sharpe": float(sr), "cagr": float(cagr), "vol": float(sd),
                "mdd": float(dd), "sortino": float(sortino),
                "calmar": float(cagr / abs(dd)) if dd < 0 else 0,
                "navx": float(c.iloc[-1])}

    IS_END = "2018-12-31"; OOS_START = "2019-01-02"
    m_full = metrics(ret); m_is = metrics(ret.loc[:IS_END]); m_oos = metrics(ret.loc[OOS_START:])
    m_spy = metrics(spy_r); m_6040 = metrics(s6040_r)

    # Components (5 sleeves)
    comps = {}
    for sname, fn, col in [("VANGUARD", "vanguard_returns.csv", "net_ret"),
                            ("ORION", "orion_returns.csv", "orion"),
                            ("HELIOS", "helios_returns.csv", "ret"),
                            ("QUANTUM", "quantum_returns.csv", "ret"),
                            ("CRYPTO", "crypto_returns.csv", "ret")]:
        df = pd.read_csv(R/fn, parse_dates=[0] if sname == "VANGUARD" else ["Date"])
        if sname == "VANGUARD":
            s = df.set_index(df.columns[0])[col]
        else:
            s = df.set_index("Date")[col]
        comps[sname] = metrics(s.reindex(dates).fillna(0))

    # Correlations (5x5)
    corr_df = pd.DataFrame()
    for sname, fn, col in [("VANGUARD", "vanguard_returns.csv", "net_ret"),
                            ("ORION", "orion_returns.csv", "orion"),
                            ("HELIOS", "helios_returns.csv", "ret"),
                            ("QUANTUM", "quantum_returns.csv", "ret"),
                            ("CRYPTO", "crypto_returns.csv", "ret")]:
        df = pd.read_csv(R/fn, parse_dates=[0] if sname == "VANGUARD" else ["Date"])
        if sname == "VANGUARD":
            s = df.set_index(df.columns[0])[col]
        else:
            s = df.set_index("Date")[col]
        corr_df[sname] = s.reindex(dates).fillna(0)
    corr_dict = {k: {k2: round(float(v), 3) for k2, v in row.items()}
                 for k, row in corr_df.corr().to_dict().items()}

    # Yearly + monthly
    mr = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly = {}
    for d, v in mr.items():
        y = str(d.year); m = d.month
        monthly.setdefault(y, {})[m] = float(v)
    yr = ret.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    yearly = [{"year": int(d.year), "ret": float(v)} for d, v in yr.items()]

    # DD + equity + rolling sharpe
    cum = (1 + ret).cumprod()
    dd_series = (cum / cum.cummax() - 1)
    dd_w = dd_series.resample("W").last()
    dd_data = [{"d": d.strftime("%Y-%m-%d"), "v": float(v)}
               for d, v in dd_w.items() if not pd.isna(v)]
    eq_w = pd.DataFrame({"p": eq_p, "spy": eq_spy, "s6040": eq_6040}).resample("W").last()
    eq_data = []
    for d, row in eq_w.iterrows():
        if pd.isna(row["p"]): continue
        eq_data.append({"d": d.strftime("%Y-%m-%d"),
                        "p": round(float(row["p"]), 2),
                        "s": round(float(row["spy"]), 2),
                        "b": round(float(row["s6040"]), 2)})
    roll = ret.rolling(252)
    rs = (roll.mean() * 252) / (roll.std() * np.sqrt(252))
    rs = rs.resample("W").last().dropna()
    rs_data = [{"d": d.strftime("%Y-%m-%d"), "v": round(float(v), 3)} for d, v in rs.items()]

    mult = prod["total_mult"]
    out = {
        "meta": {"name": "PHOENIX",
                 "subtitle": "5 uncorrelated LETF strategies with daily risk sizing",
                 "start": str(ret.index[0].date()), "end": str(ret.index[-1].date()),
                 "n_days": int(len(ret)),
                 "weights": {"VANGUARD": 0.236, "ORION": 0.327, "HELIOS": 0.185,
                             "QUANTUM": 0.152, "CRYPTO": 0.101},
                 "target_vol": 0.15, "leverage_cap": 1.0, "no_margin": True},
        "metrics": {"full": m_full, "is": m_is, "oos": m_oos, "spy": m_spy, "s6040": m_6040},
        "components": comps, "correlations": corr_dict,
        "overlay": {"avg_total_mult": float(mult.mean()),
                    "pct_at_full_exposure": float((mult > 0.99).mean()),
                    "pct_below_50": float((mult < 0.50).mean()),
                    "target_vol": 0.15, "cap": 1.0, "dd_floor": -0.10, "vol_gate_pct": 0.99},
        "is_oos_gap": round(abs(m_is["sharpe"] - m_oos["sharpe"]), 4),
        "yearly": yearly, "monthly": monthly,
        "equity": eq_data, "drawdown": dd_data, "rolling_sharpe_12m": rs_data,
    }
    (R/"phoenix_factsheet.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"  Wrote phoenix_factsheet.json — Sharpe {m_full['sharpe']:.2f} / OOS {m_oos['sharpe']:.2f} / CAGR {m_full['cagr']*100:.1f}%")
    return m_full, m_is, m_oos


def regenerate_audit_bundle():
    """Regenerate phoenix_v2_live.json + update phoenix_v2_audit.json with fresh data."""
    import pandas as pd
    import numpy as np

    prod = pd.read_csv(R/"phoenix_production_returns.csv", parse_dates=["Date"]).set_index("Date")
    ret = prod["net_ret"]
    mult = prod["total_mult"]

    van = pd.read_csv(R/"vanguard_returns.csv", parse_dates=[0], index_col=0)["net_ret"]
    ori = pd.read_csv(R/"orion_returns.csv", parse_dates=["Date"]).set_index("Date")["orion"]
    hel = pd.read_csv(R/"helios_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    qua = pd.read_csv(R/"quantum_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    cry = pd.read_csv(R/"crypto_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    idx = ret.index
    van = van.reindex(idx).fillna(0); ori = ori.reindex(idx).fillna(0)
    hel = hel.reindex(idx).fillna(0); qua = qua.reindex(idx).fillna(0)
    cry = cry.reindex(idx).fillna(0)

    w = {"VAN": 0.236, "ORI": 0.327, "HEL": 0.185, "QUA": 0.152, "CRY": 0.101}

    # Yearly contribution (recompute through latest)
    yrs = sorted(set(idx.year))
    yearly_rows = []
    for y in yrs:
        m = idx.year == y
        row = {"year": y}
        total = 0
        for nm, s in [("VAN", van), ("ORI", ori), ("HEL", hel), ("QUA", qua), ("CRY", cry)]:
            c = (s[m] * mult[m]).sum() * w[nm]
            row[nm] = round(float(c) * 100, 1)
            total += c
        row["total"] = round(float(total) * 100, 1)
        yearly_rows.append(row)

    # Last 30 days per-sleeve contribs
    last30 = idx[-30:]
    l30 = []
    for nm, s in [("VAN", van), ("ORI", ori), ("HEL", hel), ("QUA", qua), ("CRY", cry)]:
        r30 = float((1 + s.loc[last30]).prod() - 1)
        l30.append({"sleeve": nm, "weight": w[nm], "sleeve_ret": r30, "contrib": w[nm] * r30})

    # Overlay mult series (last ~120 days)
    recent = mult.loc[idx[-120:]]
    mult_series = [{"d": d.strftime("%Y-%m-%d"), "v": round(float(v), 4)}
                   for d, v in recent.items()]

    # Momentum ranking (latest close)
    ETF = ROOT / "data/etfs"
    UNIVERSE = ["TQQQ", "UPRO", "QLD", "SSO", "SOXL", "TECL", "FAS", "ERX", "DRN",
                "EDC", "YINN", "UCO", "UGL", "NUGT", "TMF", "UBT", "TYD", "IBIT"]
    mom63 = {}
    for t in UNIVERSE:
        p = ETF / f"{t}.csv"
        if not p.exists(): continue
        df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        c = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(c) >= 64:
            mom63[t] = round(float(c.iloc[-1] / c.iloc[-64] - 1), 4)

    # Regime snapshot
    def load_fred(s):
        p = ROOT / "data/fred" / f"{s}.csv"
        if not p.exists(): return None
        d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
        return pd.to_numeric(d.iloc[:, 0], errors="coerce")

    spy_file = ETF / "SPY.csv"
    spy = pd.read_csv(spy_file, parse_dates=["Date"]).set_index("Date").sort_index()
    spy = spy[~spy.index.duplicated()]
    spy_c = pd.to_numeric(spy["Close"], errors="coerce")
    spy_ma = spy_c.rolling(200).mean()
    vix = load_fred("VIXCLS")
    hy = load_fred("BAMLH0A0HYM2")
    last_spy_date = spy_c.index[-1]
    spy_ok = bool(spy_c.iloc[-1] > spy_ma.iloc[-1] and spy_ma.diff(20).iloc[-1] > 0) \
             if not pd.isna(spy_ma.iloc[-1]) else False
    vix_last = float(vix.reindex([last_spy_date], method='ffill').iloc[0]) if vix is not None else None
    hy_slope = float((hy - hy.shift(20)).reindex([last_spy_date], method='ffill').iloc[0]) if hy is not None else None
    regime_pass = spy_ok and (hy_slope is not None and hy_slope < 1.0) and (vix_last is not None and vix_last < 30)

    L = {
        "as_of": str(idx[-1].date()),
        "data_thru": str(last_spy_date.date()),
        "current_mult": float(mult.iloc[-1]),
        "mult_12m_avg": float(mult.iloc[-252:].mean()) if len(mult) > 252 else float(mult.mean()),
        "mult_12m_min": float(mult.iloc[-252:].min()) if len(mult) > 252 else float(mult.min()),
        "mult_series_120d": mult_series,
        "last30_sleeves": l30,
        "yearly_contrib": yearly_rows,
        "regime": {
            "spy_above_200dma_sloping_up": spy_ok,
            "hy_oas_20d_slope_bps": hy_slope,
            "vix_level": vix_last,
            "macro_gate_pass": regime_pass,
        },
        "mom63": mom63,
    }
    (R/"phoenix_v2_live.json").write_text(json.dumps(L, separators=(",", ":")))
    print(f"  Wrote phoenix_v2_live.json — as_of {L['as_of']}, regime {regime_pass}")

    # Update audit bundle's positions/trades from live_signal.json + refresh yearly_contrib/last30
    audit = json.loads((R/"phoenix_v2_audit.json").read_text())
    audit["yearly_contrib"] = yearly_rows
    audit["last30_sleeves"] = l30
    audit["last_date"] = str(idx[-1].date())
    # Pull positions + trades from fresh live_signal
    live = json.loads((R/"live_signal.json").read_text())
    audit["positions"] = {
        "as_of": live["context"]["as_of"],
        "overlay_mult": live["context"]["overlay_mult"],
        "current": live["target_positions"],
        "sleeve_weights": {"VANGUARD": 0.236, "ORION": 0.327, "HELIOS": 0.185,
                           "QUANTUM": 0.152, "CRYPTO": 0.101},
    }
    audit["recent_trades"] = live["recent_trades_30d"]
    (R/"phoenix_v2_audit.json").write_text(json.dumps(audit, separators=(",", ":")))
    print(f"  Updated phoenix_v2_audit.json")


def inject_into_html():
    """Inject F, A, L, LIVE into docs/phoenix.html."""
    html_path = ROOT / "docs/phoenix.html"
    html = html_path.read_text()
    for name, path in [("F", R/"phoenix_factsheet.json"),
                       ("A", R/"phoenix_v2_audit.json"),
                       ("L", R/"phoenix_v2_live.json"),
                       ("LIVE", R/"live_signal.json")]:
        data = path.read_text()
        pat = re.compile(rf'const {name} = \{{.*?\}};', re.DOTALL)
        replacement = f'const {name} = {data};'
        if pat.search(html):
            html = pat.sub(lambda m: replacement, html, count=1)
            print(f"  Re-injected const {name}")
        else:
            print(f"  [WARN] const {name} pattern not found in HTML")
    html_path.write_text(html)


def get_sleeve_last_date(csv_name):
    """Return the last date in a sleeve's returns CSV, or None if missing."""
    import pandas as pd
    p = R / csv_name
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        return pd.to_datetime(df[date_col].iloc[-1]).date()
    except Exception:
        return None


def get_latest_market_date():
    """Find the latest common market-close date across core ETFs (SPY, QQQ, IBIT)."""
    import pandas as pd
    from pathlib import Path as P
    ETF = ROOT / "data/etfs"
    dates = []
    for t in ["SPY", "QQQ", "IBIT"]:
        p = ETF / f"{t}.csv"
        if p.exists():
            df = pd.read_csv(p)
            dates.append(pd.to_datetime(df["Date"].iloc[-1]).date())
    return min(dates) if dates else None


def main():
    # 1+2. Fetch fresh prices + FRED
    fetch_latest_prices()

    # 3. Check which sleeves need extending. If a sleeve's last date >= latest
    #    common market date, it's already up to date — skip the re-run.
    latest = get_latest_market_date()
    print(f"\nLatest common market date across SPY/QQQ/IBIT: {latest}")

    sleeve_map = [
        ("VANGUARD", "vanguard_strategy.py", "vanguard_returns.csv"),
        ("ORION",    "orion_strategy.py",    "orion_returns.csv"),
        ("HELIOS",   "helios_strategy.py",   "helios_returns.csv"),
        ("QUANTUM",  "quantum_strategy.py",  "quantum_returns.csv"),
    ]
    any_extended = False
    for name, script, csv_name in sleeve_map:
        last = get_sleeve_last_date(csv_name)
        if last is None:
            print(f"\n{name}: no existing returns; full run needed")
            run(["python3", str(ALT / script)], f"Build {name}")
            any_extended = True
        elif latest is not None and last >= latest:
            print(f"\n{name}: already at {last} (matches market date) — skip rerun ✓")
        else:
            print(f"\n{name}: last date {last} < market {latest} — extend to today")
            run(["python3", str(ALT / script)], f"Extend {name}")
            any_extended = True

    # CRYPTO sleeve is rebuilt by phoenix_v2_crypto.py (idempotent, ~3s)
    cry_last = get_sleeve_last_date("crypto_returns.csv")
    if cry_last is None or (latest and cry_last < latest):
        print(f"\nCRYPTO: last date {cry_last} — extending")
        run(["python3", str(ALT / "phoenix_v2_crypto.py")], "Extend CRYPTO sleeve")
        any_extended = True
    else:
        print(f"\nCRYPTO: already at {cry_last} — skip rerun ✓")

    # 4. Re-run production strategy (reads all 5 sleeves, blends + applies overlay)
    # This is fast (~2s) and should always run so the final net_ret reflects the
    # current sleeve data.
    run(["python3", str(ALT / "phoenix_production.py")], "Re-run PHOENIX production")

    # 5. Regenerate factsheet
    print("\n=== Regenerating factsheet ===")
    regenerate_factsheet()

    # 6. Regenerate live_signal.json (latest orders)
    run(["python3", str(ALT / "live_signal.py"), "--skip-fetch"], "Generate live_signal.json")

    # 7. Regenerate audit bundle (needs live_signal.json to be current)
    print("\n=== Regenerating audit bundle ===")
    regenerate_audit_bundle()

    # 8. Inject all data into HTML
    print("\n=== Injecting into docs/phoenix.html ===")
    inject_into_html()

    print("\n✅ Full refresh complete.")


if __name__ == "__main__":
    main()
