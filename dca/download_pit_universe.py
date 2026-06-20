"""Download daily OHLCV for every ticker that was an S&P 500 member at any
point since 2004, using the fja05680 point-in-time membership list.

Delisted tickers that Yahoo no longer serves will fail; we record coverage so
survivorship effects can be quantified rather than ignored.
"""
import os, time, json
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIT = os.path.join(ROOT, "data", "pit", "sp500_pit_membership.csv")
OUT = os.path.join(ROOT, "data", "pit", "prices")
os.makedirs(OUT, exist_ok=True)

mem = pd.read_csv(PIT)
mem["date"] = pd.to_datetime(mem["date"])
mem = mem[mem["date"] >= "2004-01-01"]
tickers = sorted({t for row in mem["tickers"] for t in row.split(",")})
print(f"{len(tickers)} unique PIT tickers")

ok, fail = [], []
BATCH = 40
for i in range(0, len(tickers), BATCH):
    batch = tickers[i:i+BATCH]
    ymap = {t: t.replace(".", "-") for t in batch}
    todo = {t: y for t, y in ymap.items()
            if not os.path.exists(os.path.join(OUT, f"{t.replace('.','-')}.csv"))}
    if not todo:
        ok += batch
        continue
    try:
        df = yf.download(list(todo.values()), start="2004-01-01", interval="1d",
                         auto_adjust=True, progress=False, threads=True,
                         group_by="ticker")
    except Exception as e:
        print("batch error", i, e); time.sleep(10); continue
    for t, y in todo.items():
        try:
            sub = df[y].dropna(how="all")
            if len(sub) < 50:
                fail.append(t); continue
            sub.round(6).to_csv(os.path.join(OUT, f"{y}.csv"))
            ok.append(t)
        except Exception:
            fail.append(t)
    print(f"{i+BATCH}/{len(tickers)} ok={len(ok)} fail={len(fail)}", flush=True)
    time.sleep(1.5)

json.dump({"ok": sorted(ok), "fail": sorted(fail)},
          open(os.path.join(ROOT, "data", "pit", "coverage.json"), "w"), indent=1)
print("DONE ok:", len(ok), "fail:", len(fail))
