"""NOVA30 — Intraday momentum: first-half-hour → last-half-hour (Gao-Han-Li-Zhou 2018).

RESULT: DEAD END.  Once TC is paid on every trade (252 round-trips/yr),
both directions (continuation and reversal) go negative.  At 2 bps TC
continuation gives OOS -2.64 and reversal OOS -1.45 (the earlier
"reversal OOS +2.64" was an arithmetic bug: flipping sign on df also
flipped sign on TC, so reversal was effectively RECEIVING TC instead
of paying).  Kept here as a documented negative result.

Rule: at 15:30, if morning (09:30→10:00) return > 0 go long SPY to
15:55; size ±1.  Reversal is the opposite sign.  TC = 2 bps/trade,
charged every active day.  Basket: SPY, QQQ, IWM, DIA, GLD."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import stats


INTRA = Path("/home/user/bonds/data/intraday_5min")
TC_BPS = 2.0
TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


def fhh_lhh_returns(t):
    """Compute first-half-hour return and last-half-hour return per day."""
    df = pd.read_csv(INTRA / f"{t}.csv", parse_dates=["ts"])
    df["date"] = pd.to_datetime(df["ts"].dt.date)
    df["time"] = df["ts"].dt.time
    # First-half-hour: open (09:30) to 10:00 close
    open_px = df.groupby("date")["open"].first()
    # 10:00 corresponds to the 09:55 bar close (09:55-10:00 range)
    px_1000 = df[df["time"] == pd.to_datetime("10:00").time()].set_index("date")["close"]
    if px_1000.empty:
        px_1000 = df[df["time"] == pd.to_datetime("09:55").time()].set_index("date")["close"]
    # Last-half-hour: 15:30 close to 16:00 close
    px_1530 = df[df["time"] == pd.to_datetime("15:30").time()].set_index("date")["close"]
    px_1555 = df[df["time"] == pd.to_datetime("15:55").time()].set_index("date")["close"]

    for s in (open_px, px_1000, px_1530, px_1555):
        s.index = pd.to_datetime(s.index)
    # dedupe
    def dd(s): return s[~s.index.duplicated(keep="first")].sort_index()
    open_px, px_1000, px_1530, px_1555 = map(dd, [open_px, px_1000, px_1530, px_1555])
    common = open_px.index.intersection(px_1000.index).intersection(
        px_1530.index).intersection(px_1555.index)
    fhh = (px_1000.loc[common] / open_px.loc[common]) - 1
    lhh = (px_1555.loc[common] / px_1530.loc[common]) - 1
    return fhh, lhh


def sleeve(t):
    fhh, lhh = fhh_lhh_returns(t)
    sig = np.sign(fhh)          # +1 if FHH positive, -1 else
    ret = sig * lhh             # position at 15:30, exit at 15:55
    ret = ret - (TC_BPS / 1e4) * (sig != 0).astype(int)  # flat TC per trade
    return ret.rename(t)


def main():
    sleeves = {t: sleeve(t) for t in TICKERS}
    df = pd.DataFrame(sleeves).dropna(how="all").fillna(0)

    warm = pd.Timestamp("2016-03-01")
    CUT = pd.Timestamp("2022-01-01")
    df = df.loc[warm:]

    print("Per-ticker (intraday first→last half-hour):")
    for c in df.columns:
        x = df[c]
        sf = stats(x, c); si = stats(x.loc[:CUT], "")
        so = stats(x.loc[CUT:], "")
        print(f"  {c:10s} SR={sf['sharpe']:>5.2f}  IS={si['sharpe']:>5.2f}  "
              f"OOS={so['sharpe']:>5.2f}  Vol={sf['vol']:>5.2f}%  MDD={sf['mdd']:>7.2f}%")

    # Reversal variant — re-run with proper TC (must pay TC on BOTH directions).
    # NB: naive `-df` flips both return AND TC — that would RECEIVE TC, a bug.
    def sleeve_rev(t):
        fhh, lhh = fhh_lhh_returns(t)
        sig = -np.sign(fhh)
        ret = sig * lhh - (TC_BPS / 1e4) * (sig != 0).astype(int)
        return ret.rename(t)
    df_rev = pd.DataFrame({t: sleeve_rev(t) for t in TICKERS}).fillna(0).loc[warm:]
    print("\nReversal variant (proper TC):")
    for c in df_rev.columns:
        x = df_rev[c]
        sf = stats(x, c); si = stats(x.loc[:CUT], "")
        so = stats(x.loc[CUT:], "")
        print(f"  {c:10s} SR={sf['sharpe']:>5.2f}  IS={si['sharpe']:>5.2f}  "
              f"OOS={so['sharpe']:>5.2f}  Vol={sf['vol']:>5.2f}%  MDD={sf['mdd']:>7.2f}%")

    # EW basket (both directions)
    ew_mom = df.mean(axis=1)
    s = stats(ew_mom, "MOM EW basket")
    print(f"\n{s['label']:30s} SR={s['sharpe']:>5.2f}  IS="
          f"{stats(ew_mom.loc[:CUT],'')['sharpe']:.2f}  "
          f"OOS={stats(ew_mom.loc[CUT:],'')['sharpe']:.2f}  "
          f"Vol={s['vol']:.2f}%  MDD={s['mdd']:.2f}%")
    ew_rev = df_rev.mean(axis=1)
    s = stats(ew_rev, "REV EW basket")
    print(f"{s['label']:30s} SR={s['sharpe']:>5.2f}  IS="
          f"{stats(ew_rev.loc[:CUT],'')['sharpe']:.2f}  "
          f"OOS={stats(ew_rev.loc[CUT:],'')['sharpe']:.2f}  "
          f"Vol={s['vol']:.2f}%  MDD={s['mdd']:.2f}%")

    best = ew_mom if stats(ew_mom.loc[:CUT], "")['sharpe'] > 0 else ew_rev
    tag = "mom" if stats(ew_mom.loc[:CUT], "")['sharpe'] > 0 else "rev"
    out = pd.DataFrame({"NOVA30": best}).join(df, how="outer")
    out.to_csv("/home/user/bonds/data/results/nova30_returns.csv")
    print(f"\nSaved NOVA30 = {tag} variant")


if __name__ == "__main__":
    main()
