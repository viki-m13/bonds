import pandas as pd, numpy as np, warnings, time; warnings.filterwarnings("ignore")
t0=time.time(); data=pd.read_pickle("/tmp/sp_daily.pkl")
tk=[t for t,df in data.items() if len(df)>250]
dates=sorted(set().union(*[set(data[t].index) for t in tk])); dates=[d for d in dates if d>=pd.Timestamp("2000-01-01")]
C=pd.DataFrame({t:data[t]["c"] for t in tk}).reindex(dates)
# RSI(5) Wilder
def rsi(px,n=5):
    d=px.diff(); up=d.clip(lower=0); dn=(-d).clip(lower=0)
    ru=up.ewm(alpha=1/n,min_periods=n).mean(); rd=dn.ewm(alpha=1/n,min_periods=n).mean()
    return 100-100/(1+ru/rd.replace(0,np.nan))
RSI=rsi(C,5); MA200=C.rolling(200,min_periods=200).mean()
# entry at close t when RSI(5)<20 and close>200MA ; exit at close t+5 ; $1000/trade
entry=((RSI<20)&(C>MA200)).fillna(False)
fwd5=C.shift(-5)/C-1                                  # 5-trading-day forward return, enter/exit at close
rets=fwd5.where(entry).stack().dropna()
n=len(rets); wr=(rets>0).mean(); aw=rets[rets>0].mean(); al=rets[rets<0].mean()
print(f"BUY THE DIP repro (S&P subset {len(tk)} names, {dates[0].date()}..{dates[-1].date()}):")
print(f"  trades={n}  win rate={wr*100:.2f}%  avg trade={rets.mean()*100:+.3f}%  avgWin={aw*100:.2f}%  avgLoss={al*100:.2f}%  W/L ratio={aw/-al:.2f}")
print(f"  PUBLISHED: ~25,000 trades, win rate 56.81%, avg win > avg loss, since 2000")
# equity: $1000/trade, non-compounding sum of P&L / also compounded portfolio
print(f"  total P&L per $1000/trade: ${ (rets*1000).sum():,.0f} over {n} trades")
print(f"DONE t={time.time()-t0:.0f}s")
