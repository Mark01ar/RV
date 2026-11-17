import yfinance as yf
import pandas as pd
tk = ["IWF", "IWD", "IVW", "IVE", "SPY", "RSP", "QQQ", "EEM", "EFA", "EWZ", "EWJ", "EWY", "EWT", "XLK", "XLF", "XLE", "XLP", "XLV", "XLI", "XLU", "VTV", "VUG", "^RLG", "^RLV"]
px = yf.download(tk, start="1995-01-01", auto_adjust=True, progress=False)["Close"]
m = px.resample("ME").last()
m.to_csv("tmp/mensual.csv")
info = {}
for t in tk:
    try:
        i = yf.Ticker(t).info
        info[t] = {k: i.get(k) for k in ["trailingPE", "forwardPE", "priceToBook", "yield", "dividendYield", "longName"]}
    except Exception as e:
        info[t] = str(e)
pd.DataFrame(info).T.to_csv("tmp/info.csv")
print(m.tail(3).T)
print(m.apply(lambda s: s.first_valid_index()))
