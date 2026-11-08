import pandas as pd
import yfinance as yf

E4 = ["SPY", "QQQ", "RSP", "IVE"]
px = yf.download(E4, start="1999-03-01", auto_adjust=True, progress=False)["Close"]
eps = {"Burbuja .com 2000-02": ("2000-03-01", "2002-12-31"), "Crisis 2008": ("2007-10-01", "2009-06-30"),
       "2011": ("2011-04-15", "2011-10-31"), "2018 Q4": ("2018-09-15", "2018-12-31"), "Covid 2020": ("2020-02-10", "2020-04-30"),
       "Tasas 2022": ("2021-12-15", "2022-10-31"), "Aranceles 2025": ("2025-02-10", "2025-04-30"), "Marzo 2026": ("2026-01-15", "2026-04-30")}
for k, (a, b) in eps.items():
    w = px.loc[a:b]
    o = []
    for e in E4:
        s = w[e].dropna()
        if len(s) < 10 or s.index[0] > pd.Timestamp(a) + pd.Timedelta(days=20):
            o.append(f"{e} s/d")
            continue
        dd = (s / s.cummax() - 1).min()
        o.append(f"{e} {dd*100:.0f}%")
    print("D|" + k + "|" + " | ".join(o), flush=True)
for e in E4:
    s = px[e].dropna()
    rec = s / s.cummax() - 1
    print("F|" + e + "|desde " + str(s.index[0])[:7] + f"|peor {rec.min()*100:.0f}%", flush=True)
