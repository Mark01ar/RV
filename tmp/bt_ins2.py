import json
import time
import numpy as np
import pandas as pd
import yfinance as yf

COSTO = 0.006
N = 20
d = json.load(open("data/datos.json"))
tickers = [x["ticker"] for x in d["tickers"] if not x.get("sin_datos")]
LIQ = {x["ticker"] for x in d["tickers"] if x.get("liquidez") in ("Alta", "Media", "Baja")}

frames = {c: [] for c in ["Close", "High", "Low", "Volume"]}
for i in range(0, len(tickers), 40):
    lote = tickers[i:i + 40]
    for _ in range(3):
        try:
            df = yf.download(lote, start="2003-06-01", auto_adjust=True, progress=False, group_by="column", threads=True)
            break
        except Exception as e:
            print("reintento", e, flush=True)
            time.sleep(10)
    for c in frames:
        frames[c].append(df[c])
    time.sleep(2)
P = {c: pd.concat(v, axis=1).loc[:, lambda x: ~x.columns.duplicated()] for c, v in frames.items()}
for c in P:
    P[c].index = pd.to_datetime(P[c].index).tz_localize(None)
close = P["Close"]
spy = yf.download("SPY", start="2003-06-01", auto_adjust=True, progress=False)["Close"].squeeze()
irx = yf.download("^IRX", start="2003-06-01", auto_adjust=True, progress=False)["Close"].squeeze()
spy.index = pd.to_datetime(spy.index).tz_localize(None)
irx.index = pd.to_datetime(irx.index).tz_localize(None)
cal = spy.dropna().index
for c in P:
    P[c] = P[c].reindex(cal)
P["Close"] = P["Close"].ffill(limit=3)
close = P["Close"]
print("precios", close.shape, float(close.iloc[-1].notna().mean()), flush=True)

dias = close.index
fechas = dias.to_series().groupby(dias.to_period("M")).max()
fechas = fechas[(fechas >= "2005-12-01")]
fechas = fechas.iloc[:-1] if fechas.index[-1] == pd.Period(pd.Timestamp.today(), "M") else fechas
cm = close.reindex(dias).ffill(limit=5)
ma200 = close.rolling(200, min_periods=150).mean()
tp = (P["High"] + P["Low"] + P["Close"]) / 3
cv = P["Volume"].fillna(0).cumsum()
ctpv = (tp * P["Volume"]).fillna(0).cumsum()


FE = list(fechas.values)
pm = cm.loc[FE]
mom = pm.shift(1) / pm.shift(12) - 1
vsma = pm / ma200.loc[FE] - 1
fwd = pm.shift(-1) / pm - 1
spym = spy.reindex(dias).ffill().loc[FE]
spyf = spym.shift(-1) / spym - 1
cajaf = (irx.reindex(dias).ffill().loc[FE] / 100 / 12)
def rk(s):
    return s.rank(pct=True)

I = pd.read_csv("tmp/insiders.csv", parse_dates=["fecha"]).dropna(subset=["fecha"])
I = I.drop_duplicates(["t", "fecha", "TRANS_CODE", "usd", "ACCESSION_NUMBER"])
liq = [t for t in close.columns if t in LIQ]

GR = [(b, u, d) for b in (0.1, 0.25, 0.5) for u in (0, 100000, 500000) for d in (91,)] + [(0.25, 100000, 182)]
EST = ["Base"] + [f"b{b}_u{u//1000}k_d{d}" for b, u, d in GR]
filas, prev = [], {k: set() for k in EST}
dif_meses = {k: 0 for k in EST}
for f in FE[12:-1]:
    ok = mom.loc[f].notna() & vsma.loc[f].notna() & fwd.loc[f].notna() & mom.columns.isin(liq)
    if ok.sum() < 60:
        continue
    m, v, r = mom.loc[f][ok], vsma.loc[f][ok], fwd.loc[f][ok]
    pf = pd.Timestamp(f)
    base = (rk(m) + rk(v)) / 2
    fila = {"f": f, "caja": cajaf.loc[f]}
    planes = {"Base": list(base.sort_values(ascending=False).index[:10])}
    for b, u, d in GR:
        ven = I[(I["fecha"] <= pf) & (I["fecha"] > pf - pd.Timedelta(days=d)) & (I["TRANS_CODE"] == "P")].groupby("t")["usd"].sum()
        compra = set(ven[ven >= max(u, 1)].index)
        sc = base + pd.Series({t: b if t in compra else 0 for t in m.index})
        planes[f"b{b}_u{u//1000}k_d{d}"] = list(sc.sort_values(ascending=False).index[:10])
    for s_, top in planes.items():
        hold = {t for t in top if v[t] > 0}
        fila[s_] = (sum(r[t] for t in hold) + (10 - len(hold)) * fila["caja"]) / 10 - COSTO * len(hold ^ prev[s_]) / 10
        if set(top) != set(planes["Base"]):
            dif_meses[s_] += 1
        prev[s_] = hold
    filas.append(fila)
R = pd.DataFrame(filas).set_index("f")
R.index = pd.to_datetime(R.index)
an = lambda s: (1 + s).prod() ** (12 / len(s)) - 1
for s_ in EST:
    o = {"meses_distinta": dif_meses[s_]}
    for a_, b_ in [("2007", "2018"), ("2019", "2026")]:
        sub = R.loc[a_:b_]
        ex = sub[s_] - sub["Base"]
        o[a_] = round((an(sub[s_]) - an(sub["Base"])) * 100, 1)
        o[a_ + "_sinmejor3"] = round((an(sub[s_].where(~ex.isin(ex.nlargest(3)), sub["Base"])) - an(sub["Base"])) * 100, 1)
    print("R|" + s_ + "|" + json.dumps(o), flush=True)
