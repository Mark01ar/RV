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
EST = ["Base", "Base+Ins", "InsMA", "Ins", "Univ"]
filas, prev = [], {k: set() for k in EST}
n_ins = []
for f in FE[12:-1]:
    ok = mom.loc[f].notna() & vsma.loc[f].notna() & fwd.loc[f].notna() & mom.columns.isin(liq)
    if ok.sum() < 60:
        continue
    m, v, r = mom.loc[f][ok], vsma.loc[f][ok], fwd.loc[f][ok]
    pf = pd.Timestamp(f)
    ven = I[(I["fecha"] <= pf) & (I["fecha"] > pf - pd.Timedelta(days=91))]
    compra = ven[ven["TRANS_CODE"] == "P"].groupby("t")["usd"].sum()
    compra = set(compra[compra >= 100000].index) & set(m.index)
    n_ins.append(len(compra))
    base = (rk(m) + rk(v)) / 2
    bono = base + pd.Series({t: 0.25 if t in compra else 0 for t in m.index})
    fila = {"f": f, "spy": spyf.loc[f], "caja": cajaf.loc[f], "Univ": float(r.mean()), "Univ_exp": 1}
    planes = {"Base": list(base.sort_values(ascending=False).index[:10]), "Base+Ins": list(bono.sort_values(ascending=False).index[:10])}
    for s_, top in planes.items():
        hold = {t for t in top if v[t] > 0}
        fila[s_] = (sum(r[t] for t in hold) + (10 - len(hold)) * fila["caja"]) / 10 - COSTO * len(hold ^ prev[s_]) / 10
        fila[s_ + "_exp"] = len(hold) / 10
        prev[s_] = hold
    for s_, hold in {"Ins": compra, "InsMA": {t for t in compra if v[t] > 0}}.items():
        if hold:
            fila[s_] = float(np.mean([r[t] for t in hold])) - COSTO * len(hold ^ prev[s_]) / max(len(hold), 1)
        else:
            fila[s_] = fila["caja"]
        fila[s_ + "_exp"] = 1 if hold else 0
        prev[s_] = set(hold)
    filas.append(fila)
print("acciones con compras por mes", np.percentile(n_ins, [10, 50, 90]), flush=True)
R = pd.DataFrame(filas).set_index("f")
R.index = pd.to_datetime(R.index)
an = lambda s: (1 + s).prod() ** (12 / len(s)) - 1
out = {"meses": len(R), "desde": str(R.index[0])[:7], "hasta": str(R.index[-1])[:7]}
for a_, b_ in [("2007", "2018"), ("2019", "2026"), ("2007", "2026")]:
    sub = R.loc[a_:b_]
    o = {}
    yb = (1 + sub["Base"]).groupby(sub.index.year).prod() - 1
    yu = (1 + sub["Univ"]).groupby(sub.index.year).prod() - 1
    for s_ in EST:
        x = sub[s_]
        c = (1 + x).cumprod()
        y = (1 + x).groupby(x.index.year).prod() - 1
        o[s_] = {"anual": round(an(x) * 100, 1), "vs_base": round((an(x) - an(sub["Base"])) * 100, 1),
                 "vs_univ": round((an(x) - an(sub["Univ"])) * 100, 1), "vol": round(x.std() * np.sqrt(12) * 100, 1),
                 "dd": round(float((c / c.cummax() - 1).min()) * 100), "anios_vs_base": f"{int((y > yb).sum())}/{len(y)}",
                 "anios_vs_univ": f"{int((y > yu).sum())}/{len(y)}", "expo": round(float(sub[s_ + "_exp"].mean()) * 100)}
    o["SPY"] = round(an(sub["spy"]) * 100, 1)
    out[f"{a_}-{b_}"] = o
for kk, vv in out.items():
    print("R|" + kk + "|" + json.dumps(vv), flush=True)
