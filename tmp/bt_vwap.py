import json
import time
import numpy as np
import pandas as pd
import yfinance as yf

COSTO = 0.006
N = 20
d = json.load(open("data/datos.json"))
tickers = [x["ticker"] for x in d["tickers"] if not x.get("sin_datos")]

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


VENT = {1: 21, 3: 63, 6: 126, 12: 252}
TPV = tp * P["Volume"]
RV = {}
for mmm, w in VENT.items():
    num_ = TPV.rolling(w, min_periods=int(w * 0.8)).sum()
    den_ = P["Volume"].rolling(w, min_periods=int(w * 0.8)).sum()
    RV[mmm] = close / (num_ / den_.where(den_ > 0)) - 1


FE = list(fechas.values)
pm = cm.loc[FE]
mom = pm.shift(1) / pm.shift(12) - 1
vsma = pm / ma200.loc[FE] - 1
fwd = pm.shift(-1) / pm - 1
spym = spy.reindex(dias).ffill().loc[FE]
spyf = spym.shift(-1) / spym - 1
cajaf = (irx.reindex(dias).ffill().loc[FE] / 100 / 12)
DV = {mmm: x.loc[FE] for mmm, x in RV.items()}
print("vwap movil cobertura", {mmm: round(float(x.notna().mean().mean()), 3) for mmm, x in DV.items()}, flush=True)


def rk(s):
    return s.rank(pct=True)


filas = []
EST = ["A", "Cma"] + [f"R{mmm}" for mmm in VENT] + [f"M{mmm}" for mmm in VENT]
cart = {k: set() for k in EST}
for k, f in enumerate(FE[12:-1]):
    ok = mom.loc[f].notna() & vsma.loc[f].notna() & fwd.loc[f].notna()
    if ok.sum() < 60:
        continue
    m, v, r = mom.loc[f][ok], vsma.loc[f][ok], fwd.loc[f][ok]
    sc = (rk(m) + rk(v)) / 2
    orden = sc.sort_values(ascending=False)
    top = list(orden.index[:N])
    fila = {"f": f, "ew": r.mean(), "spy": spyf.loc[f], "caja": cajaf.loc[f]}
    plan = {"A": (set(top), N), "Cma": (set(t for t in top if v[t] > 0), N)}
    for mmm in VENT:
        dv = DV[mmm].loc[f]
        arriba = lambda t: pd.isna(dv.get(t)) or dv[t] > 0
        plan[f"R{mmm}"] = (set(t for t in top if arriba(t)), N)
        plan[f"M{mmm}"] = (set(t for t in top if arriba(t) and v[t] > 0), N)
    for s_, (h, n) in plan.items():
        prev = cart[s_]
        rot = (len(h - prev) / n) if prev else len(h) / n
        bruto = (r[list(h)].sum() + (n - len(h)) * fila["caja"]) / n
        fila[s_ + "_g"] = bruto
        fila[s_] = bruto - COSTO * 2 * rot
        fila[s_ + "_n"] = len(h)
        fila[s_ + "_rot"] = rot
        cart[s_] = h
    filas.append(fila)
print("meses validos", len(filas), flush=True)
R = pd.DataFrame(filas).set_index("f")


def st(x, b):
    x, b = x.dropna(), b.reindex(x.index)
    an = lambda s: (1 + s).prod() ** (12 / len(s)) - 1
    c = (1 + x).cumprod()
    ex = x - b
    return {"anual": round(an(x) * 100, 1), "vs_ew": round((an(x) - an(b)) * 100, 1), "vol": round(x.std() * np.sqrt(12) * 100, 1),
            "dd": round(float((c / c.cummax() - 1).min()) * 100), "gana": round(float((ex > 0).mean()) * 100),
            "t": round(float(ex.mean() / ex.std() * np.sqrt(len(ex))), 2)}


out = {"meses": len(R), "desde": str(R.index[0])[:7], "hasta": str(R.index[-1])[:7]}
for a_, b_ in [("2006", "2018"), ("2019", "2026"), ("2006", "2026")]:
    sub = R.loc[a_:b_]
    o = {s: st(sub[s], sub["ew"]) for s in EST + ["spy"]}
    o.update({s + "_g": st(sub[s + "_g"], sub["ew"]) for s in EST})
    o.update({s + "_rot": round(float(sub[s + "_rot"].mean()), 2) for s in EST})
    o.update({s + "_n": round(float(sub[s + "_n"].mean()), 1) for s in EST})
    out[f"{a_}-{b_}"] = o
for kk, vv in out.items():
    print("R|" + kk + "|" + json.dumps(vv), flush=True)
