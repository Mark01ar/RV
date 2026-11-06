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


dr = close.pct_change()
VOL = dr.rolling(126, min_periods=100).std() * np.sqrt(252)
OBJ = 0.20
filas = []
EST = ["Base", "IV", "VT", "IVVT"]
wprev = {k: {} for k in EST}
for k, f in enumerate(FE[12:-1]):
    ok = mom.loc[f].notna() & vsma.loc[f].notna() & fwd.loc[f].notna() & mom.columns.isin(list(LIQ))
    if ok.sum() < 60:
        continue
    m, v, r = mom.loc[f][ok], vsma.loc[f][ok], fwd.loc[f][ok]
    sc = ((rk(m) + rk(v)) / 2).sort_values(ascending=False)
    h = [t for t in sc.index[:10] if v[t] > 0]
    vol = VOL.loc[f, h].fillna(VOL.loc[f, h].median() if h else 0.3)
    pf = pd.Timestamp(f)
    hist = dr.loc[:pf].iloc[-126:][h]
    vport = float(hist.mean(axis=1).std() * np.sqrt(252)) if h else 0.0
    eq = {t: 0.1 for t in h}
    inv = (1 / vol) / (1 / vol).sum() * (len(h) / 10) if h else pd.Series(dtype=float)
    iv = {t: float(inv[t]) for t in h}
    vport_iv = float((hist * pd.Series(iv)).sum(axis=1).std() * np.sqrt(252) / max(sum(iv.values()), 1e-9)) if h else 0.0
    esc_eq = min(1.0, OBJ / vport) if vport > 0 else 1.0
    esc_iv = min(1.0, OBJ / vport_iv) if vport_iv > 0 else 1.0
    plan = {"Base": eq, "IV": iv, "VT": {t: w * esc_eq for t, w in eq.items()}, "IVVT": {t: w * esc_iv for t, w in iv.items()}}
    fila = {"f": f, "spy": spyf.loc[f], "caja": cajaf.loc[f]}
    for s_, w in plan.items():
        inv_tot = sum(w.values())
        bruto = sum(w[t] * r[t] for t in w) + (1 - inv_tot) * fila["caja"]
        prev = wprev[s_]
        giro = sum(abs(w.get(t, 0) - prev.get(t, 0)) for t in set(w) | set(prev))
        fila[s_] = bruto - COSTO * giro
        fila[s_ + "_exp"] = inv_tot
        wprev[s_] = w
    filas.append(fila)
R = pd.DataFrame(filas).set_index("f")
R.index = pd.to_datetime(R.index)
def st(x, b):
    x, b = x.dropna(), b.reindex(x.index)
    an = lambda s: (1 + s).prod() ** (12 / len(s)) - 1
    c = (1 + x).cumprod()
    cb = (1 + b).cumprod()
    ex = x - b
    anios = (1 + x).groupby(x.index.year).prod() - 1
    anios_b = (1 + b).groupby(b.index.year).prod() - 1
    return {"anual": round(an(x) * 100, 1), "spy_anual": round(an(b) * 100, 1), "dif": round((an(x) - an(b)) * 100, 1),
            "vol": round(x.std() * np.sqrt(12) * 100, 1), "dd": round(float((c / c.cummax() - 1).min()) * 100),
            "dd_spy": round(float((cb / cb.cummax() - 1).min()) * 100),
            "peor_anio": round(float(anios.min()) * 100), "anios_gana": f"{int((anios > anios_b).sum())}/{len(anios)}",
            "meses_gana": round(float((ex > 0).mean()) * 100), "x_final": round(float(c.iloc[-1]), 2), "x_spy": round(float(cb.iloc[-1]), 2)}


out = {"meses": len(R), "desde": str(R.index[0])[:7], "hasta": str(R.index[-1])[:7]}
eps = {"2008": ("2007-10", "2009-03"), "2009reb": ("2009-03", "2009-09"), "2020": ("2020-02", "2020-03"), "2022": ("2022-01", "2022-09")}
for a_, b_ in [("2007", "2018"), ("2019", "2026"), ("2007", "2026")]:
    sub = R.loc[a_:b_]
    o = {}
    for s_ in EST:
        x = st(sub[s_], sub["spy"])
        x["ret_vol"] = round(x["anual"] / x["vol"], 2)
        x["expo"] = round(float(sub[s_ + "_exp"].mean()) * 100)
        x["eps"] = {e: round(float((1 + sub[s_].loc[a:b]).prod() - 1) * 100) for e, (a, b) in eps.items() if len(sub[s_].loc[a:b])}
        o[s_] = x
    out[f"{a_}-{b_}"] = o
for kk, vv in out.items():
    print("R|" + kk + "|" + json.dumps(vv), flush=True)
