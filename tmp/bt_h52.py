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

sec = json.load(open("data/sectores.json"))
liq = [t for t in close.columns if t in LIQ]
ind = {t: (sec.get(t) or {}).get("industria") for t in liq}
cnt = pd.Series(ind).value_counts()
grupo = {t: (ind[t] if ind[t] and cnt.get(ind[t], 0) >= 3 else (sec.get(t) or {}).get("sector") or "?") for t in liq}
print("grupos", len(set(grupo.values())), flush=True)
h52 = pm / close.rolling(252, min_periods=200).max().loc[FE]
def rk(s):
    return s.rank(pct=True)

EST = ["Base", "H52", "H52solo", "B+H52", "B+Grupo", "GrupoTop", "Univ"]
filas, prev = [], {k: set() for k in EST}
for f in FE[12:-1]:
    ok = mom.loc[f].notna() & vsma.loc[f].notna() & fwd.loc[f].notna() & h52.loc[f].notna() & mom.columns.isin(liq)
    if ok.sum() < 60:
        continue
    m, v, r, h = mom.loc[f][ok], vsma.loc[f][ok], fwd.loc[f][ok], h52.loc[f][ok]
    g = pd.Series({t: grupo[t] for t in m.index})
    gm = m.groupby(g).mean()
    gmom = g.map(gm)
    gtop = set(gm[gm >= gm.median()].index)
    sc = {"Base": (rk(m) + rk(v)) / 2, "H52": (rk(m) + rk(h)) / 2, "H52solo": rk(h),
          "B+H52": (rk(m) + rk(v) + rk(h)) / 3, "B+Grupo": (rk(m) + rk(v) + rk(gmom)) / 3,
          "GrupoTop": ((rk(m) + rk(v)) / 2)[g.isin(gtop)]}
    fila = {"f": f, "spy": spyf.loc[f], "caja": cajaf.loc[f], "Univ": float(r.mean())}
    for s_, s in sc.items():
        top = list(s.sort_values(ascending=False).index[:10])
        hold = {t for t in top if v[t] > 0}
        bruto = (sum(r[t] for t in hold) + (10 - len(hold)) * fila["caja"]) / 10
        giro = len(hold ^ prev[s_]) / 10
        fila[s_] = bruto - COSTO * giro
        fila[s_ + "_exp"] = len(hold) / 10
        prev[s_] = hold
    fila["Univ_exp"] = 1
    filas.append(fila)
R = pd.DataFrame(filas).set_index("f")
R.index = pd.to_datetime(R.index)
an = lambda s: (1 + s).prod() ** (12 / len(s)) - 1
out = {"meses": len(R), "desde": str(R.index[0])[:7], "hasta": str(R.index[-1])[:7]}
for a_, b_ in [("2007", "2018"), ("2019", "2026"), ("2007", "2026")]:
    sub = R.loc[a_:b_]
    o = {}
    yb = (1 + sub["Base"]).groupby(sub.index.year).prod() - 1
    for s_ in EST:
        x = sub[s_]
        c = (1 + x).cumprod()
        y = (1 + x).groupby(x.index.year).prod() - 1
        o[s_] = {"anual": round(an(x) * 100, 1), "vs_base": round((an(x) - an(sub["Base"])) * 100, 1),
                 "vs_univ": round((an(x) - an(sub["Univ"])) * 100, 1), "vol": round(x.std() * np.sqrt(12) * 100, 1),
                 "dd": round(float((c / c.cummax() - 1).min()) * 100), "anios_vs_base": f"{int((y > yb).sum())}/{len(y)}",
                 "meses_vs_base": round(float((x > sub["Base"]).mean()) * 100), "expo": round(float(sub[s_ + "_exp"].mean()) * 100)}
    o["SPY"] = round(an(sub["spy"]) * 100, 1)
    out[f"{a_}-{b_}"] = o
for kk, vv in out.items():
    print("R|" + kk + "|" + json.dumps(vv), flush=True)
