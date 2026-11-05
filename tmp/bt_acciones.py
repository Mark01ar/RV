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
print("precios", close.shape, flush=True)

ER = {}
for k, t in enumerate(tickers):
    try:
        e = yf.Ticker(t).get_earnings_dates(limit=100)
        if e is not None and len(e):
            idx = pd.to_datetime(e.index)
            idx = idx.tz_convert("America/New_York") if idx.tz is not None else idx
            ef = [(x.tz_localize(None).normalize() + pd.Timedelta(days=1 if x.hour >= 16 else 0)) for x in idx]
            ER[t] = sorted(set(ef))
    except Exception as ex:
        pass
    time.sleep(0.25)
cov = {y: sum(1 for t in ER if any(x.year == y for x in ER[t])) for y in [2006, 2010, 2014, 2018, 2022, 2026]}
print("balances_cobertura", len(ER), cov, flush=True)

dias = close.index
fechas = dias.to_series().groupby(dias.to_period("M")).max()
fechas = fechas[(fechas >= "2005-12-01")]
fechas = fechas.iloc[:-1] if fechas.index[-1] == pd.Period(pd.Timestamp.today(), "M") else fechas
cm = close.reindex(dias).ffill(limit=5)
ma200 = close.rolling(200, min_periods=200).mean()
tp = (P["High"] + P["Low"] + P["Close"]) / 3
cv = P["Volume"].fillna(0).cumsum()
ctpv = (tp * P["Volume"]).fillna(0).cumsum()


def avwap_en(t, f):
    if t not in ER:
        return np.nan
    a = [x for x in ER[t] if x <= f]
    if not a:
        return np.nan
    ancla = a[-1]
    pos_a = dias.searchsorted(ancla)
    pos_f = dias.get_loc(f)
    if pos_f - pos_a < 1 or pos_a == 0:
        return np.nan
    prev = dias[pos_a - 1]
    v = cv.at[f, t] - cv.at[prev, t]
    if not v or v <= 0:
        return np.nan
    vw = (ctpv.at[f, t] - ctpv.at[prev, t]) / v
    return close.at[f, t] / vw - 1


FE = list(fechas.values)
pm = cm.loc[FE]
mom = pm.shift(1) / pm.shift(12) - 1
vsma = pm / ma200.loc[FE] - 1
fwd = pm.shift(-1) / pm - 1
spym = spy.reindex(dias).ffill().loc[FE]
spyf = spym.shift(-1) / spym - 1
cajaf = (irx.reindex(dias).ffill().loc[FE] / 100 / 12)
AV = pd.DataFrame(np.nan, index=FE, columns=close.columns)
for f in FE:
    for t in close.columns:
        if pd.notna(pm.at[f, t]):
            AV.at[f, t] = avwap_en(t, f)
print("avwap calculado", float(AV.notna().mean().mean()), flush=True)


def rk(s):
    return s.rank(pct=True)


filas = []
cart = {"A": set(), "A3": set(), "B": set(), "C": set()}
for k, f in enumerate(FE[12:-1]):
    ok = mom.loc[f].notna() & vsma.loc[f].notna() & fwd.loc[f].notna()
    if ok.sum() < 60:
        continue
    m, v, r, a = mom.loc[f][ok], vsma.loc[f][ok], fwd.loc[f][ok], AV.loc[f][ok]
    sc = (rk(m) + rk(v)) / 2
    scC = pd.concat([rk(m), rk(v), rk(a)], axis=1).mean(axis=1)
    orden = sc.sort_values(ascending=False)
    q = pd.qcut(sc.rank(method="first"), 5, labels=[5, 4, 3, 2, 1])
    nuevo = {"A": set(orden.index[:N]), "C": set(scC.sort_values(ascending=False).index[:N])}
    nuevo["A3"] = set(orden.index[:N]) if (k % 3 == 0 or not cart["A3"]) else {t for t in cart["A3"] if t in r.index} or set(orden.index[:N])
    pool = orden.index[:max(N, int(len(orden) * 0.2))]
    filt = [t for t in pool if v[t] > 0 and (pd.isna(a[t]) or a[t] > 0)][:N]
    nuevo["B"] = set(filt)
    fila = {"f": f, "ew": r.mean(), "spy": spyf.loc[f], "caja": cajaf.loc[f]}
    for qq in [1, 2, 3, 4, 5]:
        fila[f"Q{qq}"] = r[q == qq].mean()
    for s_, c_ in nuevo.items():
        prev = cart[s_]
        rot = (len(c_ - prev) / max(len(c_), 1)) if prev else 1.0
        if prev and not c_:
            rot = 1.0
        bruto = r[list(c_)].mean() if c_ else fila["caja"]
        fila[s_] = bruto - COSTO * 2 * rot
        fila[s_ + "_n"] = len(c_)
        fila[s_ + "_rot"] = rot
        cart[s_] = c_
    filas.append(fila)
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
    o = {s: st(sub[s], sub["ew"]) for s in ["A", "A3", "B", "C", "Q1", "Q2", "Q3", "Q4", "Q5", "spy", "ew"]}
    o["rot_A"] = round(float(sub["A_rot"].mean()), 2)
    o["rot_A3"] = round(float(sub["A3_rot"].mean()), 2)
    o["rot_B"] = round(float(sub["B_rot"].mean()), 2)
    o["n_B"] = round(float(sub["B_n"].mean()), 1)
    o["meses_B_caja"] = int((sub["B_n"] == 0).sum())
    out[f"{a_}-{b_}"] = o
for kk, vv in out.items():
    print("R|" + kk + "|" + json.dumps(vv), flush=True)
