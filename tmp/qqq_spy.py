import json
import numpy as np
import pandas as pd
import yfinance as yf

px = yf.download(["SPY", "QQQ", "^TNX", "^IRX"], start="1998-01-01", auto_adjust=True, progress=False)["Close"]
px.index = pd.to_datetime(px.index).tz_localize(None)
px = px.dropna(subset=["SPY", "QQQ"])
px["^TNX"] = px["^TNX"].ffill()
spy, qqq, tnx = px["SPY"], px["QQQ"], px["^TNX"]
rel = qqq / spy
relma = rel.rolling(200).mean()
ma200 = spy.rolling(200).mean()
m = json.load(open("data/macro.json"))
rt = pd.DataFrame(m["historial_rt"])
qc = pd.Series(rt["qc"].values, index=pd.to_datetime(rt["f"]) + pd.offsets.MonthEnd(0))
viernes = pd.date_range("2000-01-07", px.index[-1], freq="14D")
cortes = sorted({px.index[px.index <= v][-1] for v in viernes if (px.index <= v).any()})
rows = []
for d in cortes:
    i = px.index.get_loc(d)
    if i < 260:
        continue
    q = qc[qc.index <= d]
    sem = None
    if len(q):
        fr = q.iloc[-1] in ("EST", "DES")
        ba = spy.loc[d] < ma200.loc[d]
        sem = "rojo" if fr and ba else ("amarillo" if fr or ba else "verde")
    dt63 = tnx.loc[d] - tnx.iloc[i - 63]
    rows.append({"d": d, "rel_up": bool(rel.loc[d] > relma.loc[d]), "tasa_sube": bool(dt63 > 0.5), "sem": sem})
S = pd.DataFrame(rows).set_index("d")
P = px.loc[S.index, ["SPY", "QQQ"]]
fw = P.shift(-1) / P - 1
S = S.join(fw).dropna(subset=["SPY", "QQQ"])
W = {}
for k in (0, 25, 50, 75, 100):
    W[f"Fijo{k}"] = pd.Series(k / 100, index=S.index)
W["FuerzaRel"] = S["rel_up"].map({True: 0.75, False: 0.25})
W["Tasas"] = S["tasa_sube"].map({True: 0.25, False: 0.75})
W["Semaforo"] = S["sem"].map({"verde": 0.75, "amarillo": 0.5, "rojo": 0.25})
W["Rel+Tasas"] = 0.5 + S["rel_up"].map({True: 0.25, False: -0.25}) + S["tasa_sube"].map({True: -0.25, False: 0.25})
W["Rel+Tasas"] = W["Rel+Tasas"].clip(0, 1)
COSTO = 0.005
R = pd.DataFrame(index=S.index)
for k, w in W.items():
    w = w.astype(float)
    gir = w.diff().abs().fillna(0)
    R[k] = w * S["QQQ"] + (1 - w) * S["SPY"] - gir * COSTO
    R[k + "_w"] = w
an = lambda s: (1 + s).prod() ** (365.25 / ((s.index[-1] - s.index[0]).days + 14)) - 1
print("pct_tasa_sube", round(S["tasa_sube"].mean() * 100), "pct_rel_up", round(S["rel_up"].mean() * 100), flush=True)
for a_, b_ in [("2000", "2004"), ("2005", "2018"), ("2019", "2026"), ("2000", "2026"), ("2005", "2026")]:
    sub = R.loc[a_:b_]
    o = {}
    for k in W:
        x = sub[k].dropna()
        if len(x) < 20 or sub[k + "_w"].isna().mean() > 0.2:
            continue
        c = (1 + x).cumprod()
        y = (1 + x).groupby(x.index.year).prod() - 1
        o[k] = {"anual": round(an(x) * 100, 1), "dd": round(float((c / c.cummax() - 1).min()) * 100),
                "vol": round(x.std() * np.sqrt(26) * 100, 1), "peor_anio": round(float(y.min()) * 100),
                "peso_qqq_prom": round(float(sub[k + "_w"].mean()) * 100)}
    print("S|" + a_ + "-" + b_ + "|" + json.dumps(o), flush=True)
eps = {"2000-02": ("2000-03-24", "2002-10-09"), "2008": ("2007-10-09", "2009-03-09"), "2020": ("2020-02-19", "2020-03-23"),
       "2022": ("2022-01-03", "2022-10-12"), "2025": ("2025-02-19", "2025-04-08")}
o = {}
for e, (a, b) in eps.items():
    o[e] = {k: round(float((1 + R.loc[a:b, k]).prod() - 1) * 100) for k in ["Fijo0", "Fijo50", "Fijo100", "FuerzaRel", "Tasas", "Semaforo", "Rel+Tasas"] if R.loc[a:b, k].notna().all()}
print("E|" + json.dumps(o), flush=True)
print("HOY|" + str(S.index[-1])[:10] + "|" + json.dumps({k: float(W[k].iloc[-1]) for k in ["FuerzaRel", "Tasas", "Semaforo", "Rel+Tasas"]}), flush=True)
