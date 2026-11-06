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


MJ = json.load(open("data/macro.json"))
RT = {h["f"]: h["qc"] for h in MJ["historial_rt"]}
REV = {h["f"]: h["q"] for h in MJ["historial"]}
mk = yf.download(["XLY", "XLP", "HG=F", "GC=F"], start="1999-01-01", auto_adjust=True, progress=False)["Close"]
mk.index = pd.to_datetime(mk.index).tz_localize(None)
mkm = mk.resample("ME").last()


def zexp(x):
    return (x - x.expanding(36).mean()) / x.expanding(36).std()


g_mk = pd.concat([zexp(np.log(mkm["XLY"] / mkm["XLP"]).diff(3)), zexp(np.log(mkm["HG=F"] / mkm["GC=F"]).diff(3))], axis=1).mean(axis=1)
MKT = {d.strftime("%Y-%m"): bool(v < 0) for d, v in g_mk.dropna().items()}
print("mercado meses", len(MKT), sum(MKT.values()), flush=True)
dr = close.pct_change()
sr = spy.reindex(dias).ffill().pct_change()
mx = dr.rolling(252, min_periods=200).mean()
my = sr.rolling(252, min_periods=200).mean()
cxy = dr.mul(sr, axis=0).rolling(252, min_periods=200).mean() - mx.mul(my, axis=0)
vy = sr.rolling(252, min_periods=200).var(ddof=0)
BETA = cxy.div(vy, axis=0).loc[FE]
print("beta cobertura", round(float(BETA.notna().mean().mean()), 3), flush=True)
POOL = 30
filas = []
EST = ["Base", "RB", "RBperf", "RBmkt", "Bajo", "Alto"]
cart = {k: set() for k in EST}
for k, f in enumerate(FE[12:-1]):
    ok = mom.loc[f].notna() & vsma.loc[f].notna() & fwd.loc[f].notna() & BETA.loc[f].notna() & mom.columns.isin(list(LIQ))
    if ok.sum() < 60:
        continue
    m, v, r, b = mom.loc[f][ok], vsma.loc[f][ok], fwd.loc[f][ok], BETA.loc[f][ok]
    sc = ((rk(m) + rk(v)) / 2).sort_values(ascending=False)
    ym = pd.Timestamp(f).strftime("%Y-%m")
    ym_sig = (pd.Timestamp(f) + pd.offsets.MonthEnd(1)).strftime("%Y-%m")
    frena = RT.get(ym) in ("EST", "DES")
    frena_perf = REV.get(ym_sig) in ("EST", "DES")
    frena_mkt = MKT.get(ym, False)
    pool = [t for t in sc.index if v[t] > 0][:POOL]
    fila = {"f": f, "spy": spyf.loc[f], "caja": cajaf.loc[f], "frena": frena, "frena_perf": frena_perf, "frena_mkt": frena_mkt}

    def por_beta(obj):
        if not pool:
            return set()
        ps = sc[pool].rank(pct=True)
        cerca = (-(b[pool] - obj).abs()).rank(pct=True)
        return set((ps + cerca).sort_values(ascending=False).index[:10])

    top10 = list(sc.index[:10])
    plan = {"Base": set(t for t in top10 if v[t] > 0),
            "RB": por_beta(0.7 if frena else 1.3),
            "RBperf": por_beta(0.7 if frena_perf else 1.3),
            "RBmkt": por_beta(0.7 if frena_mkt else 1.3),
            "Bajo": por_beta(0.7), "Alto": por_beta(1.3)}
    for s_, h in plan.items():
        prev = cart[s_]
        rot = (len(h - prev) / 10) if prev else len(h) / 10
        fila[s_] = (r[list(h)].sum() + (10 - len(h)) * fila["caja"]) / 10 - COSTO * 2 * rot
        fila[s_ + "_beta"] = float(b[list(h)].mean()) if h else 0.0
        cart[s_] = h
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


out = {"meses": len(R), "desde": str(R.index[0])[:7], "hasta": str(R.index[-1])[:7], "meses_frena": int(R["frena"].sum()),
       "meses_frena_perf": int(R["frena_perf"].sum()), "meses_frena_mkt": int(R["frena_mkt"].sum()),
       "spy_frena_perf": round(float(R.loc[R["frena_perf"], "spy"].mean() * 1200), 1), "spy_crece_perf": round(float(R.loc[~R["frena_perf"], "spy"].mean() * 1200), 1),
       "spy_frena_mkt": round(float(R.loc[R["frena_mkt"], "spy"].mean() * 1200), 1), "spy_crece_mkt": round(float(R.loc[~R["frena_mkt"], "spy"].mean() * 1200), 1),
       "spy_frena_rt": round(float(R.loc[R["frena"], "spy"].mean() * 1200), 1), "spy_crece_rt": round(float(R.loc[~R["frena"], "spy"].mean() * 1200), 1)}
for a_, b_ in [("2007", "2018"), ("2019", "2026"), ("2007", "2026")]:
    sub = R.loc[a_:b_]
    o = {}
    for s_ in EST:
        x = st(sub[s_], sub["spy"])
        x["ret_vol"] = round(x["anual"] / x["vol"], 2)
        x["beta_prom"] = round(float(sub[s_ + "_beta"].mean()), 2)
        x["beta_frena"] = round(float(sub.loc[sub["frena"], s_ + "_beta"].mean()), 2) if sub["frena"].any() else None
        x["beta_crece"] = round(float(sub.loc[~sub["frena"], s_ + "_beta"].mean()), 2)
        x["anual_frena"] = round(float(sub.loc[sub["frena"], s_].mean() * 12 * 100), 1) if sub["frena"].any() else None
        x["anual_crece"] = round(float(sub.loc[~sub["frena"], s_].mean() * 12 * 100), 1)
        o[s_] = x
    out[f"{a_}-{b_}"] = o
for kk, vv in out.items():
    print("R|" + kk + "|" + json.dumps(vv), flush=True)
