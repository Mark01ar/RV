import json
import os
import sys
import time
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, "scripts")
import macro as M

KEY = os.environ["FRED_API_KEY"]
REV = ["CFNAIMA3", "ICSA", "INDPRO", "PAYEMS", "CPIAUCSL", "CPILFESL", "PCEPILFE", "NFCI", "GDPNOW"]
NOREV = ["BAA10Y", "T5YIE"]
LAGS = {"ICSA": 12, "NFCI": 12, "GDPNOW": 40}


def vint(sid):
    rows, off = [], 0
    while True:
        for _ in range(4):
            r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                             params={"series_id": sid, "api_key": KEY, "file_type": "json",
                                     "observation_start": "1995-01-01", "realtime_start": "1776-07-04",
                                     "realtime_end": "9999-12-31", "limit": 100000, "offset": off}, timeout=180)
            if r.ok:
                break
            print("reintento", sid, r.status_code, r.text[:200], flush=True)
            time.sleep(5)
        r.raise_for_status()
        o = r.json()["observations"]
        rows += o
        if len(o) < 100000:
            break
        off += 100000
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["rs"] = pd.to_datetime(df["realtime_start"])
    df["re"] = pd.to_datetime(df["realtime_end"].replace("9999-12-31", "2262-01-01"))
    df["d"] = pd.to_datetime(df["date"])
    return df[["d", "rs", "re", "value"]]


V = {}
for sid in REV:
    try:
        V[sid] = vint(sid)
    except Exception as e:
        print("sin", sid, e, flush=True)
FIRST = {sid: v.sort_values("rs").groupby("d").first() for sid, v in V.items()}
cob = {sid: [v["rs"].min().strftime("%Y-%m"), v["d"].min().strftime("%Y-%m"), len(v)] for sid, v in V.items()}
print("cobertura", cob, flush=True)


def asof(sid, T):
    v = V[sid]
    s = v[(v["rs"] <= T) & (v["re"] >= T)].groupby("d")["value"].last()
    lag = pd.Timedelta(days=LAGS.get(sid, 62))
    fi = FIRST[sid]
    extra = fi[(fi["rs"] > T) & (fi.index + lag <= T)]["value"]
    return pd.concat([s, extra[~extra.index.isin(s.index)]]).sort_index()


NR = {sid: M.fred(sid)[0] for sid in NOREV}
y = M.yahoo(["XLY", "XLP", "HG=F", "GC=F", "CL=F", "TIP", "IEF", "SPY", "^IRX"])


def extras(f, usar):
    out = []
    if "NFCI" in usar and "NFCI" in f and len(f["NFCI"]) > 20:
        m = M.mes(f["NFCI"])
        out.append({"id": "NFCI", "grupo": "crecimiento", "tipo": "mercado", "imp": (-m.diff(3)).dropna()})
    if "GDPNOW" in usar and "GDPNOW" in f and len(f["GDPNOW"]) > 4:
        q = f["GDPNOW"].copy()
        m = q.resample("ME").last().ffill()
        out.append({"id": "GDPNOW", "grupo": "crecimiento", "tipo": "dato", "imp": m.diff(3).dropna()})
    return out


def g_i(T, usar):
    f = {sid: asof(sid, T) for sid in V}
    f.update({sid: s.loc[:T] for sid, s in NR.items()})
    inds = M.indicadores(f, y.loc[:T]) + extras(f, usar)
    idx = pd.date_range("1999-01-31", T, freq="ME")
    Z = pd.DataFrame({x["id"]: M.zexp(x["imp"]).reindex(idx).ffill(limit=4) for x in inds})

    def grupo(g, t):
        return Z[[x["id"] for x in inds if x["grupo"] == g and x["tipo"] == t]].mean(axis=1)

    out = []
    for g in ["crecimiento", "inflacion"]:
        d, m = grupo(g, "dato"), grupo(g, "mercado")
        c = (M.PESO_DATO * d + (1 - M.PESO_DATO) * m).fillna(d).fillna(m)
        out.append(float(M.zexp(c).iloc[-1]))
    return out


fin = pd.Timestamp.today().to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(1)


def tramo_stats(bt, a, b):
    x = bt[(bt.index >= a) & (bt.index <= b)]
    def an(v):
        return float((1 + v).prod() ** (12 / len(v)) - 1)
    def dd(v):
        c = (1 + v).cumprod()
        return float((c / c.cummax() - 1).min())
    return {"regla": round(an(x["regla"]), 3), "spy": round(an(x["spy"]), 3),
            "dd_regla": round(dd(x["regla"]), 3), "dd_spy": round(dd(x["spy"]), 3),
            "fuera": round(float(1 - x["dentro"].mean()), 3)}


def backtest(rt):
    spy = y["SPY"].dropna()
    pm = spy.resample("ME").last()
    ma = pm.rolling(10).mean()
    irx = y["^IRX"].dropna().resample("ME").last()
    caja = (irx / 100 / 12).shift(1).reindex(pm.index).ffill().fillna(0)
    ret = pm.pct_change()
    df = pd.DataFrame({"p": pm, "ma": ma}).join(rt[["qc"]], how="inner")
    df["rojo"] = df["qc"].isin(["EST", "DES"]) & (df["p"] < df["ma"])
    filas, prev = [], True
    for d in df.index[1:]:
        if d > fin:
            break
        dentro = not df["rojo"].shift(1).get(d)
        r = ret[d] if dentro else caja[d]
        if dentro != prev:
            r -= 0.005
        prev = dentro
        filas.append({"f": d, "spy": ret[d], "regla": r, "dentro": dentro})
    return pd.DataFrame(filas).set_index("f"), df


res = {}
for nombre, usar in [("actual", set()), ("mas_nfci", {"NFCI"}), ("mas_gdpnow", {"GDPNOW"}), ("ambos", {"NFCI", "GDPNOW"})]:
    filas = [{"f": T, **dict(zip(["g", "i"], g_i(T, usar)))} for T in pd.date_range("2004-12-31", fin, freq="ME")]
    rt = pd.DataFrame(filas).set_index("f")
    rt["q"] = [M.cuadrante(a, b) for a, b in zip(rt["g"], rt["i"])]
    rt["qc"] = M.confirmar(rt)
    bt, df = backtest(rt)
    rojos = df["rojo"]
    n_alarmas = int(((rojos) & (~rojos.shift(1, fill_value=False))).loc["2005":].sum())
    res[nombre] = {"2005-hoy": tramo_stats(bt, "2005-01-01", "2100-01-01"),
                   "2005-2018": tramo_stats(bt, "2005-01-01", "2018-12-31"),
                   "2012-hoy": tramo_stats(bt, "2012-01-01", "2100-01-01"),
                   "2019-hoy": tramo_stats(bt, "2019-01-01", "2100-01-01"),
                   "alarmas": n_alarmas,
                   "hoy": [rt["q"].iloc[-1], rt["qc"].iloc[-1], round(rt["g"].iloc[-1], 2)]}
    print(nombre, json.dumps(res[nombre]), flush=True)
print("RESULTADO " + json.dumps({"cobertura": cob, "res": res}))
