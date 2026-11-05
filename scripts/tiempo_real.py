import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import macro as M

KEY = os.environ.get("FRED_API_KEY", "").strip()
REV = ["CFNAIMA3", "ICSA", "INDPRO", "PAYEMS", "CPIAUCSL", "CPILFESL", "PCEPILFE"]
NOREV = ["BAA10Y", "T5YIE"]
ARCHIVO = M.SALIDA


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


def main():
    if not KEY:
        sys.exit("falta FRED_API_KEY")
    V = {sid: vint(sid) for sid in REV}
    first = {sid: v.sort_values("rs").groupby("d").first() for sid, v in V.items()}
    cobertura = {sid: v["rs"].min().strftime("%Y-%m") for sid, v in V.items()}

    def asof(sid, T):
        v = V[sid]
        s = v[(v["rs"] <= T) & (v["re"] >= T)].groupby("d")["value"].last()
        lag = pd.Timedelta(days=12 if sid == "ICSA" else 62)
        fi = first[sid]
        extra = fi[(fi["rs"] > T) & (fi.index + lag <= T)]["value"]
        return pd.concat([s, extra[~extra.index.isin(s.index)]]).sort_index()

    nr = {sid: M.fred(sid)[0] for sid in NOREV}
    y = M.yahoo(["XLY", "XLP", "HG=F", "GC=F", "CL=F", "TIP", "IEF", "SPY", "^IRX"])

    def g_i(T):
        f = {sid: asof(sid, T) for sid in REV}
        f.update({sid: s.loc[:T] for sid, s in nr.items()})
        inds = M.indicadores(f, y.loc[:T])
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

    salida = json.loads(ARCHIVO.read_text())
    actual = pd.Timestamp.today().to_period("M").to_timestamp("M")
    filas = [{"f": T, **dict(zip(["g", "i"], g_i(T)))}
             for T in pd.date_range("2004-12-31", actual - pd.offsets.MonthEnd(1), freq="ME")]
    filas.append({"f": actual, "g": salida["g"], "i": salida["i"]})
    rt = pd.DataFrame(filas).set_index("f")
    rt["q"] = [M.cuadrante(a, b) for a, b in zip(rt["g"], rt["i"])]
    rt["qc"] = M.confirmar(rt)

    q = rt["qc"].tolist()
    dur = 1
    while dur < len(q) and q[-1 - dur] == q[-1]:
        dur += 1
    sem = M.semaforo(y, rt)
    sem["base"] = "tiempo_real"
    sem["cobertura"] = cobertura
    salida["semaforo"] = sem
    salida["actual"] = q[-1]
    salida["duracion"] = dur
    salida["historial_rt"] = [{"f": d.strftime("%Y-%m"), "q": a, "qc": b} for d, a, b in zip(rt.index, rt["q"], rt["qc"])]
    ARCHIVO.write_text(json.dumps(salida, ensure_ascii=False, separators=(",", ":")))
    print(f"tiempo real: regimen {q[-1]} ({dur} meses), semaforo {sem['estado']}, "
          f"regla {sem['resultados']['regla']['anual']:.3f} vs spy {sem['resultados']['spy']['anual']:.3f}")


if __name__ == "__main__":
    main()
