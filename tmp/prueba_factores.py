import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path(__file__).resolve().parent
EMP = ["XOM", "CVX", "SHEL", "PBR", "GPRK", "TEN"]
FAC = {"petroleo": ("CL=F", 0.10), "gas": ("NG=F", 0.10), "real": ("BRL=X", 0.05)}
TRAMOS = {"2005-2014": ("2005-01", "2014-12"), "2015-2026": ("2015-01", None)}
ROT = 999
rng = np.random.default_rng(7)


def bajar():
    tk = EMP + ["SPY"] + [v[0] for v in FAC.values()]
    px = yf.download(tk, start="2004-11-01", auto_adjust=True, progress=False, threads=True)["Close"]
    px = px.ffill().resample("ME").last()
    px.index = px.index.to_period("M")
    r = np.log(px).diff()
    return r.iloc[1:-1]


def coef(y, m, f):
    X = np.column_stack([np.ones(len(y)), m, f])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b[2]


def prueba(y, m, f):
    d = pd.concat([y, m, f], axis=1).dropna()
    if len(d) < 36:
        return None
    yv, mv, fv = d.iloc[:, 0].values, d.iloc[:, 1].values, d.iloc[:, 2].values
    c = coef(yv, mv, fv)
    n = len(fv)
    sh = rng.integers(12, n - 12, ROT)
    nul = np.array([coef(yv, mv, np.roll(fv, k)) for k in sh])
    p = (1 + np.sum(np.abs(nul) >= abs(c))) / (ROT + 1)
    return {"c": float(c), "p": float(p), "meses": int(n), "desde": str(d.index[0])}


def tramo(s, a, b):
    s = s[s.index >= pd.Period(a, "M")]
    return s if b is None else s[s.index <= pd.Period(b, "M")]


def main():
    r = bajar()
    cob = {t: str(r[t].dropna().index[0]) if t in r and r[t].notna().any() else None for t in EMP}
    ret = {t: r[t] for t in EMP if t in r}
    blq = pd.concat([v for v in ret.values()], axis=1).mean(axis=1, skipna=True)
    ret["BLOQUE"] = blq
    res = {}
    for nom, (tf, mov) in FAC.items():
        f = r[tf]
        res[nom] = {}
        for t, y in ret.items():
            fila = {}
            for tn, (a, b) in TRAMOS.items():
                z = prueba(tramo(y, a, b), tramo(r["SPY"], a, b), tramo(f, a, b))
                if z:
                    z["mov"] = round((np.exp(z["c"] * np.log(1 + mov)) - 1) * 100, 2)
                fila[tn] = z
            t1, t2 = fila["2005-2014"], fila["2015-2026"]
            fila["clara"] = bool(t1 and t2 and t1["p"] < 0.05 and t2["p"] < 0.05 and np.sign(t1["c"]) == np.sign(t2["c"]))
            res[nom][t] = fila
    (OUT / "resultado.json").write_text(json.dumps({"cobertura": cob, "res": res}, ensure_ascii=False, indent=1))
    lin = [f"cobertura {cob}"]
    for nom, (tf, mov) in FAC.items():
        lin.append(f"\n== {nom} ({tf}) sube {int(mov*100)}% ==")
        for t, fila in res[nom].items():
            p = []
            for tn in TRAMOS:
                z = fila[tn]
                p.append(f"{tn}: " + (f"{z['mov']:+.2f}% p={z['p']:.3f} n={z['meses']}" if z else "sin datos"))
            lin.append(f"{t:7s} | " + " | ".join(p) + (" | CLARA" if fila["clara"] else ""))
    (OUT / "resultado.txt").write_text("\n".join(lin))
    print("\n".join(lin))


main()
