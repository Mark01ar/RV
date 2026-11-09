import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import motores_empresa as ME

OUT = Path(__file__).resolve().parent
COD = {"CSI": "Crecimiento sin inflación", "REC": "Recalentamiento", "EST": "Estanflación", "DES": "Desinflación"}
ROT = 999
rng = np.random.default_rng(11)


def tramo(s, a, b):
    s = s[s.index >= pd.Period(a, "M")]
    return s if b is None else s[s.index <= pd.Period(b, "M")]


def prueba(x, reg, cod):
    d = pd.concat([x, reg], axis=1).dropna()
    n = len(d)
    if n < 36:
        return None
    xv = d.iloc[:, 0].values.astype(float)
    rv = d.iloc[:, 1].values
    m = rv == cod
    if m.sum() < 8 or (~m).sum() < 8:
        return None
    dif = xv[m].mean() - xv[~m].mean()
    k = rng.integers(6, n - 6, ROT)
    nul = []
    for j in k:
        mm = np.roll(m, j)
        if mm.sum() and (~mm).sum():
            nul.append(xv[mm].mean() - xv[~mm].mean())
    nul = np.array(nul)
    p = (1 + np.sum(np.abs(nul) >= abs(dif))) / (len(nul) + 1)
    return {"dif": float(dif), "p": float(p), "meses": int(m.sum())}


def etiqueta(a, r):
    sa = a is not None and a["p"] < 0.05
    sr = r is not None and r["p"] < 0.05
    if sa and sr:
        return "fuerte" if np.sign(a["dif"]) == np.sign(r["dif"]) else "vuelta"
    contra = sa and r is not None and np.sign(a["dif"]) != np.sign(r["dif"])
    if r is not None and r["p"] < 0.01 and r["meses"] >= 12 and not contra:
        return "reciente"
    if r is not None and r["p"] < 0.10 and not contra:
        return "indicio"
    return ""


def main():
    m = json.loads((ME.D / "macro.json").read_text())
    reg = pd.Series({x["f"]: x["qc"] for x in m["historial_rt"]})
    reg.index = pd.PeriodIndex(reg.index, freq="M")
    r, fs = ME.bajar([])
    fs["mercado"] = r["SPY"]
    nombres = {**{k: v[3] for k, v in ME.FAC.items()}, "mercado": "Mercado (SPY)"}
    lin, res, cnt = [], {}, {}
    for t, (a, b) in ME.TRAMOS.items():
        rr = tramo(reg, a, b)
        lin.append(f"meses por régimen {t}: " + ", ".join(f"{c}={int((rr == c).sum())}" for c in COD))
    for h in (1, 3):
        lin.append(f"\n======== régimen al cierre del mes → cambio del factor en los {h} meses siguientes ========")
        for k, f in fs.items():
            f = f.dropna()
            fut = sum(f.shift(-i) for i in range(1, h + 1)) if h > 1 else f.shift(-1)
            sd = float(f.std()) * np.sqrt(h)
            filas = []
            for cod in COD:
                z = {t: prueba(tramo(fut, a, bb), tramo(reg, a, bb), cod) for t, (a, bb) in ME.TRAMOS.items()}
                e = etiqueta(z["a"], z["r"])
                res.setdefault(f"h{h}", {}).setdefault(k, {})[cod] = {"a": z["a"], "r": z["r"], "e": e, "sd": sd}
                cnt[e or "nada"] = cnt.get(e or "nada", 0) + 1
                if e:
                    fm = lambda q: f"{q['dif'] / sd:+.2f}σ" if q else "  -  "
                    filas.append(f"    {COD[cod]:26s} {fm(z['a']):>7s} {fm(z['r']):>7s}  {e}")
            if filas:
                lin.append(f"  {nombres[k]}")
                lin.extend(filas)
    lin.append(f"\nconteo: {cnt}")
    lin.append("σ = movimientos típicos del factor en ese plazo (ej. +0,30σ en petróleo a 1 mes ≈ +3,4% más que en los demás meses)")
    (OUT / "regimen.json").write_text(json.dumps(res, ensure_ascii=False))
    (OUT / "regimen.txt").write_text("\n".join(lin))
    print("\n".join(lin))


main()
