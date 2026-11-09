import json
import math
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent.parent / "data"
FASES = ["Crecimiento sin inflación", "Recalentamiento", "Estanflación", "Desinflación"]


def sd_mensual(f):
    s = (f.get("precio_bloque") or {}).get("serie") or {}
    if not s.get("f") or not s.get("spy"):
        return None
    df = pd.DataFrame({"p": s["p"], "spy": s["spy"]}, index=pd.to_datetime(s["f"])).dropna()
    m = df.resample("ME").last().pct_change().dropna()
    if len(m) < 18:
        return None
    return float((m["p"] - m["spy"]).std())


def main():
    em = json.loads((D / "empresas.json").read_text())["empresas"]
    out = []
    for x in em:
        p = D / "fichas" / f"{x['ticker']}.json"
        if not p.exists():
            continue
        f = json.loads(p.read_text())
        rg = (f.get("sensibilidades") or {}).get("regimen") or {}
        sd = sd_mensual(f)
        if not rg or not sd:
            continue
        r = {}
        for fz in FASES:
            d = rg.get(fz)
            o = [v for k, v in rg.items() if k != fz]
            if not d or d["meses"] < 12 or not o:
                continue
            no = sum(v["meses"] for v in o)
            om = sum(v["exceso_anual"] * v["meses"] for v in o) / no
            dif = d["exceso_anual"] - om
            se = sd * 12 * math.sqrt(1 / d["meses"] + 1 / no)
            r[fz] = {"dif": round(dif, 4), "t": round(dif / se, 2), "meses": d["meses"], "vs_spy": round(d["exceso_anual"], 4)}
        if r:
            out.append({"b": x["byma"], "ticker": x["ticker"], "nombre": x["nombre"], "r": r})
    (D / "escenarios.json").write_text(json.dumps({"fases": FASES, "empresas": out}, ensure_ascii=False, separators=(",", ":")))
    print("escenarios", len(out))


if __name__ == "__main__":
    main()
