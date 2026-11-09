import json
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data"
fx = json.loads((D / "factores.json").read_text())
bl = json.loads((D / "bloques.json").read_text())["bloques"]
F = {f["id"]: f for f in fx["factores"]}


def fm(t):
    return f"{t['mov']:+.1f}%" if t else "  -  "


lin = []
for b in bl:
    lin.append(f"\n##### {b['nombre']}" + (f"  [{b['motores_nota']}]" if b.get("motores_nota") else ""))
    for m in b["motores"]:
        res = fx["res"][m]
        lin.append(f"-- si {F[m]['txt']}")
        filas = [("BLOQUE", res["bloques"].get(b["id"]))] + [(e, res["empresas"].get(e)) for e in b["empresas"]]
        for e, z in filas:
            if not z:
                lin.append(f"  {e:7s}   sin historia suficiente")
                continue
            lin.append(f"  {e:7s} {fm(z['a']):>7s} {fm(z['r']):>7s}  {z['e']}")
cnt = {}
for m in fx["res"].values():
    for z in m["empresas"].values():
        cnt[z["e"] or "nada"] = cnt.get(z["e"] or "nada", 0) + 1
lin.append(f"\nconteo todas las combinaciones empresa-factor: {cnt}")
lin.append("\nultimos meses por factor:")
for f in fx["factores"]:
    lin.append(f"  {f['id']:9s} " + "  ".join(f"{u['mes']}:{u['cambio']:+.2f}" for u in f["ultimos"]))
(Path(__file__).resolve().parent / "revision.txt").write_text("\n".join(lin))
print("\n".join(lin))
