import json
import sys
import time
import pandas as pd
sys.path.insert(0, "scripts")
import actualizar as A

u = pd.read_csv("data/universo_rv.csv")
ccl = json.load(open("data/datos.json")).get("ccl")
vivo = A.byma_vivo()
syms = set(u["byma"])
base = {s for s in vivo}
nuevos = sorted(s for s in base if s not in syms and not ((s.endswith("D") or s.endswith("C")) and s[:-1] in base))
print("ccl", ccl, "vivo", len(vivo), "no_en_csv", len(nuevos), flush=True)
print("NUEVOS|" + ",".join(nuevos), flush=True)
filas = []
for s in list(u["byma"]) + nuevos:
    h = None
    for intento in range(4):
        try:
            h = A.byma_historico(s)
        except Exception:
            h = None
        if h is not None and len(h):
            break
        time.sleep(3 * (intento + 1))
    if h is None or not len(h):
        filas.append({"byma": s, "m120": None, "m20": None, "dias": 0})
        continue
    h = h[h["fecha"] >= h["fecha"].max() - pd.Timedelta(days=185)]
    filas.append({"byma": s, "m120": float(h["monto"].median()) / ccl, "m20": float(h["monto"].tail(20).median()) / ccl,
                  "dias_con_operacion": int((h["v"].astype(float) > 0).sum()), "ultimo": str(h["fecha"].max())[:10]})
    time.sleep(0.8)
df = pd.DataFrame(filas).merge(u, on="byma", how="left")
df.to_csv("tmp/top100.csv", index=False)
for _, r in df.sort_values("m120", ascending=False).iterrows():
    print("V|" + "|".join(str(r.get(c)) for c in ["byma", "ticker_us", "nombre", "estado", "motivo", "m120", "m20", "dias_con_operacion", "ultimo"]), flush=True)
