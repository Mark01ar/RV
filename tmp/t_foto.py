import sys
import time
sys.path.insert(0, "scripts")
import fundamentales as F
for t in ["AAPL", "MRNA", "JPM", "SHEL", "TSM"]:
    t0 = time.time()
    d = F.bajar(t, t == "JPM", t in ("AAPL", "MRNA", "JPM"))
    f = d.get("foto") or {}
    print("T|", t, round(time.time() - t0, 1), "s", len(f), "datos", flush=True)
    grupos = {}
    for k in f:
        g = k.split("_")[0] if "_" in k else "info"
        grupos[g] = grupos.get(g, 0) + 1
    print("G|", t, grupos, flush=True)
    if t == "AAPL":
        for k, v in f.items():
            print("K|", k, v, flush=True)
