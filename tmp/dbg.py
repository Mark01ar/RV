import subprocess, sys, glob, shutil
out = []
for s in ["scripts/traducir.py", [f for f in glob.glob("scripts/*.py") if f.endswith("los.py") and "cic" in f][0], "scripts/hoy.py"]:
    p = subprocess.run([sys.executable, s], capture_output=True, text=True)
    out.append(f"## {s} {p.returncode}\n{p.stdout[-1500:]}\n{p.stderr[-2500:]}")
for f in ["desc_es.json", "ciclos.json", "hoy.json"]:
    shutil.copy(f"data/{f}", f"tmp/{f}")
open("tmp/dbg.txt", "w").write("\n".join(out))
