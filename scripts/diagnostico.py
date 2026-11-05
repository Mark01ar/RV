import json
import requests
import yfinance as yf
from pathlib import Path

H = {"User-Agent": "Mozilla/5.0 (monitor-rv)"}
T = ["MU", "PEP", "SNDK", "TSLA", "NVDA", "AAPL", "KO", "AAL"]
out = {}


def get(u):
    try:
        r = requests.get(u, headers=H, timeout=30)
        return r.status_code, r.json()
    except Exception as e:
        return None, str(e)


s, vivo = get("https://data912.com/live/arg_cedears")
out["vivo"] = [x for x in vivo if isinstance(x, dict) and x.get("symbol", "").rstrip("DC") in T] if isinstance(vivo, list) else vivo
s, usa = get("https://data912.com/live/usa_stocks")
out["usa"] = [x for x in usa if isinstance(x, dict) and x.get("symbol") in T] if isinstance(usa, list) else usa
out["hist"] = {}
for t in T:
    s, h = get(f"https://data912.com/historical/cedears/{t}")
    out["hist"][t] = {"status": s, "ultimos": h[-6:] if isinstance(h, list) else h}
d = yf.download(T, period="10d", interval="1d", auto_adjust=False, progress=False)
out["yf"] = {t: {str(i.date()): float(v) for i, v in d["Close"][t].dropna().items()} for t in T}
Path("data/diag.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
