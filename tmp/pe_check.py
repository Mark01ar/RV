import json, glob, time
import yfinance as yf
import pandas as pd
out = []
for f in sorted(glob.glob("data/fichas/*.json")):
    d = json.load(open(f))
    t = d.get("ticker")
    try:
        tk = yf.Ticker(t)
        i = tk.info
        q = tk.quarterly_income_stmt
        nq = None
        if q is not None and "Net Income" in q.index:
            s = q.loc["Net Income"].dropna().sort_index()
            nq = [(str(c.date()), float(v)) for c, v in s.tail(5).items()]
        out.append({"t": t, "b": d.get("byma"), "nuestro": d["val"].get("pe"), "fwd_nuestro": d["val"].get("pe_fwd"),
                    "y_trailing": i.get("trailingPE"), "y_fwd": i.get("forwardPE"), "y_eps": i.get("trailingEps"),
                    "y_mcap": i.get("marketCap"), "mcap": d.get("mcap"), "cur": i.get("financialCurrency"),
                    "q_nuestro": [(x.get("trimestre"), x.get("neto")) for x in d.get("trimestral", [])[-5:]], "q_yahoo": nq})
    except Exception as e:
        out.append({"t": t, "err": str(e)})
    time.sleep(0.3)
json.dump(out, open("pe_check.json", "w"))
print("ok", len(out))
