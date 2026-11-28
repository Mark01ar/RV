import yfinance as yf
import pandas as pd
print("yfinance", yf.__version__)
for nombre, f in [
    ("download 2y lote", lambda: yf.download(["SNDK", "MU", "AAPL"], period="2y", interval="1d", auto_adjust=True, progress=False, group_by="column")["Close"]["SNDK"]),
    ("download 10y lote", lambda: yf.download(["SNDK", "MU", "AAPL"], period="10y", interval="1d", auto_adjust=True, progress=False, group_by="column")["Close"]["SNDK"]),
    ("download 10y solo", lambda: yf.download("SNDK", period="10y", interval="1d", auto_adjust=True, progress=False)["Close"].squeeze()),
    ("history 2y", lambda: yf.Ticker("SNDK").history(period="2y", auto_adjust=True)["Close"]),
    ("history start", lambda: yf.Ticker("SNDK").history(start="2025-01-01", auto_adjust=True)["Close"]),
    ("download start lote", lambda: yf.download(["SNDK", "MU"], start="2025-01-01", interval="1d", auto_adjust=True, progress=False, group_by="column")["Close"]["SNDK"]),
]:
    try:
        s = f()
        print(nombre, "| filas", len(s), "| ultimo valido", s.last_valid_index(), "| cola", [None if pd.isna(x) else round(float(x), 2) for x in s.tail(6)])
    except Exception as e:
        print(nombre, "ERROR", e)
