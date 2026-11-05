import json
import yfinance as yf

out = {}
for t in ["AAPL", "JPM", "TSM", "XOM", "NVO", "SNDK"]:
    k = yf.Ticker(t)
    d = {}
    try:
        info = k.info or {}
        d["info_keys"] = sorted(info.keys())
        d["info"] = {x: info.get(x) for x in ["forwardPE", "trailingPE", "enterpriseToEbitda", "returnOnAssets", "returnOnEquity",
                                              "operatingMargins", "grossMargins", "freeCashflow", "marketCap", "totalDebt", "totalCash",
                                              "ebitda", "revenueGrowth", "earningsGrowth", "recommendationMean", "numberOfAnalystOpinions",
                                              "targetMeanPrice", "currentPrice", "mostRecentQuarter", "earningsTimestamp",
                                              "earningsTimestampStart", "financialCurrency", "currency", "trailingPegRatio", "pegRatio"]}
    except Exception as e:
        d["info_err"] = str(e)
    for attr in ["eps_trend", "eps_revisions", "earnings_estimate", "growth_estimates"]:
        try:
            v = getattr(k, attr)
            d[attr] = v.reset_index().astype(str).to_dict("records") if v is not None else None
        except Exception as e:
            d[attr] = f"ERR {e}"
    try:
        e = k.get_earnings_dates(limit=6)
        d["earnings_dates"] = e.reset_index().astype(str).to_dict("records") if e is not None else None
    except Exception as e:
        d["earnings_dates"] = f"ERR {e}"
    try:
        d["calendar"] = {a: str(b) for a, b in (k.calendar or {}).items()}
    except Exception as e:
        d["calendar"] = f"ERR {e}"
    try:
        q = k.quarterly_income_stmt
        d["q_income_rows"] = list(q.index)[:60]
        d["q_income_cols"] = [str(c) for c in q.columns]
    except Exception as e:
        d["q_income_rows"] = f"ERR {e}"
    try:
        a = k.income_stmt
        d["a_income_cols"] = [str(c) for c in a.columns]
    except Exception as e:
        d["a_income_cols"] = f"ERR {e}"
    out[t] = d
print("RESULTADO " + json.dumps(out, default=str))
