import re
import sys
import json
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
SALIDA = DATA / "tecnico.json"
CSV = DATA / "tecnico.csv"
VENTANAS = {"1m": 21, "3m": 63, "12m": 252}


def r(x, d=4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return round(x, d)


def num(s):
    s = (s or "").strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def leer_csv():
    if not CSV.exists():
        return []
    filas = []
    lineas = CSV.read_text(encoding="utf-8").strip().splitlines()
    if not lineas:
        return []
    cab = [c.strip().lower() for c in lineas[0].split(",")]
    for l in lineas[1:]:
        if not l.strip():
            continue
        v = [c.strip() for c in l.split(",")]
        v += [""] * (len(cab) - len(v))
        d = dict(zip(cab, v))
        tk = d.get("ticker", "").upper()
        tp = d.get("tipo", "").upper()
        if tp in ("NEUTRAL", "NEUTRO"):
            tp = "N"
        if not tk or tp not in ("OW", "UW", "N"):
            continue
        extra = [c.strip() for c in l.split(",")[len(cab):]]
        nota = ", ".join([d.get("nota", "")] + extra).strip(", ") if "nota" in cab else ""
        filas.append({
            "ticker": tk,
            "tipo": tp,
            "desde": d.get("desde", ""),
            "plazo": d.get("plazo", ""),
            "objetivo": num(d.get("objetivo")),
            "objetivo2": num(d.get("objetivo2")),
            "stop": num(d.get("stop")),
            "nota": nota,
        })
    return filas


def universo(ideas):
    tks = []
    for nombre in ("datos.json", "empresas.json"):
        p = DATA / nombre
        if not p.exists():
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        lista = j.get("tickers") or j.get("empresas") or []
        tks += [x.get("ticker") for x in lista if x.get("ticker")]
    tks += [x["ticker"] for x in ideas]
    return sorted(set(tks))


def descargar(tickers):
    import yfinance as yf
    campos = {"Close": [], "High": [], "Low": [], "Volume": []}
    for i in range(0, len(tickers), 40):
        lote = tickers[i:i + 40]
        df = None
        for intento in range(3):
            try:
                df = yf.download(lote, period="15mo", interval="1d", auto_adjust=True,
                                 progress=False, threads=True, group_by="column")
                break
            except Exception as e:
                print(f"reintento lote {i}: {e}", file=sys.stderr)
                time.sleep(10)
        if df is None or df.empty:
            continue
        for c in campos:
            if c not in df.columns.get_level_values(0):
                continue
            x = df[c]
            if isinstance(x, pd.Series):
                x = x.to_frame(lote[0])
            campos[c].append(x)
    return {c: (pd.concat(v, axis=1) if v else pd.DataFrame()) for c, v in campos.items()}


def plazo_vence(desde, plazo):
    p = (plazo or "").strip().lower()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p):
        return p
    m = re.fullmatch(r"(\d+)\s*(d|dias|días|s|sem|semanas|m|mes|meses|a|año|años|anio|anios)", p)
    if not m or not desde:
        return None
    try:
        d0 = dt.date.fromisoformat(desde)
    except ValueError:
        return None
    n, u = int(m.group(1)), m.group(2)
    if u.startswith("d"):
        d1 = d0 + dt.timedelta(days=n)
    elif u.startswith("s"):
        d1 = d0 + dt.timedelta(weeks=n)
    elif u.startswith("m"):
        y, mo = divmod(d0.month - 1 + n, 12)
        d1 = dt.date(d0.year + y, mo + 1, min(d0.day, 28))
    else:
        d1 = dt.date(d0.year + n, d0.month, min(d0.day, 28))
    return d1.isoformat()


def indicadores(c, h, l, v):
    c = c.dropna()
    if len(c) < 30:
        return None
    h = h.reindex(c.index).fillna(c)
    l = l.reindex(c.index).fillna(c)
    v = v.reindex(c.index)
    p = float(c.iloc[-1])
    o = {"fecha": c.index[-1].strftime("%Y-%m-%d"), "precio": r(p, 4)}
    mas = {}
    for n in (20, 50, 200):
        if len(c) >= n:
            ma = c.rolling(n).mean()
            mas[n] = ma
            o[f"ma{n}"] = r(ma.iloc[-1], 4)
            o[f"d{n}"] = r(p / ma.iloc[-1] - 1)
    if 200 in mas and len(c) >= 220:
        o["pend200"] = r(mas[200].iloc[-1] / mas[200].iloc[-21] - 1)
    if 50 in mas and 200 in mas:
        dif = (mas[50] - mas[200]).dropna()
        if len(dif):
            signo = np.sign(dif)
            o["cruce"] = "dorado" if signo.iloc[-1] > 0 else "muerte"
            cambios = np.where(signo.values[1:] != signo.values[:-1])[0]
            if len(cambios):
                k = cambios[-1] + 1
                o["cruce_fecha"] = dif.index[k].strftime("%Y-%m-%d")
                o["cruce_ruedas"] = int(len(dif) - k - 1)
    a50, a200 = o.get("ma50"), o.get("ma200")
    if a50 and a200:
        if p > a50 > a200:
            o["tendencia"] = "Alcista"
        elif p < a50 < a200:
            o["tendencia"] = "Bajista"
        else:
            o["tendencia"] = "Mixta"
    niv = {}
    for k, n in VENTANAS.items():
        if len(c) < min(n, 21):
            continue
        mx, mn = float(h.iloc[-n:].max()), float(l.iloc[-n:].min())
        niv[k] = {"max": r(mx, 4), "min": r(mn, 4), "a_max": r(p / mx - 1), "a_min": r(p / mn - 1)}
    o["niveles"] = niv
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1).dropna()
    if len(tr) >= 14:
        atr = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
        o["atr"] = r(atr, 4)
        o["atr_pct"] = r(atr / p)
    vv = v.dropna()
    if len(vv) >= 21 and vv.iloc[-21:-1].mean() > 0:
        base = float(vv.iloc[-21:-1].mean())
        o["vol_rel"] = r(float(vv.iloc[-1]) / base, 2)
        o["vol_5d"] = r(float(vv.iloc[-5:].mean()) / base, 2)
    return o


def eventos(c, h, l, v):
    c = c.dropna()
    if len(c) < 60:
        return []
    h = h.reindex(c.index).fillna(c)
    l = l.reindex(c.index).fillna(c)
    v = v.reindex(c.index)
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().shift(1)
    vbase = v.rolling(20).mean().shift(1)
    max12, min12 = h.rolling(252, min_periods=120).max().shift(1), l.rolling(252, min_periods=120).min().shift(1)
    max3, min3 = h.rolling(63).max().shift(1), l.rolling(63).min().shift(1)
    out = []
    n = len(c)
    previo = {}
    lado_p = lado_m = 0
    for k in range(max(1, n - 126), n):
        f = c.index[k].strftime("%Y-%m-%d")
        p, q = c.iloc[k], c.iloc[k - 1]
        hoy = {}
        reciente = k >= n - 63
        if np.isfinite(ma200.iloc[k]):
            rp = p / ma200.iloc[k] - 1
            nuevo = 1 if rp > 0.01 else -1 if rp < -0.01 else lado_p
            if lado_p and nuevo != lado_p and reciente:
                out.append([f, "pos" if nuevo > 0 else "neg", f"Cruzó hacia {'arriba' if nuevo > 0 else 'abajo'} su MA200"])
            lado_p = nuevo
            rm = ma50.iloc[k] / ma200.iloc[k] - 1
            nuevo = 1 if rm > 0.005 else -1 if rm < -0.005 else lado_m
            if lado_m and nuevo != lado_m and reciente:
                out.append([f, "pos" if nuevo > 0 else "neg", f"La MA50 cruzó {'arriba' if nuevo > 0 else 'abajo'} de la MA200"])
            lado_m = nuevo
        if not reciente:
            continue
        if np.isfinite(max12.iloc[k]) and p > max12.iloc[k]:
            hoy["max"] = "Nuevo máximo de 12 meses"
        elif np.isfinite(max3.iloc[k]) and p > max3.iloc[k]:
            hoy["max"] = "Rompió el máximo de 3 meses"
        if np.isfinite(min12.iloc[k]) and p < min12.iloc[k]:
            hoy["min"] = "Nuevo mínimo de 12 meses"
        elif np.isfinite(min3.iloc[k]) and p < min3.iloc[k]:
            hoy["min"] = "Perforó el mínimo de 3 meses"
        for key, tipo in (("max", "pos"), ("min", "neg")):
            if key in hoy:
                if k - previo.get(key, -99) > 15:
                    out.append([f, tipo, hoy[key]])
                previo[key] = k
        mv = p / q - 1
        if np.isfinite(atr.iloc[k]) and atr.iloc[k] > 0 and abs(p - q) >= 2.5 * atr.iloc[k]:
            out.append([f, "pos" if mv > 0 else "neg", f"{'Subió' if mv > 0 else 'Bajó'} {abs(mv) * 100:.1f}% en el día".replace(".", ",") + f" ({abs(p - q) / atr.iloc[k]:.1f} ATR)".replace(".", ",")])
        if np.isfinite(vbase.iloc[k]) and vbase.iloc[k] > 0 and v.iloc[k] >= 3 * vbase.iloc[k]:
            out.append([f, "neu", f"Volumen {v.iloc[k] / vbase.iloc[k]:.1f} veces el promedio".replace(".", ",")])
    return out[-15:]


def balances():
    res = {}
    for q in (DATA / "fichas").glob("*.json"):
        try:
            j = json.loads(q.read_text(encoding="utf-8"))
        except Exception:
            continue
        f = (j.get("ultimo_balance") or {}).get("fecha")
        if f and j.get("ticker"):
            res[j["ticker"]] = f
    return res


def evaluar_idea(x, c, h, l, ind):
    x = dict(x)
    if x["tipo"] == "N":
        for k in ("objetivo", "objetivo2", "stop"):
            x[k] = None
    o = dict(x)
    o["vence"] = plazo_vence(x["desde"], x["plazo"])
    c = c.dropna()
    if not len(c) or not ind:
        o["estado"] = "Sin precio"
        return o
    p = float(c.iloc[-1])
    o["precio"] = r(p, 4)
    ow = x["tipo"] == "OW"
    try:
        d0 = pd.Timestamp(x["desde"])
    except Exception:
        d0 = None
    if d0 is not None:
        previos = c[c.index <= d0]
        if len(previos):
            o["precio_desde"] = r(previos.iloc[-1], 4)
            o["ret"] = r(p / float(previos.iloc[-1]) - 1)
    for k in ("objetivo", "objetivo2", "stop"):
        if x.get(k):
            o["a_" + k] = r(x[k] / p - 1)
    if x.get("stop") and ind.get("atr"):
        o["stop_atr"] = r(abs(p - x["stop"]) / ind["atr"], 2)
    if x.get("objetivo") and x.get("stop"):
        gana = (x["objetivo"] - p) if ow else (p - x["objetivo"])
        pierde = (p - x["stop"]) if ow else (x["stop"] - p)
        if pierde > 0:
            o["rb"] = r(gana / pierde, 2)
    estado, cuando = ("Vigente" if x["tipo"] != "N" else "Neutral"), None
    if d0 is not None and x["tipo"] != "N":
        hh = h.reindex(c.index)[c.index > d0].fillna(c)
        ll = l.reindex(c.index)[c.index > d0].fillna(c)
        eventos = []
        for k, nom in (("objetivo2", "Objetivo 2 alcanzado"), ("objetivo", "Objetivo alcanzado")):
            t = x.get(k)
            if t:
                toca = hh[hh >= t] if ow else ll[ll <= t]
                if len(toca):
                    eventos.append((toca.index[0], nom, k))
        s = x.get("stop")
        if s:
            toca = ll[ll <= s] if ow else hh[hh >= s]
            if len(toca):
                eventos.append((toca.index[0], "Stop tocado", "stop"))
        if eventos:
            stops = [e for e in eventos if e[2] == "stop"]
            objs = sorted([e for e in eventos if e[2] != "stop"], key=lambda e: e[0])
            primero = min(eventos, key=lambda e: e[0])
            if primero[2] == "stop":
                estado, cuando = "Stop tocado", primero[0]
            else:
                o2 = [e for e in objs if e[2] == "objetivo2"]
                e_ = o2[0] if o2 else objs[0]
                estado, cuando = e_[1], e_[0]
                if stops and stops[0][0] > e_[0]:
                    o["stop_despues"] = stops[0][0].strftime("%Y-%m-%d")
    if estado in ("Vigente", "Neutral") and o["vence"] and o["vence"] < dt.date.today().isoformat():
        estado = "Plazo vencido"
    o["estado"] = estado
    if cuando is not None:
        o["estado_fecha"] = cuando.strftime("%Y-%m-%d")
    return o


def main():
    ideas = leer_csv()
    tks = universo(ideas)
    print(f"{len(tks)} tickers, {len(ideas)} ideas técnicas")
    px = descargar(tks)
    C, H, L, V = px["Close"], px["High"], px["Low"], px["Volume"]
    vacio = pd.Series(dtype=float)
    col = lambda df, t: df[t] if t in df.columns else vacio
    out = {}
    bal = balances()
    for t in tks:
        try:
            ind = indicadores(col(C, t), col(H, t), col(L, t), col(V, t))
            if ind:
                ind["eventos"] = eventos(col(C, t), col(H, t), col(L, t), col(V, t))
                if t in bal:
                    ind["balance"] = bal[t]
        except Exception as e:
            print(f"{t}: {e}", file=sys.stderr)
            ind = None
        if ind:
            out[t] = ind
    ev = []
    for x in ideas:
        t = x["ticker"]
        ev.append(evaluar_idea(x, col(C, t), col(H, t), col(L, t), out.get(t)))
    if len(out) < len(tks) * 0.5:
        print(f"Solo {len(out)} de {len(tks)} con datos: no se guarda", file=sys.stderr)
        sys.exit(1)
    res = {"actualizado": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "tickers": out, "ideas": ev}
    SALIDA.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"OK: {len(out)} tickers")


if __name__ == "__main__":
    main()
