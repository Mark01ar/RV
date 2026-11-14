import hashlib
import json
import time
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA / "desc_es.json"
WEB = {"User-Agent": "Mozilla/5.0"}


def traducir(texto):
    partes, actual = [], []
    for frase in texto.replace("\n", " ").split(". "):
        if sum(len(x) for x in actual) + len(frase) > 1400 and actual:
            partes.append(". ".join(actual))
            actual = []
        actual.append(frase)
    if actual:
        partes.append(". ".join(actual))
    out = []
    for p in partes:
        r = requests.post("https://translate.googleapis.com/translate_a/single", params={"client": "gtx", "sl": "en", "tl": "es", "dt": "t"},
                          data={"q": p.strip()}, headers=WEB, timeout=30)
        r.raise_for_status()
        out.append("".join(seg[0] for seg in r.json()[0] if seg and seg[0]))
        time.sleep(0.7)
    return ". ".join(x.strip().rstrip(".") for x in out) + "."


def main():
    try:
        cache = json.loads(CACHE.read_text())
    except Exception:
        cache = {}
    nuevos, fallas = 0, 0
    for f in sorted((DATA / "fichas").glob("*.json")):
        t = f.stem
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        en = (d.get("descripcion") or "").strip()
        if not en:
            continue
        h = hashlib.md5(en.encode()).hexdigest()[:10]
        if cache.get(t, {}).get("h") == h:
            continue
        if fallas >= 5:
            break
        try:
            cache[t] = {"h": h, "es": traducir(en)}
            nuevos += 1
        except Exception as e:
            fallas += 1
            print(f"{t}: {str(e)[:100]}")
    CACHE.write_text(json.dumps(dict(sorted(cache.items())), ensure_ascii=False, indent=0))
    print(f"traducidas: {nuevos}; total: {len(cache)}")


if __name__ == "__main__":
    main()
