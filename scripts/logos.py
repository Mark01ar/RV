import io
import json
import os
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOGOS = ROOT / "data" / "logos"
DOM = {"SKHY": "skhynix.com", "IREN": "iren.com", "HUT": "hut8.com", "GT": "goodyear.com", "MO": "altria.com", "OKLO": "oklo.com",
       "TEN": "tenaris.com", "CAT": "caterpillar.com", "JMIA": "jumia.com", "IBM": "ibm.com", "UBER": "uber.com", "V": "visa.com",
       "MRVL": "marvell.com", "NKE": "nike.com", "NIO": "nio.com", "ANF": "abercrombie.com", "ADGO": "adecoagro.com", "RGTI": "rigetti.com",
       "DISN": "disney.com", "HIMS": "hims.com", "LMT": "lockheedmartin.com", "BB": "blackberry.com", "UNH": "unitedhealthgroup.com",
       "CEG": "constellationenergy.com", "BA": "boeing.com", "VRTX": "vrtx.com", "NOW": "servicenow.com", "SATL": "satellogic.com"}
TICK = {"SKHY": "000660.KS", "DISN": "DIS", "BA.C": "BAC"}


def bajar(url):
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if r.ok and r.headers.get("content-type", "").startswith("image") and len(r.content) > 300:
            im = Image.open(io.BytesIO(r.content))
            im.load()
            return im
    except Exception:
        return None


def mejor(b):
    t = TICK.get(b, b)
    cands = []
    for url in [f"https://financialmodelingprep.com/image-stock/{t}.png", f"https://img.logo.dev/ticker/{t}?size=256&format=png",
                f"https://www.google.com/s2/favicons?domain={DOM.get(b, '')}&sz=256", f"https://icon.horse/icon/{DOM.get(b, '')}"]:
        if "domain=&" in url or url.endswith("/icon/"):
            continue
        im = bajar(url)
        if im is not None:
            cands.append((min(im.size), url, im))
    cands = [c for c in cands if c[0] >= 48]
    return max(cands, key=lambda c: c[0]) if cands else None


def luz(im):
    im = im.convert("RGBA")
    im.thumbnail((96, 96))
    px = [p for p in im.getdata() if p[3] > 60]
    if not px:
        return 1.0, 0.0
    lum = sum((0.299 * r + 0.587 * g + 0.114 * b) / 255 for r, g, b, a in px) / len(px)
    return lum, len(px) / (im.width * im.height)


def main():
    rehacer = [x.strip() for x in os.environ.get("LOGOS", "").split(",") if x.strip()]
    for b in rehacer:
        r = mejor(b)
        if r:
            r[2].convert("RGBA").save(LOGOS / f"{b}.png")
            print(b, r[0], r[1])
        else:
            print(b, "sin alternativa")
    meta = {}
    for p in sorted(LOGOS.glob("*.png")):
        try:
            lum, cob = luz(Image.open(p))
        except Exception:
            continue
        if lum > 0.82:
            meta[p.stem] = "oscuro"
    (ROOT / "data" / "logos_meta.json").write_text(json.dumps(meta, indent=1))
    print("fondo oscuro:", sorted(meta))


if __name__ == "__main__":
    main()
