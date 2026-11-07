"""At pixel ratios below 1: where do the drawn cells and the filter's pixels land?"""
import io, random, sys
import numpy as np
from PIL import Image
from pathlib import Path
from playwright.sync_api import sync_playwright
random.seed(7)
pat = np.array([[random.random() < 0.35 for _ in range(14)] for _ in range(14)])
# identity filter that also marks (0,0) red and (13,13) blue
dbg = ('<filter id="dbg" x="0" y="0" width="14" height="14" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" color-interpolation-filters="sRGB">'
       '<feComponentTransfer in="SourceGraphic" result="t"><feFuncR type="discrete" tableValues="0 1"/><feFuncG type="discrete" tableValues="0 1"/><feFuncB type="discrete" tableValues="0 1"/></feComponentTransfer>'
       '<feFlood flood-color="#f00" x="0" y="0" width="1" height="1" result="red"/><feFlood flood-color="#00f" x="13" y="13" width="1" height="1" result="blue"/>'
       '<feMerge><feMergeNode in="t"/><feMergeNode in="red"/><feMergeNode in="blue"/></feMerge></filter>')
mode = sys.argv[1]
with sync_playwright() as p:
    b = p.chromium.launch()
    for dpr in (0.75, 0.9, 1.0):
        pg = b.new_page(viewport={"width": 800, "height": 500}, device_scale_factor=dpr)
        pg.goto(Path("site/test.html").resolve().as_uri())
        css = "body{background:#404040 !important} .mirror rect.m { transition: none !important; } .out { filter: none !important; }"
        if mode == "nolayer": css += ".out{will-change:auto !important}"
        if mode == "unit": css += ".net{transform:none !important}"
        pg.add_style_tag(content=css)
        if mode != "net":
            pg.evaluate("""f => { document.querySelector('.out defs').insertAdjacentHTML('beforeend', f); document.querySelector('.mirror').setAttribute('filter', 'url(#dbg)'); }""", dbg)
        pg.evaluate("""cells => { const r = document.querySelectorAll('.mirror rect.m'); cells.forEach((v, i) => r[i].style.fill = v ? '#fff' : '#000'); }""", pat.ravel().tolist())
        info = pg.evaluate("() => [devicePixelRatio, getComputedStyle(document.querySelector('.net')).transform, JSON.stringify(document.querySelector('.out').getBoundingClientRect())]")
        box = pg.locator(".out").bounding_box()
        im = np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("RGB")).astype(int)
        ox, oy = int(round(box["x"] * dpr)), int(round(box["y"] * dpr))
        reg = im[oy - 2:oy + 20, ox - 2:ox + 20]
        print(dpr, info, "screenshot", im.shape)
        for row in reg[:, :, :]:
            print("".join("R" if (c[0] > 200 and c[1] < 80) else "B" if (c[2] > 200 and c[0] < 80) else "#" if c.min() > 200 else "." if c.max() < 30 else "-" if abs(c[0]-64) < 6 else "?" for c in row))
        pg.close()
    b.close()
