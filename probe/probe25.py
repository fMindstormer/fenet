import io, random
import numpy as np
from PIL import Image
from pathlib import Path
from playwright.sync_api import sync_playwright
random.seed(7)
pat = np.array([[random.random() < 0.35 for _ in range(14)] for _ in range(14)])
dbg = ('<filter id="dbg" x="0" y="0" width="14" height="14" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" color-interpolation-filters="sRGB">'
       '<feComponentTransfer in="SourceGraphic" result="t"><feFuncR type="discrete" tableValues="0 1"/><feFuncG type="discrete" tableValues="0 1"/><feFuncB type="discrete" tableValues="0 1"/></feComponentTransfer>'
       '<feFlood flood-color="#f00" x="0" y="0" width="1" height="1" result="red"/><feFlood flood-color="#00f" x="13" y="13" width="1" height="1" result="blue"/>'
       '<feMerge><feMergeNode in="t"/><feMergeNode in="red"/><feMergeNode in="blue"/></feMerge></filter>')
with sync_playwright() as p:
    b = p.chromium.launch()
    for dpr in (1.25, 1.5, 2):
        pg = b.new_page(viewport={"width": 800, "height": 500}, device_scale_factor=dpr)
        pg.goto(Path("site/test.html").resolve().as_uri())
        pg.add_style_tag(content="body{background:#404040 !important} .mirror rect.m { transition: none !important; } .out { filter: none !important; }")
        pg.evaluate("""f => { document.querySelector('.out defs').insertAdjacentHTML('beforeend', f); document.querySelector('.mirror').setAttribute('filter', 'url(#dbg)'); }""", dbg)
        pg.evaluate("""cells => { const r = document.querySelectorAll('.mirror rect.m'); cells.forEach((v, i) => r[i].style.fill = v ? '#fff' : '#000'); }""", pat.ravel().tolist())
        box = pg.locator(".out").bounding_box()
        im = np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("RGB")).astype(int)
        ox, oy = int(box["x"] * dpr), int(box["y"] * dpr)
        reg = im[oy - 2:oy + 17, ox - 2:ox + 17]
        red = np.argwhere((reg[:, :, 0] > 200) & (reg[:, :, 1] < 60)); blue = np.argwhere((reg[:, :, 2] > 200) & (reg[:, :, 0] < 60))
        white = (reg.min(axis=2) > 200)
        # best alignment of the pattern inside the region
        best = min(((white[y:y + 14, x:x + 14] != pat).sum(), x - 2, y - 2) for x in range(0, 5) for y in range(0, 5))
        grey = int(((reg.max(axis=2) > 30) & (reg.min(axis=2) < 220) & ~((reg[:, :, 0] > 200) & (reg[:, :, 1] < 60)) & ~((reg[:, :, 2] > 200) & (reg[:, :, 0] < 60))).sum())
        print(f"dpr {dpr} origin ({box['x']*dpr:.2f},{box['y']*dpr:.2f}): red marker at {red.tolist()[:2]} (-2 offset), blue at {blue.tolist()[:2]}; pattern best offset {best[1:]} with {best[0]} wrong; grey {grey}")
        pg.close()
    b.close()
