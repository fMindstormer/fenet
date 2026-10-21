import sys, io
import numpy as np
from PIL import Image
from pathlib import Path
from playwright.sync_api import sync_playwright
dpr = float(sys.argv[1]); filt = sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=dpr)
    pg.goto(Path("site/test.html").resolve().as_uri())
    pg.add_style_tag(content=f".out{{filter:{filt} !important}}")
    pg.wait_for_timeout(800)
    print(pg.evaluate("() => [getComputedStyle(document.querySelector('.mirror rect.m.s')).fill, getComputedStyle(document.querySelector('.out')).filter, document.querySelectorAll('.mirror rect.m.s').length]"))
    box = pg.locator(".out").bounding_box()
    im = np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("L")).astype(int)
    ox, oy = int(box["x"] * dpr), int(box["y"] * dpr)
    reg = im[oy:oy + 16, ox:ox + 46]
    for row in reg:
        print("".join("#" if v > 200 else "+" if v > 100 else "-" if v > 50 else "." for v in row))
    b.close()
