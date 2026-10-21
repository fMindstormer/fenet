import sys, io
import numpy as np
from PIL import Image
from pathlib import Path
from playwright.sync_api import sync_playwright
dpr = float(sys.argv[1]); filt = sys.argv[2]; w = int(sys.argv[3]); h = int(sys.argv[4])
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=dpr)
    pg.goto(Path("site/test.html").resolve().as_uri())
    pg.add_style_tag(content=f".out{{filter:{filt} !important}} body{{background:#808080 !important}} .sheet{{background:#808080 !important}}")
    pg.wait_for_timeout(800)
    box = pg.locator(".out").bounding_box()
    print("box", box["x"] * dpr, box["y"] * dpr)
    im = np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("L")).astype(int)
    ox, oy = int(box["x"] * dpr), int(box["y"] * dpr)
    reg = im[oy - 1:oy + h, ox - 1:ox + w]
    for row in reg:
        print(" ".join(f"{v:3d}" for v in row))
    b.close()
