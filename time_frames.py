"""Frame cost on the real page: screenshot with nothing changed vs. after one painted cell."""
import os, sys, time
import numpy as np
from pathlib import Path
from playwright.sync_api import sync_playwright
from readout import launch
page = sys.argv[1] if len(sys.argv) > 1 else "site/test.html"
with sync_playwright() as p:
    b = launch(p)
    for dpr in [float(v) for v in os.environ.get("DPRS", "1,2").split(",")]:
        pg = b.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=dpr)
        pg.goto(Path(page).resolve().as_uri())
        if os.environ.get("EXTRA_CSS"): pg.add_style_tag(content=os.environ["EXTRA_CSS"])
        pg.wait_for_timeout(1500)
        clip = pg.locator(".sheet").bounding_box()
        base, change = [], []
        rects = pg.locator(".mirror rect.m")
        for i in range(8):
            t0 = time.time(); pg.screenshot(clip=clip); base.append(time.time() - t0)
            pg.evaluate("i => { const r = document.querySelectorAll('.mirror rect.m')[60 + i]; r.style.transition = 'none'; r.style.fill = '#fff'; }", i)
            t0 = time.time(); pg.screenshot(clip=clip); change.append(time.time() - t0)
        print(f"{page} dpr {dpr}: unchanged {1000 * np.median(base):.0f} ms, one cell changed {1000 * np.median(change):.0f} ms", flush=True)
        pg.close()
    b.close()
