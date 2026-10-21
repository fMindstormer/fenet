import sys, os
from pathlib import Path
from playwright.sync_api import sync_playwright
dpr = float(sys.argv[1]); name = sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1100, "height": 800}, device_scale_factor=dpr)
    pg.goto(Path("site/test.html").resolve().as_uri())
    if os.environ.get("EXTRA_CSS"): pg.add_style_tag(content=os.environ["EXTRA_CSS"])
    pg.wait_for_timeout(1200)
    pg.locator(".sheet").screenshot(path=f"{name}.png")
    b.close()
