import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
dpr = float(sys.argv[1]); w = int(sys.argv[2]) if len(sys.argv) > 2 else 1100; scheme = sys.argv[3] if len(sys.argv) > 3 else "light"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": w, "height": 900}, device_scale_factor=dpr, color_scheme=scheme)
    pg.goto(Path("site/test.html").resolve().as_uri())
    pg.wait_for_timeout(1500)
    pg.screenshot(path=f"shot_{dpr}_{w}_{scheme}.png", full_page=True)
    b.close()
