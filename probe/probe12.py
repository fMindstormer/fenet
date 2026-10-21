import io, re
from PIL import Image
from playwright.sync_api import sync_playwright
exec(open("probe/probe11.py").read().split("random.seed")[0])
with sync_playwright() as p:
    b = p.chromium.launch()
    for dpr in (1, 2):
        pg = b.new_page(viewport={"width": 200, "height": 200}, device_scale_factor=dpr)
        flt = gather_filter("g", dpr)
        W = N / dpr
        S = 2 * (W * (CELL * dpr - 1) + CELL)
        c0 = 0.5 + (CELL / 2) / S; c1 = 0.5 + (W * (CELL * dpr - 1) + CELL / 2) / S
        flt_map = re.sub(r'<feDisplacementMap[^>]*/>', '', flt)   # output the map itself
        html = (f'<!doctype html><html><body style="margin:0;background:#000"><svg width="0" height="0" style="position:absolute"><defs>{flt_map}</defs></svg>'
                f'<div style="position:absolute;left:16px;top:16px;width:{N*CELL}px;height:{N*CELL}px;background:#fff;filter:url(#g)"></div></body></html>')
        pg.set_content(html)
        img = Image.open(io.BytesIO(pg.screenshot())).convert("RGB")
        ox = round(16 * dpr)
        row = [img.getpixel((ox + i, ox + 2))[0] for i in range(N)]
        want = [round(255 * (c0 + (c1 - c0) * (i + 0.5) / N)) for i in range(N)]
        print(f"dpr {dpr} S={S:.1f}\n  got  {row}\n  want {want}")
        pg.close()
    b.close()
