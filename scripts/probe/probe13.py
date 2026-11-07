import io
from PIL import Image
from playwright.sync_api import sync_playwright
exec(open("probe/probe11.py").read().split("random.seed")[0])
cells = "".join(f'<i style="background:rgb({c*16+8},{r*16+8},0)"></i>' for r in range(N) for c in range(N))
with sync_playwright() as p:
    b = p.chromium.launch()
    for dpr in (1, 2):
        pg = b.new_page(viewport={"width": 200, "height": 200}, device_scale_factor=dpr)
        html = (f'<!doctype html><html><body style="margin:0;background:#fff"><svg width="0" height="0" style="position:absolute"><defs>{gather_filter("g", dpr)}</defs></svg>'
                f'<style>.m{{position:absolute;left:16px;top:16px;display:grid;grid-template-columns:repeat({N},{CELL}px);filter:url(#g)}} .m i{{width:{CELL}px;height:{CELL}px}}</style>'
                f'<div class="m">{cells}</div></body></html>')
        pg.set_content(html)
        img = Image.open(io.BytesIO(pg.screenshot())).convert("RGB")
        ox = round(16 * dpr)
        print(f"dpr {dpr} sampled (col,row) along row 3:", [((v[0] - 8) / 16, (v[1] - 8) / 16) for v in (img.getpixel((ox + i, ox + 3)) for i in range(N))])
        print(f"dpr {dpr} sampled (col,row) down col 3:", [((v[0] - 8) / 16, (v[1] - 8) / 16) for v in (img.getpixel((ox + 3, ox + i)) for i in range(N))])
        pg.close()
    b.close()
