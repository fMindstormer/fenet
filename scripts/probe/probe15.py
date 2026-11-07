"""A mirror grid whose cells are exactly one device pixel (cell = 1/DPR CSS px), filtered by a convolution."""
import io, random
from PIL import Image
from playwright.sync_api import sync_playwright
N = 14
random.seed(5)
pat = [[random.random() < 0.3 for _ in range(N)] for _ in range(N)]
with sync_playwright() as p:
    b = p.chromium.launch()
    for dpr in (1, 1.25, 1.5, 1.75, 2, 2.5, 3):
        pg = b.new_page(viewport={"width": 200, "height": 200}, device_scale_factor=dpr)
        cell = 1 / dpr
        cells = "".join(f'<i style="background:{"#000" if pat[r][c] else "#fff"}"></i>' for r in range(N) for c in range(N))
        # kernel 3x1 with tap at the left: output(x) = input(x+1)  (flipped convolution: kernel index 0 -> offset +1)
        flt = ('<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">'
               '<feConvolveMatrix order="3 1" kernelMatrix="1 0 0" divisor="1" preserveAlpha="true" edgeMode="none"/></filter>')
        for left in (16, 17.3):
            html = (f'<!doctype html><html><body style="margin:0;background:#888"><svg width="0" height="0" style="position:absolute"><defs>{flt}</defs></svg>'
                    f'<div style="position:absolute;left:{left}px;top:16px;display:grid;grid-template-columns:repeat({N},{cell}px);grid-auto-rows:{cell}px;filter:url(#f)">'
                    f'<style>i{{display:block}}</style>{cells}</div></body></html>')
            pg.set_content(html)
            img = Image.open(io.BytesIO(pg.screenshot())).convert("L")
            ox, oy = round(left * dpr), round(16 * dpr)
            # find the grid's device origin by searching near the expected place for the best match
            best = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    bad = sum((img.getpixel((ox + dx + c, oy + dy + r)) < 128) != (pat[r][c + 1] if c + 1 < N else False)
                              for r in range(N) for c in range(N))
                    if best is None or bad < best[0]:
                        best = (bad, dx, dy)
            print(f"dpr {dpr} left {left}: {best[0]} wrong cells (origin offset {best[1]},{best[2]})")
        pg.close()
    b.close()
