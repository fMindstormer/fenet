import io, random
from PIL import Image
from playwright.sync_api import sync_playwright
N = 14
random.seed(5)
pat = [[random.random() < 0.3 for _ in range(N)] for _ in range(N)]
flt = ('<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">'
       '<feConvolveMatrix order="3 1" kernelMatrix="1 0 0" divisor="1" preserveAlpha="true" edgeMode="none"/></filter>')
with sync_playwright() as p:
    b = p.chromium.launch()
    for dpr in (1, 1.25, 1.5, 1.75, 2, 2.5, 3):
        pg = b.new_page(viewport={"width": 120, "height": 120}, device_scale_factor=dpr)
        cell = 1 / dpr
        rects = "".join(f'<rect x="{c}" y="{r}" width="1" height="1" fill="{"#000" if pat[r][c] else "#fff"}"/>' for r in range(N) for c in range(N))
        res = []
        for left in (16, 17.3):
            el = f'<svg style="position:absolute;left:{left}px;top:16px;filter:url(#f)" width="{N*cell}" height="{N*cell}" viewBox="0 0 {N} {N}" shape-rendering="crispEdges">{rects}</svg>'
            pg.set_content(f'<!doctype html><html><body style="margin:0;background:#888"><svg width="0" height="0" style="position:absolute"><defs>{flt}</defs></svg>{el}</body></html>')
            img = Image.open(io.BytesIO(pg.screenshot())).convert("L")
            ox, oy = round(left * dpr), round(16 * dpr)
            best = min((sum((img.getpixel((ox + dx + c, oy + dy + r)) < 128) != pat[r][c + 1]
                            for r in range(N) for c in range(N - 1)), dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
            res.append(f"left {left}: {best[0]} wrong (offset {best[1]},{best[2]})")
        print(f"dpr {dpr}:", "; ".join(res))
        pg.close()
    b.close()
