import io, random
from PIL import Image
from playwright.sync_api import sync_playwright
N = 14
random.seed(5)
pat = [[random.random() < 0.3 for _ in range(N)] for _ in range(N)]
with sync_playwright() as p:
    b = p.chromium.launch()
    for dpr in (2, 1.25):
        pg = b.new_page(viewport={"width": 100, "height": 100}, device_scale_factor=dpr)
        cell = 1 / dpr
        for mode in ("grid", "svg"):
            if mode == "grid":
                cells = "".join(f'<i style="display:block;background:{"#000" if pat[r][c] else "#fff"}"></i>' for r in range(N) for c in range(N))
                el = f'<div style="position:absolute;left:16px;top:16px;display:grid;grid-template-columns:repeat({N},{cell}px);grid-auto-rows:{cell}px">{cells}</div>'
            else:
                rects = "".join(f'<rect x="{c}" y="{r}" width="1" height="1" fill="{"#000" if pat[r][c] else "#fff"}"/>' for r in range(N) for c in range(N))
                el = f'<svg style="position:absolute;left:16px;top:16px" width="{N*cell}" height="{N*cell}" viewBox="0 0 {N} {N}" shape-rendering="crispEdges">{rects}</svg>'
            pg.set_content(f'<!doctype html><html><body style="margin:0;background:#888">{el}</body></html>')
            img = Image.open(io.BytesIO(pg.screenshot())).convert("L")
            ox = oy = round(16 * dpr)
            bad = sum((img.getpixel((ox + c, oy + r)) < 128) != pat[r][c] for r in range(N) for c in range(N))
            grey = sum(60 < img.getpixel((ox + c, oy + r)) < 200 for r in range(N) for c in range(N))
            print(f"dpr {dpr} {mode}: {bad} wrong, {grey} grey (antialiased) pixels")
        pg.close()
    b.close()
