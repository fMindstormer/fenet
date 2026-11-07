import io
from PIL import Image
from playwright.sync_api import sync_playwright
def html(f, body): return f'<!doctype html><html><body style="margin:0;background:#fff"><svg width="0" height="0" style="position:absolute"><defs>{f}</defs></svg>{body}</body></html>'
def conv(ox, oy):
    k = ["0"] * (ox * oy); k[len(k) // 2] = "0.5"
    return f'<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB"><feConvolveMatrix order="{ox} {oy}" kernelMatrix="{" ".join(k)}" divisor="1" preserveAlpha="true"/></filter>'
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 100, "height": 100})
    for ox, oy in [(16, 16), (1, 257), (1, 300), (1, 400), (1, 512), (1, 513), (16, 32), (1, 1024)]:
        pg.set_content(html(conv(ox, oy), '<div style="width:60px;height:60px;background:#fff;filter:url(#f)"></div>'))
        v = Image.open(io.BytesIO(pg.screenshot())).convert("L").getpixel((30, 30))
        print(f"{ox}x{oy} (area {ox*oy}): {'works' if 120 < v < 136 else 'ignored' if v > 250 else v}")
    b.close()
