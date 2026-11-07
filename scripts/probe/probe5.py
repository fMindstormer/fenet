import io, time
from PIL import Image
from playwright.sync_api import sync_playwright

def html(filters, body):
    return f'<!doctype html><html><body style="margin:0;background:#fff"><svg width="0" height="0" style="position:absolute"><defs>{filters}</defs></svg>{body}</body></html>'

def conv(ox, oy, taps, extra=""):
    k = [0.0] * (ox * oy)
    for (x, y), v in taps.items():
        k[y * ox + x] = v
    return (f'<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">'
            f'<feConvolveMatrix order="{ox} {oy}" kernelMatrix="{" ".join(map(str, k))}" divisor="1" preserveAlpha="true" edgeMode="none" {extra}/></filter>')

cells = "".join(f'<div style="width:4px;height:20px;background:{"#000" if i == 5 else "#fff"}"></div>' for i in range(12))
BODY = f'<div style="display:flex;width:48px;filter:url(#f)">{cells}</div>'
with sync_playwright() as p:
    b = p.chromium.launch()
    for dpr in (1, 1.5, 2):
        pg = b.new_page(viewport={"width": 100, "height": 40}, device_scale_factor=dpr)
        # kernel 9x1 with a single 1 at x=0, target at the centre (4): output(x) = input(x - 4 px)
        for name, extra in [("plain", ""), ("unit4", 'kernelUnitLength="4"')]:
            pg.set_content(html(conv(9, 1, {(0, 0): 1.0}, extra), BODY))
            img = Image.open(io.BytesIO(pg.screenshot())).convert("L")
            row = [img.getpixel((int((x + 0.5) * dpr), int(10 * dpr))) for x in range(48)]
            dark = [x for x, v in enumerate(row) if v < 128]
            print(f"dpr{dpr} {name}: black stripe (CSS px) {dark[:1]}..{dark[-1:]}  (input stripe at 20..23)")
        pg.close()
    # kernel size limits and speed
    pg = b.new_page(viewport={"width": 300, "height": 300})
    for ox, oy in [(5, 5), (25, 25), (64, 16), (128, 8), (255, 3), (400, 2)]:
        taps = {(0, 0): 0.5, (ox - 1, oy - 1): 0.5}
        pg.set_content(html(conv(ox, oy, taps), '<div style="width:240px;height:120px;background:linear-gradient(90deg,#000,#fff);filter:url(#f)"></div>'))
        t = time.time()
        img = Image.open(io.BytesIO(pg.screenshot())).convert("L")
        print(f"kernel {ox}x{oy}: {(time.time()-t)*1000:.0f} ms, pixel {img.getpixel((120, 60))}")
    b.close()
