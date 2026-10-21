import io
from PIL import Image
from playwright.sync_api import sync_playwright
def html(filters, body):
    return f'<!doctype html><html><body style="margin:0;background:#fff"><svg width="0" height="0" style="position:absolute"><defs>{filters}</defs></svg>{body}</body></html>'
def conv(ox, taps):
    k = ["0"] * ox
    for x, v in taps.items(): k[x] = str(v)
    return (f'<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">'
            f'<feConvolveMatrix order="{ox} 1" kernelMatrix="{" ".join(k)}" divisor="1" preserveAlpha="true" edgeMode="none"/></filter>')
cells = "".join(f'<div style="width:4px;height:20px;background:{"#000" if i == 5 else "#fff"}"></div>' for i in range(12))
BODY = f'<div style="display:flex;width:48px;filter:url(#f)">{cells}</div>'
with sync_playwright() as p:
    b = p.chromium.launch()
    for dpr in (1, 2):
        pg = b.new_page(viewport={"width": 60, "height": 30}, device_scale_factor=dpr)
        for taps in ({4: 1}, {0: 1}, {8: 1}):
            pg.set_content(html(conv(9, taps), BODY))
            img = Image.open(io.BytesIO(pg.screenshot())).convert("L")
            row = [img.getpixel((x, int(10 * dpr))) for x in range(int(52 * dpr))]
            print(f"dpr{dpr} taps {taps}:", "".join("#" if v < 100 else ("+" if v < 200 else ".") for v in row))
        pg.close()
    b.close()
