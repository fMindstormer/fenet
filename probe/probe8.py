import time
from playwright.sync_api import sync_playwright
def html(filters, body):
    return f'<!doctype html><html><body style="margin:0;background:#fff"><svg width="0" height="0" style="position:absolute"><defs>{filters}</defs></svg>{body}</body></html>'
def conv_layers(L, size):
    s, prev = "", "SourceGraphic"
    for l in range(L):
        acc = None
        for t in range(9):
            dx, dy = (t % 3 - 1), (t // 3 - 1)
            s += f'<feOffset in="{prev}" dx="{dx}" dy="{dy}" result="L{l}o{t}"/>'
            s += f'<feColorMatrix in="L{l}o{t}" type="matrix" values="0.1 0 0 0 0.05  0 0.1 0 0 0.05  0 0 0.1 0 0.05  0 0 0 1 0" result="L{l}m{t}"/>'
            if acc is None: acc = f"L{l}m{t}"
            else:
                s += f'<feComposite in="{acc}" in2="L{l}m{t}" operator="arithmetic" k2="1" k3="1" k4="-0.05" result="L{l}a{t}"/>'; acc = f"L{l}a{t}"
        s += f'<feComponentTransfer in="{acc}" result="L{l}out"><feFuncR type="linear" slope="1"/></feComponentTransfer>'
        prev = f"L{l}out"
    return s
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 300, "height": 300})
    for size in (16, 32, 64):
        for L in (2, 4, 6, 8, 12):
            body = f'<div style="width:{size}px;height:{size}px;background:linear-gradient(90deg,#000,#fff);filter:url(#f)"></div>'
            pg.set_content(html(f'<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">{conv_layers(L, size)}</filter>', body))
            t = time.time()
            try:
                pg.screenshot(timeout=30000); print(f"{size}px L={L}: {(time.time()-t)*1000:.0f} ms", flush=True)
            except Exception:
                print(f"{size}px L={L}: timeout", flush=True); break
    b.close()
