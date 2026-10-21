import time
from playwright.sync_api import sync_playwright

def html(filters, body):
    return f'<!doctype html><html><body style="margin:0;background:#fff"><svg width="0" height="0" style="position:absolute"><defs>{filters}</defs></svg>{body}</body></html>'
BODY = '<div id="d" style="width:240px;height:240px;background:linear-gradient(90deg,#000,#fff);filter:url(#f)"></div>'

def conv_layers(L, taps=9, extra=""):
    s, prev = "", "SourceGraphic"
    for l in range(L):
        acc = None
        for t in range(taps):
            dx, dy = (t % 3 - 1) * 10, (t // 3 - 1) * 10
            s += f'<feOffset in="{prev}" dx="{dx}" dy="{dy}" result="L{l}o{t}"/>'
            s += f'<feColorMatrix in="L{l}o{t}" type="matrix" values="0.1 0 0 0 0.05  0 0.1 0 0 0.05  0 0 0.1 0 0.05  0 0 0 1 0" result="L{l}m{t}"/>'
            if acc is None:
                acc = f"L{l}m{t}"
            else:
                s += f'<feComposite in="{acc}" in2="L{l}m{t}" operator="arithmetic" k2="1" k3="1" k4="-0.05" result="L{l}a{t}"/>'
                acc = f"L{l}a{t}"
        s += f'<feComponentTransfer in="{acc}" result="L{l}out"><feFuncR type="linear" slope="1"/></feComponentTransfer>{extra.format(l=l)}'
        prev = f"L{l}out" if not extra else f"L{l}x"
    return s

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 300, "height": 300})
    for label, extra in [("plain", ""), ("tile-reset", '<feTile in="L{l}out" result="L{l}x"/>'), ("merge", '<feMerge result="L{l}x"><feMergeNode in="L{l}out"/></feMerge>')]:
        for L in (1, 2, 3, 4, 6):
            pg.set_content(html(f'<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">{conv_layers(L, extra=extra)}</filter>', BODY))
            t = time.time()
            try:
                pg.screenshot(timeout=60000)
                print(f"{label} L={L}: {(time.time()-t)*1000:.0f} ms", flush=True)
            except Exception:
                print(f"{label} L={L}: timeout", flush=True)
                break
    b.close()
