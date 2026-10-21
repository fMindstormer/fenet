import time
from playwright.sync_api import sync_playwright
def html(filters, body):
    return f'<!doctype html><html><body style="margin:0;background:#fff"><svg width="0" height="0" style="position:absolute"><defs>{filters}</defs></svg>{body}</body></html>'
BODY = '<div style="width:240px;height:240px;background:linear-gradient(90deg,#000,#fff);filter:url(#f)"></div>'
K = "0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1"
def diamonds(n, node):
    s = ""
    for i in range(n):
        prev = "SourceGraphic" if i == 0 else f"c{i-1}"
        if node == "conv":
            s += f'<feConvolveMatrix in="{prev}" order="3" kernelMatrix="{K}" preserveAlpha="true" result="o{i}"/>'
        elif node == "transfer":
            s += f'<feComponentTransfer in="{prev}" result="o{i}"><feFuncR type="linear" slope="0.9"/></feComponentTransfer>'
        elif node == "morph":
            s += f'<feMorphology in="{prev}" operator="dilate" radius="1" result="o{i}"/>'
        s += f'<feComposite in="o{i}" in2="{prev}" operator="arithmetic" k2="0.5" k3="0.5" result="c{i}"/>'
    return s
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 300, "height": 300})
    for node in ("conv", "transfer", "morph"):
        for n in (4, 8, 12, 16, 20):
            pg.set_content(html(f'<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">{diamonds(n, node)}</filter>', BODY))
            t = time.time()
            try:
                pg.screenshot(timeout=30000); print(f"{node} {n}: {(time.time()-t)*1000:.0f} ms", flush=True)
            except Exception:
                print(f"{node} {n}: timeout", flush=True); break
    b.close()
