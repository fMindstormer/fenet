import time
from playwright.sync_api import sync_playwright
N = 14
def page(nflood, nconv):
    floods = "".join(f'<feFlood id="m{i}" flood-color="{"#fff" if i % 3 == 0 else "#000"}" x="{i % N}" y="{i // N}" width="1" height="1" result="c{i}"/>' for i in range(nflood))
    merge = '<feFlood flood-color="#000" x="0" y="0" width="14" height="14" result="bg"/><feMerge x="0" y="0" width="14" height="14" result="inp"><feMergeNode in="bg"/>' + "".join(f'<feMergeNode in="c{i}"/>' for i in range(nflood)) + "</feMerge>"
    blur = '<feConvolveMatrix in="inp" order="3" kernelMatrix="1 2 1 2 4 2 1 2 1" divisor="16" preserveAlpha="true" x="0" y="0" width="14" height="14" result="bl"/>'
    k = " ".join(["0.005"] * 196)
    convs = "".join(f'<feConvolveMatrix in="bl" order="14" targetX="{j % N}" targetY="{j // N}" kernelMatrix="{k}" divisor="1" preserveAlpha="true" x="{j % N}" y="{j // N}" width="1" height="1" result="h{j}"/>' for j in range(nconv))
    merge2 = '<feMerge x="0" y="0" width="14" height="14"><feMergeNode in="bg"/>' + "".join(f'<feMergeNode in="h{j}"/>' for j in range(nconv)) + "</feMerge>"
    flt = f'<filter id="f" x="0" y="0" width="14" height="14" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" color-interpolation-filters="sRGB">{floods}{merge}{blur}{convs}{merge2}</filter>'
    return f'<!doctype html><html><body><svg width="100" height="100"><defs>{flt}</defs><g filter="url(#f)"><rect width="14" height="14"/></g></svg></body></html>'
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    for nf, nc in [(196, 0), (196, 16), (196, 64), (196, 128), (49, 128), (0, 128)]:
        pg.set_content(page(nf, nc))
        t = time.time()
        try:
            pg.screenshot(timeout=30000); print(f"floods {nf}, neurons {nc}: {(time.time()-t)*1000:.0f} ms", flush=True)
        except Exception:
            print(f"floods {nf}, neurons {nc}: timeout", flush=True)
    b.close()
