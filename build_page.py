"""Build the page: a drawing grid, and an SVG whose filters run the network on a mirror of it.

    python build_page.py            writes site/index.html (the Artifact page, no skeleton)
                                    and site/test.html (with a document skeleton, for Chromium)

How the result SVG is drawn:
  * `.net` holds the 14x14 mirror of the drawing, scaled by 1/devicePixelRatio so one cell is one
    device pixel; the network runs on it as one filter per layer (#net1..#net3, from compile_mlp) on
    nested groups. Its output: the winner's 3x3 block around slot k and the ink flag at (13, 9) in red,
    the score of class k as a 3x3 block around slot k in green.
  * The CSS filter on `.out` is two filters in a row. The first (#x<ratio>, one per supported
    ratio) copies those single device pixels into 4x4-CSS-px blocks of a small control image. The
    second (#render, shared) reads the blocks and draws the answer as a 5x7 dot-matrix digit and
    ten score bars, in colours taken from the page's theme (flood-color).
  * The .out element is its own compositing layer (will-change), which Chromium rasterizes from a
    whole device pixel, at a scale of max(1, devicePixelRatio). Below 1 the group is not scaled.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from compile_mlp import INK, N, SLOTS, build, emulate, to_svg

ROOT = Path(__file__).resolve().parent
CELL = 20                    # CSS px per drawing cell
PAD = N * CELL               # the drawing pad, 280 px
W, H = 280, 316              # the readout
DOT, PITCH = 10.5, 28        # dot-matrix glyph: dot radius and pitch
GX, GY = (W - 5 * PITCH) // 2, 10
BY, BH, BW = 232, 76, 14     # bars: top, height, width; column k is 28 px wide

# Chromium's preset zoom levels times common device scale factors: every ratio above 1 needs its own
# scale for the mirror and its own sampling filter. Ratios at or below 1 need nothing.
BASES = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.625, 2.75, 3, 3.5, 4]
ZOOMS = [2 / 3, 0.75, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2]


LOW = [0.5, 7 / 12, 0.625, 2 / 3, 0.75, 0.8, 5 / 6, 0.875, 0.9, 0.9375]   # ratios below 1 that get a scaled readout
READ_H = 334                 # the readout's height: the SVG and the digit labels


def ratios():
    vals = sorted({round(b * z, 4) for b in BASES for z in ZOOMS if 1.005 < b * z <= 5.0})
    out = []
    for v in vals:
        if not out or v - out[-1] > 0.006:
            out.append(v)
    return out


FONT = {  # 5x7, rows top to bottom
    "0": "01110 10001 10011 10101 11001 10001 01110", "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111", "3": "11111 00010 00100 00010 00001 10001 01110",
    "4": "00010 00110 01010 10010 11111 00010 00010", "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110", "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110", "9": "01110 10001 10001 01111 00001 00010 01100",
    "?": "01110 10001 00001 00010 00100 00000 00100", "all": "11111 11111 11111 11111 11111 11111 11111",
}


def data_svg(body: str, w: float, h: float) -> str:
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>{body}</svg>"
    return "data:image/svg+xml," + svg.replace("<", "%3C").replace(">", "%3E").replace("#", "%23").replace('"', "'")


def glyph_uri(ch: str) -> str:
    rows = FONT[ch].split()
    dots = "".join(f"<circle cx='{c * PITCH + PITCH / 2}' cy='{r * PITCH + PITCH / 2}' r='{DOT}'/>"
                   for r, row in enumerate(rows) for c, v in enumerate(row) if v == "1")
    return data_svg(f"<g fill='white'>{dots}</g>", 5 * PITCH, 7 * PITCH)


SEG = 4                      # bar segment pitch (3 px lit, 1 px gap), like an LED meter


def ramp_uri() -> str:
    """Grey level = the score a segment needs to light: solid steps, since Chromium dithers gradients.
    The gaps are white, which no score reaches."""
    n = BH // SEG
    rects = "".join(f"<rect y='{BH - (i + 1) * SEG + 1}' width='10' height='{SEG - 1}' fill='rgb({round(255 * (i + 0.5) / n)},0,0)'/>"
                    for i in range(n))
    return data_svg(f"<rect width='10' height='{BH}' fill='rgb(255,0,0)'/>{rects}", 10, BH)


def tag(d: float) -> str:
    return f"{d:.4f}".rstrip("0").rstrip(".").replace(".", "_")


def extract_filter(d: float) -> tuple[str, str]:
    """Device pixel (x, y) of the net's output -> a 4x4 CSS px block of the control image (red: winner
    flag or ink, green: score)."""
    fid = f"x{tag(d)}"
    px = 1 / max(d, 1)
    items = ([(sx, sy, 4 * k, 0) for k, (sx, sy) in enumerate(SLOTS)]           # winner flag (red), score (green)
             + [(INK[0], INK[1], 40, 0)])                                      # anything drawn (red)
    # the region covers the net's output too: a filter only sees its source inside its own region
    parts = [f'<filter id="{fid}" x="0" y="0" width="44" height="{N}" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" color-interpolation-filters="sRGB">']
    for i, (sx, sy, tx, ty) in enumerate(items):
        # the middle pixel of a 3x3 block: at fractional ratios the compositor filters at a coarser
        # scale than the content was drawn, so single pixels blur but the middle of a block holds
        parts.append(f'<feOffset in="SourceGraphic" x="{sx * px:.6g}" y="{sy * px:.6g}" width="{px:.6g}" height="{px:.6g}" result="a{i}"/>'
                     f'<feTile in="a{i}" x="{tx}" y="{ty}" width="4" height="4" result="b{i}"/>')
    parts.append("<feMerge>" + "".join(f'<feMergeNode in="b{i}"/>' for i in range(len(items))) + "</feMerge></filter>")
    return fid, "".join(parts)


A_IS_R = 'type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  1 0 0 0 0"'
R_IS_G = 'type="matrix" values="0 1 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0"'


CROP = os.environ.get("RENDER_CROP", "inner")


def crop(bx, by):
    """Subregion that reads the control block whose corner is (bx, by)."""
    if CROP == "block":
        return f'x="{bx}" y="{by}" width="4" height="4"'
    if CROP == "inner":
        return f'x="{bx + 0.5}" y="{by + 0.5}" width="3" height="3"'
    return f'x="{bx + 1}" y="{by + 1}" width="1" height="1"'


def render_filter() -> str:
    full = f'x="0" y="0" width="{W}" height="{H}"'
    glyph = f'x="{GX}" y="{GY}" width="{5 * PITCH}" height="{7 * PITCH}"'
    p = [f'<filter id="render" {full} filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" color-interpolation-filters="sRGB">',
         f'<feOffset in="SourceGraphic" {crop(40, 0)} result="i0"/><feTile in="i0" {full} result="i1"/>',
         f'<feColorMatrix in="i1" {A_IS_R} result="ink"/>',
         f'<feImage href="{ramp_uri()}" x="0" y="{BY}" width="{W}" height="{BH}" preserveAspectRatio="none" result="ramp"/>']
    for k in range(10):
        cx = 28 * k + (28 - BW) / 2
        col = f'x="{cx}" y="{BY}" width="{BW}" height="{BH}"'
        p.append(f'<feOffset in="SourceGraphic" {crop(4 * k, 0)} result="w{k}a"/>'
                 f'<feTile in="w{k}a" {full} result="w{k}b"/><feColorMatrix in="w{k}b" {A_IS_R} result="w{k}"/>'
                 f'<feImage href="{glyph_uri(str(k))}" {glyph} result="g{k}"/>'
                 f'<feComposite in="g{k}" in2="w{k}" operator="in" result="G{k}"/>'
                 f'<feTile in="w{k}a" {col} result="s{k}t"/><feColorMatrix in="s{k}t" {R_IS_G} result="s{k}b"/>'
                 # 255 (score - height) + 1: on where the score reaches this height, off a step below
                 f'<feComposite in="s{k}b" in2="ramp" operator="arithmetic" k2="255" k3="-255" k4="1" {col} result="c{k}"/>'
                 f'<feColorMatrix in="c{k}" {A_IS_R} result="bar{k}"/>'
                 f'<feComposite in="bar{k}" in2="w{k}" operator="in" result="wb{k}"/>')
    merge = lambda names, res: f'<feMerge result="{res}">' + "".join(f'<feMergeNode in="{n}"/>' for n in names) + "</feMerge>"
    p += [merge([f"G{k}" for k in range(10)], "G"), merge([f"bar{k}" for k in range(10)], "B"),
          merge([f"wb{k}" for k in range(10)], "WB"),
          '<feFlood class="fl-mark" result="fm"/><feFlood class="fl-bar" result="fb"/><feFlood class="fl-dim" result="fd"/>',
          f'<feImage href="{glyph_uri("all")}" {glyph} result="all"/><feComposite in="fd" in2="all" operator="in" result="matrix"/>',
          '<feComposite in="fm" in2="G" operator="in" result="Gc"/><feComposite in="fb" in2="B" operator="in" result="Bc"/>',
          '<feComposite in="fm" in2="WB" operator="in" result="WBc"/>',
          merge(["Bc", "WBc", "Gc"], "lit"), '<feComposite in="lit" in2="ink" operator="in" result="shown"/>',
          f'<feImage href="{glyph_uri("?")}" {glyph} result="q"/><feComposite in="q" in2="ink" operator="out" result="q1"/>',
          '<feComposite in="fb" in2="q1" operator="in" result="qc"/>',
          merge(["matrix", "qc", "shown"], "out"), "</filter>"]
    return "".join(p)


def sample_cells() -> list[int]:
    s = json.loads((ROOT / "sample.json").read_text())
    return [i for i, v in enumerate(s["cells"]) if v]


def page(g, test=False, facts=None):
    net = to_svg(g, "net")
    rs = ratios()
    extracts = [extract_filter(1)] + [extract_filter(d) for d in rs]
    cells = "".join(f'<rect x="{i % N}" y="{i // N}" width="1" height="1"/>' for i in range(N * N))
    mirror = "".join(f'<rect class="m" x="{i % N}" y="{i // N}" width="1" height="1"/>' for i in range(N * N))
    # the opening example: an overlay on the pad and on its mirror, after the cells so nth-child counts don't move
    ex = '<g class="ex">' + "".join(f'<rect x="{i % N}" y="{i // N}" width="1" height="1"/>' for i in sample_cells()) + "</g>"
    # each drawn cell lights its mirror cell (rects are children 1..196 of the grid, 2..197 of the mirror)
    rules = "\n".join(f".app:has(.grid:active rect:nth-child({i}):hover) .mirror rect.m:nth-of-type({i + 1}) {{ fill: #fff; transition: fill 0s; }}"
                      for i in range(1, N * N + 1))
    # below 1, filters on the layer run at the screen's scale but its content is drawn at 1: scaling the
    # readout up by 1/ratio makes the two agree (the readout keeps the size it has at 100%)
    media = [f"@media (max-resolution: 0.996dppx), (min-resolution: 1.004dppx) {{ .out {{ display: none; }} .zoomnote {{ display: block; }} }}"]
    for d in LOW:
        media.append(f"@media (min-resolution: {d - 0.003:.4f}dppx) and (max-resolution: {d + 0.003:.4f}dppx) {{"
                     f" .readout {{ transform: scale({1 / d:.6f}); margin: 0 {PAD * (1 / d - 1):.2f}px {READ_H * (1 / d - 1):.2f}px 0; }}"
                     f" .field.read {{ width: auto; }} .out {{ display: block; }} .zoomnote {{ display: none; }} }}")
    for d, (fid, _) in zip(rs, extracts[1:]):
        media.append(f"@media (min-resolution: {d - 0.003:.4f}dppx) and (max-resolution: {d + 0.003:.4f}dppx) {{"
                     f" .net {{ transform: scale({1 / d:.7f}); }} .out {{ filter: url(#{fid}) url(#render); display: block; }} .zoomnote {{ display: none; }} }}")
    tmpl = (ROOT / "site" / "template.html").read_text()
    grid = (f'<svg class="grid" viewBox="0 0 {N} {N}" width="{PAD}" height="{PAD}" shape-rendering="crispEdges" '
            f'role="img" aria-label="Drawing pad, 14 by 14 cells">{cells}{ex}</svg>')
    # one group per filter, the first layer innermost: each layer reads the one below as a finished image
    nested = (f'<g class="mirror" filter="url(#net1)" shape-rendering="crispEdges">'
              f'<rect class="bg" width="{N}" height="{N}" fill="#000"/>{mirror}{ex}</g>')
    for k in range(2, len(g.stages()) + 1):
        nested = f'<g filter="url(#net{k})">{nested}</g>'
    out = (f'<svg class="out" width="{W}" height="{H}" viewBox="0 0 {W} {H}" aria-hidden="true">'
           f'<defs>{net}{"".join(s for _, s in extracts)}{render_filter()}</defs>'
           f'<g class="net">{nested}</g></svg>')
    from diagram import html as diagram          # imports this module, so not at the top
    html = (tmpl.replace("/*RULES*/", rules).replace("/*MEDIA*/", "\n".join(media))
            .replace("<!--GRID-->", grid).replace("<!--OUT-->", out).replace("<!--DIAGRAM-->", diagram()))
    for k, v in (facts or {}).items():
        html = html.replace("{{" + k + "}}", str(v))
    if test:
        html = f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>{html}</body></html>'
    return html, len(rs)


def main():
    model = dict(np.load(ROOT / "mlp.npz"))
    alpha = float(json.loads((ROOT / "alpha.json").read_text())["alpha"])
    g = build(model, alpha)
    facts = json.loads((ROOT / "facts.json").read_text()) if (ROOT / "facts.json").exists() else {}
    facts["primitives"] = len(g.ops)
    h1, h2 = model["W1"].shape[1], model["W2"].shape[1]
    facts.update(h1=h1, h2=h2, convs=sum(kind in ("conv", "neuron", "taps") for _, kind, _ in g.ops),
                 weights=f'{sum(v.size for v in model.values()):,}')
    facts["ratios"] = len(ratios())
    facts["nratios"] = len(ratios()) + len(LOW) + 1
    from compile_mlp import fmt, positions
    j = positions(model["W1"].shape[1]).index((5, 1))          # the neuron quoted on the page
    facts["eg_bias"] = fmt(float(model["b1"][j]))
    gif = ROOT / "site" / "fenet.gif"                  # the recording (record.py), which needs the page first
    head = gif.read_bytes()[6:10] if gif.exists() else (640).to_bytes(2, "little") + (400).to_bytes(2, "little")
    facts["gif_w"], facts["gif_h"] = int.from_bytes(head[:2], "little"), int.from_bytes(head[2:], "little")
    for test in (False, True):
        html, nr = page(g, test, facts)
        (ROOT / "site" / ("test.html" if test else "index.html")).write_text(html)
    print(f"site/index.html: {len((ROOT / 'site' / 'index.html').read_text()) / 1000:.0f} kB, {len(g.ops)} network primitives, {nr} ratios above 1")


if __name__ == "__main__":
    main()
