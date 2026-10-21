"""Compile the trained network (model.npz) into an SVG filter graph, and emulate that graph in NumPy.

Coordinates are device pixels: the page draws the 14x14 input as one-pixel cells inside a group
scaled by 1/devicePixelRatio, so the filter's user units are device pixels at any zoom.

Layout of every feature image: channel i is a 14x14 tile at (0, i*T). A layer is one
feConvolveMatrix per output channel; its kernel reaches every input tile at once (offset i*T),
computes the output at the origin tile, and an feOffset moves it to its own tile. The tiles are
summed onto a black background, so the gaps read as zero padding.
"""
from __future__ import annotations

import numpy as np

N = 14
T = 17                      # tile period: 14 cells + 3 cells of zero padding (largest dilation)
TAPS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


# ------------------------------------------------------------------ the graph
class Graph:
    def __init__(self, width, height):
        self.ops = []           # (name, kind, params)
        self.W, self.H = width, height

    def add(self, kind, **p):
        name = f"r{len(self.ops)}"
        self.ops.append((name, kind, p))
        return name


def kernel_from_offsets(taps: dict):
    """{(u, v): weight} -> (orderX, orderY, targetX, targetY, flat kernel) in feConvolveMatrix
    convention: result(X,Y) = sum source(X - tx + j, Y - ty + i) * K[(oy-1-i)*ox + (ox-1-j)]."""
    us = [u for u, v in taps] + [0]
    vs = [v for u, v in taps] + [0]
    umin, umax, vmin, vmax = min(us), max(us), min(vs), max(vs)
    ox, oy = umax - umin + 1, vmax - vmin + 1
    tx, ty = -umin, -vmin
    K = np.zeros(ox * oy)
    for (u, v), w in taps.items():
        j, i = u + tx, v + ty
        K[(oy - 1 - i) * ox + (ox - 1 - j)] += w
    return ox, oy, tx, ty, K


def build(model, alpha):
    C1, C2, C3 = [int(c) for c in model["C"]]
    dil = [int(d) for d in model["DIL"]]
    H = max(C1, C2, C3) * T + 8
    g = Graph(24, H)
    black = g.add("flood", color=0.0, region=(0, 0, g.W, g.H))
    prev, cin = "SourceGraphic", 1
    for K, b, cout, d in (("K1", "b1", C1, dil[0]), ("K2", "b2", C2, dil[1]), ("K3", "b3", C3, dil[2])):
        Wk, bk = model[K], model[b]
        acc = black
        for o in range(cout):
            taps = {}
            for c in range(cin):
                for t, (dy, dx) in enumerate(TAPS):
                    w = float(Wk[c * 9 + t, o])
                    if w != 0.0:
                        taps[(dx * d, c * T + dy * d)] = w
            conv = g.add("conv", src=prev, taps=taps, bias=float(bk[o]), region=(0, 0, N, N))
            moved = g.add("offset", src=conv, dx=0, dy=o * T, region=(0, o * T, N, N)) if o else conv
            acc = g.add("sum", a=acc, b=moved, region=(0, 0, g.W, g.H))
        prev, cin = acc, cout
    # global average pool: the mean of tile p lands at (0, p*T)
    pool = g.add("conv", src=prev, taps={(u, v): 1 / (N * N) for u in range(N) for v in range(N)}, bias=0.0,
                 region=(0, 0, 1, C3 * T))
    one = g.add("flood", color=1.0, region=(0, C3 * T, 1, 1))          # a constant 1 for the class biases
    # an opaque background where the scores will be computed: a convolution keeps its source's alpha
    back = g.add("flood", color=0.0, region=(0, 0, 44, C3 * T + 1))
    pool0 = g.add("sum", a=back, b=pool, region=(0, 0, 44, C3 * T + 1))
    pool1 = g.add("sum", a=pool0, b=one, region=(0, 0, 44, C3 * T + 1))
    # class scores at (4k+1, 1): s_k = 0.5 + alpha * (W m + c); the 0.5 and the bias come through the
    # constant pixel, so every other pixel of the row stays 0
    taps = {}
    for k in range(10):
        for p in range(C3):
            taps[(-(4 * k + 1), p * T - 1)] = alpha * float(model["W"][p, k])
        taps[(-(4 * k + 1), C3 * T - 1)] = 0.5 + alpha * float(model["c"][k])
    scores = g.add("conv", src=pool1, taps=taps, bias=0.0, region=(0, 1, 40, 1))
    best = g.add("dilate", src=scores, rx=40, ry=0, region=(0, 1, 40, 1))
    diff = g.add("diff", a=scores, b=best, region=(0, 1, 40, 1))
    onehot = g.add("iszero", src=diff, region=(0, 1, 40, 1))
    # the winner as a 3x3 block, readable even if the page lands a pixel off
    block = g.add("dilate", src=onehot, rx=1, ry=1, region=(0, 0, 40, 3))
    ink = g.add("dilate", src="SourceGraphic", rx=N, ry=N, region=(0, 0, 1, 1))
    inkb = g.add("offset", src=ink, dx=41, dy=1, region=(40, 0, 3, 3))
    inkblk = g.add("dilate", src=inkb, rx=1, ry=1, region=(40, 0, 3, 3))
    out = g.add("sum", a=block, b=inkblk, region=(0, 0, 44, 3))
    g.output = out
    g.meta = {"C": (C1, C2, C3), "dil": dil, "scores": scores}
    return g


# ------------------------------------------------------------------ emulator
def emulate(g, x, quant=True):
    """Run the graph on one 14x14 input; values outside a primitive's region are transparent (0)."""
    W, H = g.W + 40, g.H + 8
    img = {}
    src = np.zeros((H, W))
    src_a = np.zeros((H, W))
    src_a[:g.H, :g.W] = 1.0
    src[:N, :N] = x
    img["SourceGraphic"] = (src, src_a)

    def clip_region(v, a, region):
        x0, y0, w, h = region
        m = np.zeros((H, W), bool)
        m[y0:y0 + h, x0:x0 + w] = True
        return np.where(m, v, 0.0), np.where(m, a, 0.0)

    def q(v):
        return np.round(v * 255) / 255 if quant else v

    for name, kind, p in g.ops:
        if kind == "flood":
            v = np.full((H, W), p["color"]); a = np.ones((H, W))
        elif kind == "conv":
            sv, sa = img[p["src"]]
            v = np.zeros((H, W))
            for (u, w_), wt in p["taps"].items():
                sh = np.zeros((H, W))
                ys, xs = slice(max(0, -w_), min(H, H - w_)), slice(max(0, -u), min(W, W - u))
                yd, xd = slice(max(0, w_), min(H, H + w_)), slice(max(0, u), min(W, W + u))
                sh[ys, xs] = sv[yd, xd]
                v += wt * sh
            v = np.clip(v + p["bias"], 0, 1)
            a = sa.copy()                                  # preserveAlpha
            v = np.where(a > 0, v, 0)
        elif kind == "offset":
            sv, sa = img[p["src"]]
            v = np.zeros((H, W)); a = np.zeros((H, W))
            dx, dy = p["dx"], p["dy"]
            v[dy:, dx:] = sv[:H - dy, :W - dx]; a[dy:, dx:] = sa[:H - dy, :W - dx]
        elif kind == "sum":
            av, aa = img[p["a"]]; bv, ba = img[p["b"]]
            v = np.clip(av + bv, 0, 1); a = np.clip(aa + ba, 0, 1)
        elif kind == "dilate":
            sv, sa = img[p["src"]]
            from scipy import ndimage
            v = ndimage.maximum_filter(sv, size=(2 * p["ry"] + 1, 2 * p["rx"] + 1), mode="constant")
            a = ndimage.maximum_filter(sa, size=(2 * p["ry"] + 1, 2 * p["rx"] + 1), mode="constant")
        elif kind == "diff":
            av, aa = img[p["a"]]; bv, ba = img[p["b"]]
            v = np.abs(av - bv); a = np.maximum(aa, ba)
        elif kind == "iszero":
            sv, sa = img[p["src"]]
            v = (sv < 1 / 256).astype(float); a = np.ones((H, W))
        else:
            raise ValueError(kind)
        v, a = clip_region(v, a, p["region"])
        if kind in ("conv",):
            v = q(v)
        img[name] = (v, a)
    out = img[g.output][0]
    winners = [k for k in range(10) if out[1, 4 * k + 1] > 0.5]
    scores = img[g.meta["scores"]][0][1, [4 * k + 1 for k in range(10)]]
    return winners, scores, out[1, 41] > 0.5


# ------------------------------------------------------------------ SVG
def fmt(w):
    if w == 0:
        return "0"
    s = f"{w:.4g}"
    return s.replace("e-0", "e-").replace("0.", ".", 1) if s.startswith("0.") else (s.replace("-0.", "-.", 1) if s.startswith("-0.") else s)


def to_svg(g, fid="net"):
    out = [f'<filter id="{fid}" x="0" y="0" width="{g.W + 24}" height="{g.H}" filterUnits="userSpaceOnUse" '
           f'primitiveUnits="userSpaceOnUse" color-interpolation-filters="sRGB">']
    for name, kind, p in g.ops:
        x0, y0, w, h = p["region"]
        reg = f'x="{x0}" y="{y0}" width="{w}" height="{h}"'
        if kind == "flood":
            c = round(p["color"] * 255)
            out.append(f'<feFlood flood-color="rgb({c},{c},{c})" flood-opacity="1" {reg} result="{name}"/>')
        elif kind == "conv":
            ox, oy, tx, ty, K = kernel_from_offsets(p["taps"])
            out.append(f'<feConvolveMatrix in="{p["src"]}" order="{ox} {oy}" targetX="{tx}" targetY="{ty}" divisor="1" '
                       f'bias="{fmt(p["bias"])}" preserveAlpha="true" edgeMode="none" kernelMatrix="{" ".join(fmt(k) for k in K)}" {reg} result="{name}"/>')
        elif kind == "offset":
            out.append(f'<feOffset in="{p["src"]}" dx="{p["dx"]}" dy="{p["dy"]}" {reg} result="{name}"/>')
        elif kind == "sum":
            out.append(f'<feComposite in="{p["a"]}" in2="{p["b"]}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" {reg} result="{name}"/>')
        elif kind == "dilate":
            out.append(f'<feMorphology in="{p["src"]}" operator="dilate" radius="{p["rx"]} {p["ry"]}" {reg} result="{name}"/>')
        elif kind == "diff":
            out.append(f'<feBlend in="{p["a"]}" in2="{p["b"]}" mode="difference" {reg} result="{name}"/>')
        elif kind == "iszero":
            table = " ".join(["1"] + ["0"] * 255)
            out.append(f'<feComponentTransfer in="{p["src"]}" {reg} result="{name}"><feFuncR type="discrete" tableValues="{table}"/>'
                       f'<feFuncG type="discrete" tableValues="{table}"/><feFuncB type="discrete" tableValues="{table}"/>'
                       f'<feFuncA type="linear" slope="0" intercept="1"/></feComponentTransfer>')
    out.append("</filter>")
    return "\n".join(out)
