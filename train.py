"""Train a digit classifier shaped for SVG filters, in plain NumPy.

Input: a 14x14 grid of ink cells (0 or 1), as drawn on the page.
Network (every layer is one feConvolveMatrix per output channel in the browser):
    a1 = clip01(conv3x3(x,  dilation 1) + b1)        C1 channels
    a2 = clip01(conv3x3(a1, dilation 2) + b2)        C2 channels
    a3 = clip01(conv3x3(a2, dilation 3) + b3)        C3 channels
    m  = mean of a3 over the grid                      C3 values
    z  = W m + c                                       10 class scores
clip01 is what a filter primitive does to its result anyway (a ReLU capped at 1). Activations are
rounded to 8 bits in the forward pass (straight-through gradients), as a browser may store them.

    python train.py            writes model.npz and prints validation accuracies
"""
from __future__ import annotations

import gzip
import pickle
import sys
import time

import numpy as np
from scipy import ndimage

N = 14
C1, C2, C3 = 8, 16, 24
DIL = (1, 2, 3)
rng = np.random.default_rng(0)


# ------------------------------------------------------------------ data
def load():
    """(train, validation, test) as (images 28x28, labels): 50,000, 10,000 and 10,000 digits.
    Training fits on train; every choice made from results (checkpoint, score scale, opening example)
    reads validation; test is read only by evaluate.py and the Chromium check, once the rest is fixed."""
    splits = pickle.load(gzip.open("data/mnist.pkl.gz"), encoding="latin1")
    return tuple((x.reshape(-1, 28, 28), y) for x, y in splits)


def to_grid(img28, thr):
    """28x28 grey -> 14x14 ink cells, like a stroke painted on the page's grid."""
    small = img28.reshape(N, 2, N, 2).mean(axis=(1, 3))
    return (small > thr).astype(np.float32)


def augment(img28, r):
    """Random size, slant, rotation, position and stroke weight, then the grid."""
    s = r.uniform(0.75, 1.3)
    rot = np.deg2rad(r.uniform(-14, 14))
    sh = r.uniform(-0.25, 0.25)
    A = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]) @ np.array([[1, sh], [0, 1]]) / s
    c = np.array([13.5, 13.5])
    shift = r.uniform(-4, 4, size=2)
    off = c - A @ (c + shift)
    out = ndimage.affine_transform(img28, A, offset=off, order=1, mode="constant")
    k = r.random()
    if k < 0.25:
        out = ndimage.grey_dilation(out, size=(2, 2))
    elif k < 0.35:
        out = ndimage.grey_erosion(out, size=(2, 2))
    g = to_grid(out, r.uniform(0.2, 0.45))
    if r.random() < 0.3:                       # gaps from a fast stroke
        holes = r.random(g.shape) < 0.08
        g[holes] = 0
    return g


def batch_grid(X, idx, r, aug=True):
    return np.stack([augment(X[i], r) if aug else to_grid(X[i], 0.3) for i in idx])[:, None]


# ------------------------------------------------------------------ layers
def taps(dil):
    return [(dy * dil, dx * dil) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def im2col(a, dil):
    B, C, H, W = a.shape
    p = dil
    ap = np.pad(a, ((0, 0), (0, 0), (p, p), (p, p)))
    cols = np.stack([ap[:, :, p + dy:p + dy + H, p + dx:p + dx + W] for dy, dx in taps(dil)], axis=2)  # B,C,9,H,W
    return cols.transpose(0, 3, 4, 1, 2).reshape(B * H * W, C * 9)


def col2im(dcols, shape, dil):
    B, C, H, W = shape
    p = dil
    d = dcols.reshape(B, H, W, C, 9).transpose(0, 3, 4, 1, 2)
    out = np.zeros((B, C, H + 2 * p, W + 2 * p), dtype=np.float32)
    for t, (dy, dx) in enumerate(taps(dil)):
        out[:, :, p + dy:p + dy + H, p + dx:p + dx + W] += d[:, :, t]
    return out[:, :, p:p + H, p:p + W]


def q8(v):
    return np.round(v * 255) / 255


class Net:
    def __init__(self):
        def he(cout, cin):
            return (rng.normal(size=(cin * 9, cout)) * np.sqrt(2 / (cin * 9))).astype(np.float32)
        self.p = {"K1": he(C1, 1), "b1": np.full(C1, 0.0, np.float32),
                  "K2": he(C2, C1), "b2": np.zeros(C2, np.float32),
                  "K3": he(C3, C2), "b3": np.zeros(C3, np.float32),
                  "W": (rng.normal(size=(C3, 10)) * 0.5).astype(np.float32), "c": np.zeros(10, np.float32)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x, quant=True):
        cache = []
        a = x
        for i, (K, b, C, d) in enumerate([("K1", "b1", C1, DIL[0]), ("K2", "b2", C2, DIL[1]), ("K3", "b3", C3, DIL[2])]):
            B, _, H, W = a.shape
            cols = im2col(a, d)
            z = cols @ self.p[K] + self.p[b]
            z = z.reshape(B, H, W, C).transpose(0, 3, 1, 2)
            out = np.clip(z, 0, 1)
            if quant:
                out = q8(out)
            cache.append((a.shape, cols, z, d, K))
            a = out
        mean = a.mean(axis=(2, 3))
        if quant:
            mean = q8(mean)
        logits = mean @ self.p["W"] + self.p["c"]
        return logits, (cache, a, mean)

    def step(self, x, y, lr):
        logits, (cache, a3, mean) = self.forward(x)
        B = x.shape[0]
        z = logits - logits.max(1, keepdims=True)
        pr = np.exp(z)
        pr /= pr.sum(1, keepdims=True)
        loss = -np.log(pr[np.arange(B), y] + 1e-9).mean()
        dl = pr
        dl[np.arange(B), y] -= 1
        dl /= B
        g = {"W": mean.T @ dl, "c": dl.sum(0)}
        dmean = dl @ self.p["W"].T
        da = np.repeat(np.repeat(dmean[:, :, None, None], N, 2), N, 3) / (N * N)
        for (shape, cols, zz, d, K), b in zip(reversed(cache), ("b3", "b2", "b1")):
            dz = da * ((zz > 0) & (zz < 1))
            Bc, C, H, W = dz.shape
            dzf = dz.transpose(0, 2, 3, 1).reshape(-1, C)
            g[K] = cols.T @ dzf
            g[b] = dzf.sum(0)
            da = col2im(dzf @ self.p[K].T, shape, d)
        self.t += 1
        for k in self.p:
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        return loss

    def predict(self, x, quant=True):
        out = []
        for i in range(0, len(x), 500):
            out.append(self.forward(x[i:i + 500], quant)[0].argmax(1))
        return np.concatenate(out)


def main():
    (X, y), (Xv, yv), _ = load()
    r = np.random.default_rng(1)
    clean_val = batch_grid(Xv, range(len(Xv)), r, aug=False)
    aug_val = batch_grid(Xv, range(len(Xv)), np.random.default_rng(2), aug=True)
    net = Net()
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    B = 128
    for ep in range(epochs):
        t0 = time.time()
        lr = 3e-3 * (0.5 ** (ep // 4))
        perm = r.permutation(len(X))
        losses = []
        for i in range(0, len(X) - B + 1, B):
            idx = perm[i:i + B]
            xb = batch_grid(X, idx, r, aug=r.random() < 0.85)
            losses.append(net.step(xb, y[idx], lr))
        acc_c = (net.predict(clean_val) == yv).mean()
        acc_a = (net.predict(aug_val) == yv).mean()
        print(f"epoch {ep + 1}: loss {np.mean(losses):.3f}, val clean {acc_c:.4f}, val augmented {acc_a:.4f} ({time.time() - t0:.0f}s)", flush=True)
        np.savez("model.npz", **net.p, C=np.array([C1, C2, C3]), DIL=np.array(DIL))


if __name__ == "__main__":
    main()
