"""The perceptron the page runs, in NumPy, and the inputs it sees.

    x  = blur(ink)                       fixed 3x3 blur of the 14x14 drawing, one feConvolveMatrix
    h1 = clip01(W1 x + b1)               H1 units: each one feConvolveMatrix (14x14 = 196 taps) writing one pixel
    h2 = clip01(W2 h1 + b2)              H2 units, the same way over the 14x14 image of h1
    z  = W3 h2 + b3                      10 scores, the same way; the page shows the largest

clip01 is the clamp every filter primitive applies; activations are rounded to 8 bits in the forward
pass with straight-through gradients. Training is scripts/train_mlp.py.
"""
from __future__ import annotations

import os

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

import fenet.train as T

N = 14
H1, H2 = int(os.environ.get("H1", 192)), int(os.environ.get("H2", 96))
BLUR = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16


def blur(x):
    """x: (B, 14, 14) -> (B, 196), zero padding, 8-bit."""
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    out = sum(BLUR[i, j] * p[:, i:i + N, j:j + N] for i in range(3) for j in range(3))
    return T.q8(np.clip(out, 0, 1)).reshape(len(x), -1)


def thin_augment(img28, r):
    """A one-cell-wide stroke, as a mouse paints it on the grid: transform, skeleton, pool, gaps."""
    s = r.uniform(0.75, 1.3)
    rot = np.deg2rad(r.uniform(-14, 14))
    sh = r.uniform(-0.25, 0.25)
    A = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]) @ np.array([[1, sh], [0, 1]]) / s
    c = np.array([13.5, 13.5])
    off = c - A @ (c + r.uniform(-4, 4, size=2))
    out = ndimage.affine_transform(img28, A, offset=off, order=1, mode="constant")
    sk = skeletonize(out > 0.3)
    if r.random() < 0.3:
        sk = ndimage.binary_dilation(sk, np.ones((2, 2), bool))
    g = sk.reshape(N, 2, N, 2).max(axis=(1, 3)).astype(np.float32)
    if r.random() < 0.3:
        g[r.random(g.shape) < 0.06] = 0
    return g


class MLP:
    def __init__(self, seed=0):
        r = np.random.default_rng(seed)
        self.p = {"W1": (r.normal(size=(N * N, H1)) * np.sqrt(2 / (N * N)) * 0.7).astype(np.float32), "b1": np.full(H1, 0.05, np.float32),
                  "W2": (r.normal(size=(H1, H2)) * np.sqrt(2 / H1) * 0.7).astype(np.float32), "b2": np.full(H2, 0.05, np.float32),
                  "W3": (r.normal(size=(H2, 10)) * np.sqrt(1 / H2)).astype(np.float32), "b3": np.zeros(10, np.float32)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, xb):
        z1 = xb @ self.p["W1"] + self.p["b1"]
        a1 = T.q8(np.clip(z1, 0, 1))
        z2 = a1 @ self.p["W2"] + self.p["b2"]
        a2 = T.q8(np.clip(z2, 0, 1))
        return a2 @ self.p["W3"] + self.p["b3"], (z1, a1, z2, a2)

    def step(self, xb, y, lr, wd=1e-5):
        logits, (z1, a1, z2, a2) = self.forward(xb)
        B = len(y)
        e = np.exp(logits - logits.max(1, keepdims=True))
        pr = e / e.sum(1, keepdims=True)
        loss = -np.log(pr[np.arange(B), y] + 1e-9).mean()
        d = pr; d[np.arange(B), y] -= 1; d /= B
        g = {"W3": a2.T @ d, "b3": d.sum(0)}
        d2 = (d @ self.p["W3"].T) * ((z2 > 0) & (z2 < 1))
        g["W2"], g["b2"] = a1.T @ d2, d2.sum(0)
        d1 = (d2 @ self.p["W2"].T) * ((z1 > 0) & (z1 < 1))
        g["W1"], g["b1"] = xb.T @ d1, d1.sum(0)
        self.t += 1
        for k in self.p:
            gk = g[k] + (wd * self.p[k] if k.startswith("W") else 0)
            self.m[k] = 0.9 * self.m[k] + 0.1 * gk
            self.v[k] = 0.999 * self.v[k] + 0.001 * gk ** 2
            self.p[k] -= lr * (self.m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8)
        return loss

    def predict(self, xb):
        return np.concatenate([self.forward(xb[i:i + 1000])[0].argmax(1) for i in range(0, len(xb), 1000)])


def thin14(g):
    return skeletonize(g > 0).astype(np.float32)


def thin28(img):
    sk = skeletonize(img > 0.3)
    return sk.reshape(14, 2, 14, 2).max(axis=(1, 3)).astype(np.float32)
