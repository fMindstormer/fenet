"""The perceptron with a convolutional layer in front, in shapes SVG filters can run.

    x  = blur(ink)                                  14x14, as for the perceptron
    c  = clip01(conv5x5(x) + bc)                    C = 4 channels, each one feConvolveMatrix (25 taps), zero padding
    m  = 3x3 max of c, kept at every other pixel    feMorphology dilate; 4 x 7 x 7 = 196 values
    f  = the four 7x7 maps interleaved into 14x14   channel k at (2i + k % 2, 2j + k // 2)
    then the perceptron's layers on f               H1, H2 and 10 units, as in fenet/mlp.py

Interleaving keeps the input of the dense layers a single 14x14 image, so they compile exactly like the
perceptron's. Activations are rounded to 8 bits, with straight-through gradients.
"""
from __future__ import annotations

import numpy as np

import fenet.train as T
from fenet.mlp import MLP, N

C, K = 4, 5                  # channels, kernel size
M = N // 2                   # pooled side


def im2col(x):
    """(B, 14, 14) -> (B * 196, 25): each pixel's 5x5 neighbourhood, zero outside."""
    B = len(x)
    xp = np.pad(x, ((0, 0), (K // 2, K // 2), (K // 2, K // 2)))
    cols = np.stack([xp[:, i:i + N, j:j + N] for i in range(K) for j in range(K)], axis=-1)
    return cols.reshape(B * N * N, K * K)


def load(path):
    """A trained network from its .npz: the perceptron, or this one if it has a convolutional layer."""
    m = dict(np.load(path))
    net = ConvNet() if "Wc" in m else MLP()
    net.p = m
    return net


class ConvNet(MLP):
    def __init__(self, seed=0):
        super().__init__(seed)
        r = np.random.default_rng(seed + 100)
        self.p["Wc"] = (r.normal(size=(K * K, C)) * np.sqrt(2 / (K * K))).astype(np.float32)
        self.p["bc"] = np.zeros(C, np.float32)
        for k in ("Wc", "bc"):
            self.m[k] = np.zeros_like(self.p[k]); self.v[k] = np.zeros_like(self.p[k])

    def front(self, xb):
        """(B, 196) blurred drawings -> (B, 196) interleaved pooled channels, and what backward needs."""
        B = len(xb)
        cols = im2col(xb.reshape(B, N, N))
        zc = (cols @ self.p["Wc"] + self.p["bc"]).reshape(B, N, N, C).transpose(0, 3, 1, 2)     # B, C, 14, 14
        ac = T.q8(np.clip(zc, 0, 1))
        ap = np.pad(ac, ((0, 0), (0, 0), (1, 1), (1, 1)))
        win = np.stack([ap[:, :, a:a + N:2, b:b + N:2] for a in range(3) for b in range(3)], axis=-1)   # B, C, 7, 7, 9
        arg = win.argmax(-1)
        pooled = np.take_along_axis(win, arg[..., None], -1)[..., 0]
        f = np.zeros((B, N, N), pooled.dtype)
        for k in range(C):
            f[:, k // 2::2, k % 2::2] = pooled[:, k]
        return f.reshape(B, -1), (cols, zc, arg)

    def forward(self, xb):
        return self.dense(self.front(xb)[0])

    def step(self, xb, y, lr, wd=1e-5):
        f, (cols, zc, arg) = self.front(xb)
        loss, g, d1 = self.backward(f, y)
        B = len(xb)
        df = (d1 @ self.p["W1"].T).reshape(B, N, N)
        dpool = np.stack([df[:, k // 2::2, k % 2::2] for k in range(C)], axis=1)               # B, C, 7, 7
        dap = np.zeros((B, C, N + 2, N + 2), dpool.dtype)
        a, b = np.divmod(arg, 3)
        bi, ci, ii, jj = np.indices(arg.shape)
        np.add.at(dap, (bi, ci, 2 * ii + a, 2 * jj + b), dpool)
        dz = dap[:, :, 1:N + 1, 1:N + 1] * ((zc > 0) & (zc < 1))
        dz = dz.transpose(0, 2, 3, 1).reshape(B * N * N, C)
        g["Wc"], g["bc"] = cols.T @ dz, dz.sum(0)
        self.update(g, lr, wd)
        return loss
