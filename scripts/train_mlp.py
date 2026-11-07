"""Train the perceptron the page runs (fenet/mlp.py), in NumPy.

Training digits: MNIST's 50,000, and on average PEN (0.3) of each batch drawn from the pen paths of
Pendigits' training writers (fenet/pendigits.py), whose styles MNIST lacks: 9s with a curled tail, 1s with a
long upstroke. The checkpoint is the epoch with the best sum of four validation accuracies: MNIST
clean, augmented and thin, and Pendigits validation.

    python -m scripts.train_mlp [epochs] [out]    writes model/mlp.npz (or out): the epoch with the best
                                                  validation accuracy
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import fenet.pendigits as PD
import fenet.train as T
from fenet.mlp import MLP, blur, thin_augment

PEN = float(os.environ.get("PEN", 0.3))


def batch(X, idx, r, aug):
    if not aug:
        return T.batch_grid(X, idx, r, aug=False)[:, 0]
    return np.stack([thin_augment(X[i], r) if r.random() < 0.4 else T.augment(X[i], r) for i in idx])


def main():
    (X, y), (Xv, yv), _ = T.load()
    r = np.random.default_rng(1)
    clean = blur(T.batch_grid(Xv, range(len(Xv)), r, aug=False)[:, 0])
    aug = blur(T.batch_grid(Xv, range(len(Xv)), np.random.default_rng(2), aug=True)[:, 0])
    r3 = np.random.default_rng(3)
    thin = blur(np.stack([thin_augment(Xv[i], r3) for i in range(len(Xv))]))
    pen = PD.split("train")
    Gp, yp = PD.load("val")
    vals = [(clean, yv), (aug, yv), (thin, yv), (blur(Gp), yp)]
    net = MLP()
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    out = sys.argv[2] if len(sys.argv) > 2 else "model/mlp.npz"
    B = 128
    best = 0
    for ep in range(epochs):
        t0 = time.time()
        lr = 2e-3 * (0.5 ** (ep // 8))
        perm = r.permutation(len(X))
        losses = []
        for i in range(0, len(X) - B + 1, B):
            idx = perm[i:i + B]
            gb, yb = batch(X, idx, r, aug=r.random() < 0.85), y[idx].copy()
            k = r.binomial(B, PEN)               # the first k become Pendigits drawings
            for j, q in enumerate(r.integers(len(pen), size=k)):
                gb[j], yb[j] = PD.augment(pen[q][1], r), pen[q][0]
            losses.append(net.step(blur(gb), yb, lr))
        acc = [(net.predict(v) == t).mean() for v, t in vals]
        print(f"epoch {ep + 1}: loss {np.mean(losses):.3f}, val clean {acc[0]:.4f}, augmented {acc[1]:.4f}, thin {acc[2]:.4f}, "
              f"pen {acc[3]:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if sum(acc) > best:                      # the checkpoint is chosen on validation, never test
            best, best_ep = sum(acc), ep + 1
            np.savez(out, **net.p)
    print(f"saved epoch {best_ep} to {out}")


if __name__ == "__main__":
    main()
