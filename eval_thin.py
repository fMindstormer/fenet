"""How well does the model read thin strokes, like a one-cell brush drawn with a mouse? (validation digits)"""
import numpy as np
from skimage.morphology import skeletonize
import train as T
from train_mlp import MLP, blur

def thin14(g):
    return skeletonize(g > 0).astype(np.float32)

def thin28(img):
    sk = skeletonize(img > 0.3)
    return sk.reshape(14, 2, 14, 2).max(axis=(1, 3)).astype(np.float32)

if __name__ == "__main__":
    _, (Xv, yv), _ = T.load()
    n = 5000
    net = MLP(); net.p = dict(np.load("mlp.npz"))
    clean = np.stack([T.to_grid(Xv[i], 0.3) for i in range(n)])
    variants = {"clean": clean, "skeleton at 14x14": np.stack([thin14(g) for g in clean]),
                "skeleton at 28x28, max-pooled": np.stack([thin28(Xv[i]) for i in range(n)])}
    for name, G in variants.items():
        print(f"{name:32s} accuracy {(net.predict(blur(G)) == yv[:n]).mean():.4f}   mean ink cells {G.sum((1, 2)).mean():.1f}")
