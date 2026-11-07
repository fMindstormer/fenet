"""Compare trained models on validation digits: python -m scripts.compare_models a.npz b.npz ..."""
import sys
import numpy as np
import fenet.train as T
from fenet.mlp import MLP, blur, thin28, thin_augment
_, (Xv, yv), _ = T.load()
clean = np.stack([T.to_grid(Xv[i], 0.3) for i in range(len(Xv))])
sk = np.stack([thin28(Xv[i]) for i in range(len(Xv))])
r = np.random.default_rng(3); ta = np.stack([thin_augment(Xv[i], r) for i in range(len(Xv))])
for f in sys.argv[1:]:
    net = MLP(); net.p = dict(np.load(f))
    print(f, " ".join(f"{n} {(net.predict(blur(G)) == yv).mean():.4f}" for n, G in (("clean", clean), ("skeleton", sk), ("thin+transform", ta))))
