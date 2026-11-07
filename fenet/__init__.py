"""fenet's library: MNIST and Pendigits data, the perceptron, the network-to-filter compiler, the page's
readout geometry and the screenshot reader. The entry points are in scripts/, run from the repository
root as modules: python -m scripts.build_page."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# which trained network the checks use: mlp (the page's) or conv (fenet/convnet.py)
MODEL = os.environ.get("MODEL", "mlp")
WEIGHTS = ROOT / "model" / f"{MODEL}.npz"
ALPHA = ROOT / "model" / ("alpha.json" if MODEL == "mlp" else f"{MODEL}_alpha.json")
PAGE = ROOT / "site" / ("test.html" if MODEL == "mlp" else f"{MODEL}.html")
