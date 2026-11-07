"""Train the perceptron with a convolutional layer in front (fenet/convnet.py) exactly as train_mlp.py
trains the perceptron: the same digits, augmentation, schedule and validation checkpoint.

    python -m scripts.train_conv [epochs] [out]    writes model/conv.npz (or out)
"""
from fenet.convnet import ConvNet
from scripts.train_mlp import main

if __name__ == "__main__":
    main(ConvNet, "model/conv.npz")
