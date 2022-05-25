import torch

from pix2pix.models.discriminator import Discriminator


def test_model():
    x = torch.randn((1, 3, 512, 1200))
    y = torch.randn((1, 3, 512, 1200))
    model = Discriminator(in_channels=3)
    preds = model(x, y)
    print(model)
    print(preds.shape)