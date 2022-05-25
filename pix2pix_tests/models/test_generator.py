import torch

from pix2pix.models.generator import Generator

def test_generator():
    x = torch.randn((1, 3, 512, 512))
    model = Generator()
    preds = model(x)
    print(preds.shape)
