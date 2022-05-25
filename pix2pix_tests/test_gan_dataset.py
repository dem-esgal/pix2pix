from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pix2pix.pair_image_dataset import PairImageDataset

def manual_test_gan_dataset():
    dataset = PairImageDataset("data/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        save_image(x, "image.png")
        save_image(y, "target.png")
        break
