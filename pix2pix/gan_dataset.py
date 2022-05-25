import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import pix2pix.config as config


class PairImageDataset(Dataset):
    def __init__(self, root_dir, image_half):
        self.root_dir = root_dir
        self.image_half = image_half
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :self.image_half, :]
        target_image = image[:, self.image_half:, :]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image
