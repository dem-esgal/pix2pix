import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

IM_SIZE = 256
LR = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
NUM_WORKERS = 2
L1_LAMBDA = 100
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "discriminator.pt"
CHECKPOINT_GEN = "generator.pt"
CHECKPOINT_SAVE_STEP = 5

both_transform = A.Compose(
    [A.Resize(width=IM_SIZE, height=IM_SIZE)], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.14),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)
