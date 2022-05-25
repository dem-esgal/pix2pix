import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from pix2pix.pair_image_dataset import PairImageDataset
from pix2pix.models.discriminator import Discriminator
from pix2pix.models.generator import Generator
from utils import save_checkpoint, load_checkpoint, generate_images

torch.backends.cudnn.benchmark = True


def train_fn(discriminator, generator, loader, discriminator_optimizer, generator_optimizer, l1_loss, bce,
             amp_generator_scaler, amp_discriminator_scaler):
    for idx, (x, y) in enumerate(tqdm(loader, leave=True)):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Discriminator training.
        with torch.cuda.amp.autocast():
            y_fake = generator(x)
            D_real = discriminator(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = discriminator(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        discriminator.zero_grad()
        amp_discriminator_scaler.scale(D_loss).backward()
        amp_discriminator_scaler.step(discriminator_optimizer)
        amp_discriminator_scaler.update()

        # Generator training
        with torch.cuda.amp.autocast():
            D_fake = discriminator(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        generator_optimizer.zero_grad()
        amp_generator_scaler.scale(G_loss).backward()
        amp_generator_scaler.step(generator_optimizer)
        amp_generator_scaler.update()


def train():
    generator = Generator().to(config.DEVICE)
    discriminator = Discriminator().to(config.DEVICE)

    generator_optimizer = optim.Adam(generator.parameters(), lr=config.LR, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=config.LR, betas=(0.5, 0.999), )

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, generator, generator_optimizer, config.LR,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, discriminator, discriminator_optimizer, config.LR,
        )

    train_dataset = PairImageDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    amp_generator_scaler = torch.cuda.amp.GradScaler()
    amp_discriminator_scaler = torch.cuda.amp.GradScaler()
    val_dataset = PairImageDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            discriminator, generator, train_loader, discriminator_optimizer, generator_optimizer, l1_loss, bce_loss,
            amp_generator_scaler, amp_discriminator_scaler,
        )

        if config.SAVE_MODEL and epoch % config.CHECKPOINT_SAVE_STEP == 0:
            save_checkpoint(generator, generator_optimizer, filename=config.CHECKPOINT_GEN)
            save_checkpoint(discriminator, discriminator_optimizer, filename=config.CHECKPOINT_DISC)

        generate_images(generator, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    train()
