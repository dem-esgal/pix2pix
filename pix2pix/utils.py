import torch
from torchvision.utils import save_image

import config


def generate_images(generator, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    generator.eval()
    with torch.no_grad():
        y_fake = generator(x)
        y_fake = (y_fake + 1) / 2  # to 0..1 range
        save_image(y_fake, folder + f"/generated_{epoch}.png")
        save_image((x + 1) / 2, folder + f"/input_{epoch}.png")
        save_image((y + 1) / 2, folder + f"/real_{epoch}.png")
    generator.train()


def save_checkpoint(model, optimizer, filename="checkpoint.pt"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
