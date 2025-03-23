import torch
import config
import matplotlib.pyplot as plt
import os

def save_checkpoint(model, optimizer, epoch, filename="my_checkpoint.pth.tar"):
    saved_filename = f"epoch{epoch}_{filename}"

    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, saved_filename)

    previous_checkpoint = f"epoch{epoch-1}_{filename}"
    if os.path.exists(previous_checkpoint):
            os.remove(previous_checkpoint)


def load_checkpoint(checkpoint_file, model, optimizer=None, lr=None, epoch=config.PREVIOUS_EPOCH):
    print("=> Loading checkpoint")
    filename = f"epoch{epoch}_{checkpoint_file}"

    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and lr is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def plot_losses(G_losses, D_losses, iterations, saved_dir=config.SAVED_OUTPUT):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(iterations, G_losses)
    axes[0].set_title("Generator loss")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Loss")

    axes[1].plot(iterations, D_losses)
    axes[1].set_title("Discriminator loss")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Loss")

    os.makedirs(saved_dir, exist_ok=True)
    plt.savefig(f"{saved_dir}losses_iteration.png", dpi=100)


def rescale_frame(x, y, w, h, frame, scale_factor=config.SCALE_FACTOR):
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    x_center = x + w // 2
    y_center = y + h // 2
    x_new = max(0, x_center - new_w // 2)
    y_new = max(0, y_center - new_h // 2)
    x_new_end = min(frame.shape[1], x_new + new_w)
    y_new_end = min(frame.shape[0], y_new + new_h)
    return x_new, x_new_end, y_new, y_new_end, new_w, new_h