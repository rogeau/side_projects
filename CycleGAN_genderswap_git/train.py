import torch
from dataset import MaleFemaleDataset
from utils import save_checkpoint, load_checkpoint, plot_losses
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
import os
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

def train_fn(disc_M, disc_F, gen_M, gen_F, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch, G_losses, D_losses, total_iterations):
    loop = tqdm(loader, leave=True)


    for idx, (male, female) in enumerate(loop):

        real_idx = idx + len(loader) * (epoch)
        male = male.to(config.DEVICE)
        female = female.to(config.DEVICE)

        with torch.amp.autocast("cuda"):
            fake_female = gen_F(male)
            D_F_real = disc_F(female)
            D_F_fake = disc_F(fake_female.detach())
            D_F_real_loss = mse(D_F_real, torch.ones_like(D_F_real))
            D_F_fake_loss = mse(D_F_fake, torch.zeros_like(D_F_fake))
            D_F_loss = D_F_real_loss + D_F_fake_loss

            fake_male = gen_M(female)
            D_M_real = disc_M(male)
            D_M_fake = disc_M(fake_male.detach())
            D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real))
            D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake))
            D_M_loss = D_M_real_loss + D_M_fake_loss

            D_loss = D_F_loss / 2 + D_M_loss / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.amp.autocast("cuda"):
            D_M_fake = disc_M(fake_male)
            D_F_fake = disc_F(fake_female)
            
            loss_G_M = mse(D_M_fake, torch.ones_like(D_M_fake))
            loss_G_F = mse(D_F_fake, torch.ones_like(D_F_fake))

            cycle_male = gen_M(fake_female.detach())
            cycle_female = gen_F(fake_male.detach())
            cycle_male_loss = L1(male, cycle_male)
            cycle_female_loss = L1(female, cycle_female)

            identity_male = gen_M(male)
            identity_female = gen_F(female)
            identity_male_loss = L1(male, identity_male)
            identity_female_loss = L1(female, identity_female)

            G_loss = (
                loss_G_M + loss_G_F + cycle_male_loss * config.LAMBDA_CYCLE + cycle_female_loss * config.LAMBDA_CYCLE + identity_male_loss * config.LAMBDA_IDENTITY + identity_female_loss * config.LAMBDA_IDENTITY
            )
        
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if real_idx % config.SAVE_FREQUENCY == 0:
            os.makedirs(config.SAVED_OUTPUT, exist_ok=True)
            save_image(male*0.5+0.5, f"{config.SAVED_OUTPUT}{real_idx}_genF_input.png")
            save_image(fake_female*0.5+0.5, f"{config.SAVED_OUTPUT}{real_idx}_genF_output.png")
            save_image(female*0.5+0.5, f"{config.SAVED_OUTPUT}{real_idx}_genM_input.png")
            save_image(fake_male*0.5+0.5, f"{config.SAVED_OUTPUT}{real_idx}_genM_output.png")

            G_losses.append(G_loss.detach().item())
            D_losses.append(D_loss.detach().item())
            total_iterations.append(real_idx)
            plot_losses(G_losses, D_losses, total_iterations)
    return


def main():
    disc_M = Discriminator(in_channels=3).to(config.DEVICE)
    disc_F = Discriminator(in_channels=3).to(config.DEVICE)
    gen_M = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_F = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_M.parameters()) + list(disc_F.parameters()),
        lr = config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    opt_gen = optim.Adam(
        list(gen_M.parameters()) + list(gen_F.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    L1 = nn.L1Loss().to(config.DEVICE)
    mse = nn.MSELoss().to(config.DEVICE)

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_M, gen_M, opt_gen, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_F, gen_F, opt_gen, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_M, disc_M, opt_disc, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_F, disc_F, opt_disc, config.LEARNING_RATE
        )

    dataset = MaleFemaleDataset(
        root_male=config.TRAIN_DIR+"/male", root_female=config.TRAIN_DIR+"/female", transform=config.transforms
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    g_scaler = torch.amp.GradScaler("cuda")
    d_scaler = torch.amp.GradScaler("cuda")

    G_losses = []
    D_losses = []
    total_iterations = []

    if config.LOAD_MODEL: first_epoch = config.PREVIOUS_EPOCH + 1
    else: first_epoch = 0

    for epoch in range(first_epoch, config.NUM_EPOCHS):
        
        train_fn(disc_M, disc_F, gen_M, gen_F, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch, G_losses, D_losses, total_iterations)

        if config.SAVE_MODEL:
                save_checkpoint(gen_M, opt_gen, epoch, filename=config.CHECKPOINT_GEN_M)
                save_checkpoint(gen_F, opt_gen, epoch, filename=config.CHECKPOINT_GEN_F)
                save_checkpoint(disc_M, opt_disc, epoch, filename=config.CHECKPOINT_DISC_M)
                save_checkpoint(disc_F, opt_disc, epoch, filename=config.CHECKPOINT_DISC_F)

if __name__ == "__main__":
    main()