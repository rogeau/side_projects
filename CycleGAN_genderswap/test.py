import torch
from torchvision.utils import save_image
from generator_model import Generator
from PIL import Image
import os
import numpy as np
import config
from utils import load_checkpoint
from torch.utils.data import DataLoader
from dataset import TestOneGenderDataset

TEST_DIR = "test_images/"

TEST_IMG_PATHS = [os.path.join(TEST_DIR, basename) for basename in os.listdir(TEST_DIR)]

def test_model_with_dataloader(model, test_img_paths, batch_size):
    dataset = TestOneGenderDataset(test_img_paths, transform=config.transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.to(config.DEVICE)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(config.DEVICE)
            fake_test = model(batch)
            fake_path, _ = os.path.splitext(test_img_paths[i])
            save_image(fake_test * 0.5 + 0.5, f"{fake_path}_fake.png")

# Example usage

gen_M = Generator(img_channels=3)  
load_checkpoint("256genm_batch1.pth.tar", gen_M)
test_model_with_dataloader(gen_M, TEST_IMG_PATHS, batch_size=config.BATCH_SIZE)