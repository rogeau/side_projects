import torch
import albumentations as A
from albumentations import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "CelebA_HQ"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 5
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 20
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_M = "256genm_batch1.pth.tar"
CHECKPOINT_GEN_F = "256genf_batch1.pth.tar"
CHECKPOINT_DISC_M = "256discm_batch1.pth.tar"
CHECKPOINT_DISC_F = "256discf_batch1.pth.tar"
SAVED_OUTPUT = "saved_images/"
SAVE_FREQUENCY = 800
PREVIOUS_EPOCH = 9
NUM_EPOCHS_CLASSIFIER = 20
SCALE_FACTOR = 2

transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(256, 256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ],
    additional_targets={"image0": "image"}
)