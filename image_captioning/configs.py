import torch

# Dataset parameters
PATH_IMG, PATH_CAPTIONS = 'dataset/Images', 'dataset/captions.txt'
IMG_SIZE = 224

# Transformer parameters
EMBED_DIM = 1024
MAX_LENGTH = 50
BLOCK_NUMBER = 12

# Training parameters
DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
EPOCHS = 15