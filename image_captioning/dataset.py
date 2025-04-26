import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import configs

class Flickr8kDataset(Dataset):
    def __init__(self, caption_file = configs.PATH_CAPTIONS, img_dir = configs.PATH_IMG, transform=None, codebook=None):
        self.img_dir = img_dir
        self.image_captions = self._load_captions(caption_file)
        self.transform = transform
        self.codebook = codebook

    def _load_captions(self, caption_file):
        img2cap = []
        with open(caption_file, 'r') as f:
            next(f)
            for line in f:
                line = line.strip()
                if not line: continue
                img_id, caption = line.split(',', 1)
                img2cap.append((img_id, caption))
        return img2cap
    
    def __len__(self):
        return len(self.image_captions)
    
    
    def __getitem__(self, idx):
        img_filename, caption = self.image_captions[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        
        image = Image.open(img_path).convert("RGB")

        if self.transform and self.codebook:
            tensor_image = self.transform(image)

            tokenized_caption, mask = tokenize(caption, self.codebook)

            return tensor_image, tokenized_caption, mask, idx
        
        else:
            return image, caption

def build_vocab(dataset):
    vocab = {'[SOS]': 0, '[PAD]': 1, '[UNK]': 2, '[EOS]': 3}
    idx = 4
    
    for _, caption in dataset:
        caption = caption.replace('"', '')
        tokens = [token.lower() for token in caption.split()]
        for token in tokens:
            if token not in vocab:
                vocab[token] = idx
                idx += 1

    return vocab

def tokenize(caption, codebook, max_length=50):
    caption = caption.replace('"', '')
    tokens = [token.lower() for token in caption.split()]
    input_ids = [codebook['[SOS]']] + [codebook.get(token, codebook['[UNK]']) for token in tokens] + [codebook['[EOS]']]
    input_ids = input_ids[:max_length] + [codebook['[PAD]']] * max(0, max_length - len(input_ids))
    attention_mask = [1 if id != codebook['[PAD]'] else 0 for id in input_ids]

    return torch.tensor(input_ids), torch.tensor(attention_mask)

def detokenize(input_ids, codebook):
    id_to_token = {v: k for k, v in codebook.items()}
    
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    tokens = []

    for i in input_ids:
        token = id_to_token[i]
        if token == '[EOS]':
            break
        if token not in ['[PAD]', '[SOS]']:
            tokens.append(token)

    return ' '.join(tokens)

def resize_and_pad(image, target_size=configs.IMG_SIZE):
    width, height = image.size
    scale = target_size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = F.resize(image, (new_height, new_width))
    pad_w = target_size - new_width
    pad_h = target_size - new_height
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    image = F.pad(image, padding, fill=0)
    return image

def val_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_and_pad(img, target_size=configs.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def train_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_and_pad(img, target_size=configs.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    
if __name__ == "__main__":
    dataset = Flickr8kDataset()
    print(build_vocab(dataset))