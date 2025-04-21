from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch

class MaleFemaleDataset(Dataset):
    def __init__(self, root_male, root_female, transform=None, size=None):
        super().__init__()
        self.root_male = root_male
        self.root_female = root_female
        self.transform = transform
        self.male_images = os.listdir(root_male)
        self.female_images = os.listdir(root_female)
        self.male_len = len(self.male_images)
        self.female_len = len(self.female_images)
        self.length_dataset = max(self.male_len, self.female_len)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        male_img = self.male_images[index % self.male_len]
        female_img = self.female_images[index % self.female_len]
        
        male_path = os.path.join(self.root_male, male_img)
        female_path = os.path.join(self.root_female, female_img)

        male_img = np.array(Image.open(male_path).convert("RGB"))
        female_img = np.array(Image.open(female_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=male_img, image0=female_img)
            male_img = augmentations["image"]
            female_img = augmentations["image0"]
        
        return male_img, female_img
    
class TestOneGenderDataset(Dataset):
    def __init__(self, root_imgs, transform=None):
        self.root_imgs = root_imgs
        self.transform = transform

    def __len__(self):
        return len(self.root_imgs)

    def __getitem__(self, idx):
        img = Image.open(self.root_imgs[idx]).convert("RGB")
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img
    

class TestOneGenderDetectWebcam(Dataset):
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.transform:
            img = self.transform(image=self.imgs)["image"]
        return img
    

class ClassifierDataset(Dataset):
    def __init__(self, root_male, root_female, transform=None, size=None):
        super().__init__()
        self.root_male = root_male
        self.root_female = root_female
        self.transform = transform
        self.male_images = os.listdir(root_male)
        self.female_images = os.listdir(root_female)

    def __len__(self):
        return len(self.male_images) + len(self.female_images)
    
    def __getitem__(self, index):
        if index < len(self.male_images):
            image_path = os.path.join(self.root_male, self.male_images[index])
            label = 0
        else:
            index -= len(self.male_images)
            image_path = os.path.join(self.root_female, self.female_images[index])
            label = 1  

        image = np.array(Image.open(image_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label