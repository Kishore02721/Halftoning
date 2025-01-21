# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class DullCrispDataset(Dataset):
    def __init__(self, dull_dir, crisp_dir, transform=None, image_size=(256, 256)):
        self.dull_dir = dull_dir
        self.crisp_dir = crisp_dir
        self.dull_images = sorted(os.listdir(dull_dir))
        self.crisp_images = sorted(os.listdir(crisp_dir))
        self.transform = transform
        self.image_size = image_size  # Add a target image size

    def __len__(self):
        return len(self.dull_images)

    def __getitem__(self, idx):
        dull_img_path = os.path.join(self.dull_dir, self.dull_images[idx])
        crisp_img_path = os.path.join(self.crisp_dir, self.crisp_images[idx])

        dull_img = Image.open(dull_img_path).convert('RGB')
        crisp_img = Image.open(crisp_img_path).convert('RGB')

        # Resize images to the target size
        dull_img = dull_img.resize(self.image_size, Image.BICUBIC)
        crisp_img = crisp_img.resize(self.image_size, Image.BICUBIC)

        if self.transform:
            dull_img = self.transform(dull_img)
            crisp_img = self.transform(crisp_img)

        return dull_img, crisp_img

