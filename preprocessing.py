import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Dataset Class
class ImageDataset(Dataset):
    def __init__(self, unclear_dir, clear_dir, transform=None):
        """
        Args:
            unclear_dir (str): Path to the folder containing unclear images.
            clear_dir (str): Path to the folder containing clear images.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.unclear_dir = unclear_dir
        self.clear_dir = clear_dir
        self.transform = transform
        self.image_names = os.listdir(unclear_dir)  # Assumes paired images with the same name in both directories
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        unclear_path = os.path.join(self.unclear_dir, self.image_names[idx])
        clear_path = os.path.join(self.clear_dir, self.image_names[idx])

        unclear_image = Image.open(unclear_path).convert("RGB")
        clear_image = Image.open(clear_path).convert("RGB")

        if self.transform:
            unclear_image = self.transform(unclear_image)
            clear_image = self.transform(clear_image)

        return unclear_image, clear_image

# Preprocessing Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Datasets
train_dataset = ImageDataset("dataset/train/unclear", "dataset/train/clear", transform)
val_dataset = ImageDataset("dataset/val/unclear", "dataset/val/clear", transform)
test_dataset = ImageDataset("dataset/test/unclear", "dataset/test/clear", transform)

# DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Example: Iterate through the train DataLoader
for unclear_image, clear_image in train_loader:
    print(f"Unclear Image Batch Shape: {unclear_image.shape}")
    print(f"Clear Image Batch Shape: {clear_image.shape}")
    break

