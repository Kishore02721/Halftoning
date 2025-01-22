import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import UNet  # Import the UNet model
from preprocessing import ImageDataset  # Import ImageDataset from preprocessing.py
from torchvision import transforms

# Hyperparameters and Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# Preprocessing Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Dataset and DataLoader from preprocessing.py
train_dataset = ImageDataset("dataset/train/unclear", "dataset/train/clear", transform)
val_dataset = ImageDataset("dataset/val/unclear", "dataset/val/clear", transform)
test_dataset = ImageDataset("dataset/test/unclear", "dataset/test/clear", transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, and Optimizer
model = UNet().to(device)
criterion = nn.MSELoss()  # Mean Squared Error Loss for image reconstruction
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for unclear_images, clear_images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            unclear_images, clear_images = unclear_images.to(device), clear_images.to(device)
            
            optimizer.zero_grad()
            outputs = model(unclear_images)
            loss = criterion(outputs, clear_images)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader)}")
        
        # Validation step (optional)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for unclear_images, clear_images in val_loader:
                unclear_images, clear_images = unclear_images.to(device), clear_images.to(device)
                outputs = model(unclear_images)
                loss = criterion(outputs, clear_images)
                val_loss += loss.item()
        print(f"Validation Loss after Epoch {epoch+1}: {val_loss / len(val_loader)}")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

