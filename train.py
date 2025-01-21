import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import VDSR
from dataset import DullCrispDataset
from config import DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DULL_IMAGE_PATH, CRISP_IMAGE_PATH
import torchvision.transforms as transforms

# Dataset transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
dataset = DullCrispDataset(DULL_IMAGE_PATH, CRISP_IMAGE_PATH, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model, loss function, and optimizer
model = VDSR().to(DEVICE)
criterion = nn.MSELoss()  # MSE Loss for residuals
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    for i, (dull_images, crisp_images) in enumerate(dataloader):
        dull_images = dull_images.to(DEVICE)
        crisp_images = crisp_images.to(DEVICE)
        
        # Bicubic interpolation for the low-res image
        low_res_images = torch.nn.functional.interpolate(dull_images, scale_factor=0.25, mode='bicubic', align_corners=False)
        # Forward pass: Predict residual using the VDSR model
        residual = model(low_res_images)
        # Resize the residual to match the size of the target (crisp images)
        residual_resized = torch.nn.functional.interpolate(residual, size=crisp_images.size()[2:], mode='bicubic', align_corners=False)
        # Calculate ground truth residual
        low_res_images_resized = torch.nn.functional.interpolate(low_res_images, size=crisp_images.size()[2:], mode='bicubic', align_corners=False)
        residual_gt = crisp_images - low_res_images_resized
        # Calculate the loss (MSE loss for residuals)
        loss = criterion(residual_resized, residual_gt)

        # Backward pass
        optimizer.zero_grad()  # Zero the gradients before backward pass
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model weights

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), 'vdsr_model.pth')

