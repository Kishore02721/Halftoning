# config.py
import torch

# Device configuration (use GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

# Dataset paths
DULL_IMAGE_PATH = 'dataset/dull'
CRISP_IMAGE_PATH = 'dataset/crisp'

