# utils.py
import torch
from PIL import Image

# Function to load an image
def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

# Function to visualize the output
def visualize_output(input_img, output_img, target_img):
    # You can use matplotlib to visualize images side-by-side
    import matplotlib.pyplot as plt
    plt.subplot(1, 3, 1)
    plt.imshow(input_img.permute(1, 2, 0).cpu().detach().numpy())
    plt.title("Input Image")
    
    plt.subplot(1, 3, 2)
    plt.imshow(output_img.permute(1, 2, 0).cpu().detach().numpy())
    plt.title("Output Image")
    
    plt.subplot(1, 3, 3)
    plt.imshow(target_img.permute(1, 2, 0).cpu().detach().numpy())
    plt.title("Target Image")
    
    plt.show()

