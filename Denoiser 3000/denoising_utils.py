import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

def denoise_image(image, model, device) -> Image:
    image_path = './rgb.png'  
    image = Image.open(image_path).convert('RGB')  # Ensure the image is RGB

    transform_full = transforms.Compose([
        transforms.ToTensor()
    ])

    original_image = transform_full(image)

    image = original_image.unsqueeze(0).to(device)  # Add batch dimension

    model.eval()  
    with torch.no_grad(): 
        reconstructed_image = model(image)

    reconstructed_image = reconstructed_image.squeeze().cpu()

    original_image = transforms.ToPILImage()(original_image)
    reconstructed_image = transforms.ToPILImage()(reconstructed_image)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(reconstructed_image)
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
