
import os
import yaml
import torch
import matplotlib.pyplot as plt
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.generator import Generator


def super_resolve_image(generator, image_path, device, show_result=True):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    lr_img_tensor = transform(img).unsqueeze(0).to(device)

    generator.eval()  # <<=== SMALL 'g' here
    with torch.no_grad():
        sr_img_tensor = generator(lr_img_tensor)

    lr_img_np = lr_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
    sr_img_np = sr_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)

    if show_result:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(lr_img_np)
        axs[0].set_title("Original Low-Res")
        axs[0].axis("off")

        axs[1].imshow(sr_img_np)
        axs[1].set_title("Super-Resolved (SRGAN)")
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()
    else:
        return lr_img_np, sr_img_np

image_path = r'data\val\LR\lr_images\3.jpg'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# super_resolve_image(Generator, image_path, device)

# 1. Load model
generator = Generator().to(device)

# (Optional but important!) Load the trained weights
checkpoint_path = "checkpoints/srgan_step_147600.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
generator.load_state_dict(checkpoint['generator'])
generator.eval()  # Very important: eval mode!

# 2. Call your function correctly
super_resolve_image(generator, image_path, device)

