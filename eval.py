import os
import yaml
import torch
import matplotlib.pyplot as plt
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.generator import Generator


# -----------------------------
# Config loader
# -----------------------------
def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# -----------------------------
# Custom dataset for SRGAN
# -----------------------------
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        lr_img = self.transform(lr_img)
        hr_img = self.transform(hr_img)

        return lr_img, hr_img


# -----------------------------
# Evaluation and visualization
# -----------------------------
def evaluate_srgan(generator, val_loader, checkpoint_path, device, num_samples=5):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    print(f"✅ Loaded generator from checkpoint: {checkpoint_path}")

    # Pick random samples from validation set
    samples = random.sample(range(len(val_loader.dataset)), num_samples)

    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axs = [axs]  # Ensure iterable if only 1 sample

    with torch.no_grad():
        for i, idx in enumerate(samples):
            lr_img, hr_img = val_loader.dataset[idx]
            lr_img_tensor = lr_img.unsqueeze(0).to(device)

            sr_img_tensor = generator(lr_img_tensor)

            # Convert tensors to numpy images for visualization
            lr_img_np = lr_img.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            hr_img_np = hr_img.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            sr_img_np = sr_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)

            axs[i][0].imshow(lr_img_np)
            axs[i][0].set_title("Low-Res Input")
            axs[i][0].axis("off")

            axs[i][1].imshow(sr_img_np)
            axs[i][1].set_title("Super-Resolved (SRGAN)")
            axs[i][1].axis("off")

            axs[i][2].imshow(hr_img_np)
            axs[i][2].set_title("High-Res Ground Truth")
            axs[i][2].axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    config = load_config('config.yaml')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)

    # Setup validation paths
    val_root = config['data']['val_dir']
    lr_dir = os.path.join(val_root, 'LR', 'lr_images')
    hr_dir = os.path.join(val_root, 'HR', 'hr_images')

    val_dataset = SRDataset(lr_dir=lr_dir, hr_dir=hr_dir, transform=transforms.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    checkpoint_path = "checkpoints\srgan_step_123000.pt"
    evaluate_srgan(generator, val_loader, checkpoint_path, device, num_samples=5)
