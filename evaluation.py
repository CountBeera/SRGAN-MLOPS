import os
import yaml
import torch
import random
import lpips
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.metrics import structural_similarity as structural_similarity_index_measure, peak_signal_noise_ratio
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
# Evaluation
# -----------------------------
def evaluate_srgan(generator, val_loader, checkpoint_path, device, num_samples=5, visualize=True):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    print(f"✅ Loaded generator from checkpoint: {checkpoint_path}")

    lpips_loss = lpips.LPIPS(net='alex').to(device)

    total_psnr, total_ssim, total_lpips = 0, 0, 0
    count = 0

    with torch.no_grad():
        for lr_img, hr_img in tqdm(val_loader, desc="🔍 Evaluating"):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            sr_img = generator(lr_img)

            sr_np = sr_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
            hr_np = hr_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)

            psnr = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
            ssim = structural_similarity_index_measure(hr_np, sr_np, channel_axis=2, data_range=1.0)

            sr_lpips = (sr_img * 2) - 1
            hr_lpips = (hr_img * 2) - 1
            lpips_score = lpips_loss(sr_lpips, hr_lpips).item()

            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips_score
            count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_lpips = total_lpips / count

    print(f"\n📊 Average Scores on Validation Set:")
    print(f"🔹 PSNR:  {avg_psnr:.2f}")
    print(f"🔹 SSIM:  {avg_ssim:.4f}")
    print(f"🔹 LPIPS: {avg_lpips:.4f} \n")

    if visualize:
        samples = random.sample(range(len(val_loader.dataset)), num_samples)
        fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axs = [axs]

        for i, idx in enumerate(samples):
            lr_img, hr_img = val_loader.dataset[idx]
            lr_tensor = lr_img.unsqueeze(0).to(device)
            sr_tensor = generator(lr_tensor)

            lr_np = lr_img.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            sr_np = sr_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
            hr_np = hr_img.permute(1, 2, 0).cpu().numpy().clip(0, 1)

            axs[i][0].imshow(lr_np)
            axs[i][0].set_title("Low-Res Input")
            axs[i][0].axis("off")

            axs[i][1].imshow(sr_np)
            axs[i][1].set_title("Super-Resolved")
            axs[i][1].axis("off")

            axs[i][2].imshow(hr_np)
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

    val_root = config['data']['val_dir']
    lr_dir = os.path.join(val_root, 'LR', 'lr_images')
    hr_dir = os.path.join(val_root, 'HR', 'hr_images')

    val_dataset = SRDataset(lr_dir=lr_dir, hr_dir=hr_dir, transform=transforms.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    checkpoint_path = os.path.join("checkpoints", "srgan_step_146800.pt")
    evaluate_srgan(generator, val_loader, checkpoint_path, device, num_samples=5)
