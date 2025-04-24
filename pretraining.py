from models.generator import Generator
from dataloader import SRDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision
import os
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def calculate_metrics(sr, hr):
    sr_img = sr.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    hr_img = hr.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    sr_img = np.clip(sr_img, 0, 1)
    hr_img = np.clip(hr_img, 0, 1)

    psnr_value = psnr(hr_img, sr_img, data_range=1)
    # ssim_value = ssim(hr_img, sr_img, multichannel=True, data_range=1)
    ssim_value = ssim(hr_img, sr_img, data_range=1, channel_axis=-1, win_size=min(7, min(hr_img.shape[0], hr_img.shape[1])))

    return psnr_value, ssim_value

def pretrain_generator(config, device):
    generator = Generator().to(device)
    generator.train()

    train_dir    = config['data']['train_dir']
    crop_size    = config['data']['crop_size']
    upscale      = config['data']['upscale_factor']
    batch_size   = config['training']['batch_size']

    dataset = SRDataset(train_dir, crop_size, upscale)
    print(f"[DEBUG] Found {len(dataset)} samples in {train_dir}")
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty please check your train_dir & subfolder names")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=config['generator'].get('pretrain_lr', 1e-4), betas=(0.9, 0.999))

    save_path = config.get("pretrain_checkpoint_path", "checkpoints/")
    os.makedirs(save_path, exist_ok=True)

    psnr_list = []
    ssim_list = []
    loss_list = []

    for epoch in range(1, config.get('pretrain_epochs', 10) + 1):
        loop = tqdm(dataloader, leave=True)
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        for lr_imgs, hr_imgs in loop:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            optimizer.zero_grad()
            sr_imgs = generator(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Just use first image in batch for metrics to save time
            p, s = calculate_metrics(sr_imgs[0:1], hr_imgs[0:1])
            epoch_psnr += p
            epoch_ssim += s

            loop.set_description(f"[Pretrain Epoch {epoch}]")
            loop.set_postfix(pixel_loss=f"{loss.item():.4f}", psnr=f"{p:.2f}", ssim=f"{s:.4f}")

        num_batches = len(dataloader)
        loss_list.append(epoch_loss / num_batches)
        psnr_list.append(epoch_psnr / num_batches)
        ssim_list.append(epoch_ssim / num_batches)

        if epoch % 100 == 0 or epoch == config.get('pretrain_epochs', 10):
            torch.save({
                "generator": generator.state_dict(),
                "g_optim": optimizer.state_dict(),
                "epoch": epoch
            }, os.path.join(save_path, f"pretrained_generator_epoch_{epoch}.pt"))

    # Plot training metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(loss_list, label="MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(psnr_list, label="PSNR", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("PSNR over Epochs")
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(ssim_list, label="SSIM", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("SSIM over Epochs")
    plt.grid()

    plt.tight_layout()
    plt.savefig("pretraining_metrics.png")
    plt.show()

    print("Pretraining completed.")
    print(f" Final PSNR: {psnr_list[-1]:.2f}, SSIM: {ssim_list[-1]:.4f}")
    print("Saved graph as pretraining_metrics.png")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrain_generator(config, device)
