

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
# from models.losses import VGGPerceptualLoss
from models.losses import VGGPerceptualLoss
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import pandas as pd




# ──────────────────────────────────────────────────────────────
# Total Variation Loss
class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, img):
        batch_size = img.size(0)
        h_tv = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return (h_tv + w_tv) / batch_size
# ──────────────────────────────────────────────────────────────

def train_srgan(generator, discriminator, dataloader, config, device, save_path="checkpoints/"):
    generator.train()
    discriminator.train()
    psnr_vals, ssim_vals = [], []
    best_psnr = -float("inf")
    best_ssim = -float("inf")

    g_optim = torch.optim.Adam(
        generator.parameters(),
        lr=config['generator']['learning_rate'],
        betas=(0.9, 0.999)
    )
    d_optim = torch.optim.Adam(
        discriminator.parameters(),
        lr=config['discriminator']['learning_rate'],
        betas=(0.9, 0.999)
    )

    adversarial_loss = nn.BCELoss()
    pixel_loss_fn = nn.MSELoss()
    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    tv_loss_fn = TotalVariationLoss().to(device)

    plt.ion()
    fig, ax = plt.subplots()
    steps, d_vals, g_vals = [], [], []
    line_d, = ax.plot([], [], label='Discriminator Loss')
    line_g, = ax.plot([], [], label='Generator Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.canvas.manager.window.setWindowTitle('SRGAN Training Loss')

    step = 0
    if config.get("pretrained_generator"):
        print(f"Loading pretrained generator from {config['pretrained_generator']}")
        pretrained_state = torch.load(config["pretrained_generator"])
        if "generator" in pretrained_state:
            generator.load_state_dict(pretrained_state["generator"])
        else:
            generator.load_state_dict(pretrained_state)
    if config.get("resume_from"):
        checkpoint = torch.load(config["resume_from"])
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        g_optim.load_state_dict(checkpoint["g_optim"])
        d_optim.load_state_dict(checkpoint["d_optim"])
        step = checkpoint["step"]
        print(f"Resumed from step {step}")

    for epoch in range(1, config['training']["epochs"] + 1):
        loop = tqdm(dataloader, leave=True)
        for lr_imgs, hr_imgs in loop:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # === Train Discriminator ===
            fake_hr = generator(lr_imgs)
            real_labels = torch.ones((lr_imgs.size(0), 1), device=device)
            fake_labels = torch.zeros((lr_imgs.size(0), 1), device=device)

            d_real = discriminator(hr_imgs)
            d_fake = discriminator(fake_hr.detach())

            d_loss_real = adversarial_loss(d_real, real_labels)
            d_loss_fake = adversarial_loss(d_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # === Train Generator ===
            fake_hr = generator(lr_imgs)
            g_adv = discriminator(fake_hr)

            pixel_loss = pixel_loss_fn(fake_hr, hr_imgs)
            adv_loss = adversarial_loss(g_adv, real_labels)
            perc_loss = perceptual_loss_fn(fake_hr, hr_imgs)
            tv_loss = tv_loss_fn(fake_hr)

            g_loss = (
                config["loss_weights"]["pixel"] * pixel_loss +
                config["loss_weights"]["adversarial"] * adv_loss +
                config["loss_weights"]["perceptual"] * perc_loss +
                config["loss_weights"].get("tv", 0) * tv_loss
            )

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            
            # Compute PSNR & SSIM
            with torch.no_grad():
                fake_np = fake_hr.detach().cpu().numpy()
                hr_np = hr_imgs.detach().cpu().numpy()
                batch_psnr, batch_ssim = 0.0, 0.0

                for i in range(fake_np.shape[0]):
                    sr_img = np.clip(fake_np[i].transpose(1, 2, 0), 0, 1)
                    hr_img = np.clip(hr_np[i].transpose(1, 2, 0), 0, 1)
                    batch_psnr += psnr(hr_img, sr_img, data_range=1.0)
                    batch_ssim += ssim(hr_img, sr_img, data_range=1.0, channel_axis=-1)

                avg_psnr = batch_psnr / fake_np.shape[0]
                avg_ssim = batch_ssim / fake_np.shape[0]
                psnr_vals.append(avg_psnr)
                ssim_vals.append(avg_ssim)



            # TQDM update
            loop.set_description(f"Epoch [{epoch}/{config['training']['epochs']}]")
            loop.set_postfix(
                d_loss=f"{d_loss.item():.4f}",
                g_loss=f"{g_loss.item():.4f}",
                PSNR=f"{avg_psnr:.2f}",
                SSIM=f"{avg_ssim:.4f}"
            )
            

            # metrics_df = pd.DataFrame({
            #     "step": steps,
            #     "discriminator_loss": d_vals,
            #     "generator_loss": g_vals,
            #     "psnr": psnr_vals,
            #     "ssim": ssim_vals
            # })
            # metrics_df.to_csv(os.path.join(save_path, "training_metrics.csv"), index=False)
            # Ensure all lists have the same length
            # if len(steps) == len(d_vals) == len(g_vals) == len(psnr_vals) == len(ssim_vals):
            #     metrics_df = pd.DataFrame({
            #         "step": steps,
            #         "discriminator_loss": d_vals,
            #         "generator_loss": g_vals,
            #         "psnr": psnr_vals,
            #         "ssim": ssim_vals
            #     })
            #     metrics_df.to_csv(os.path.join(save_path, "training_metrics.csv"), index=False)
            # else:
            #     print(f"Warning: Data lists have different lengths: {len(steps)}, {len(d_vals)}, {len(g_vals)}, {len(psnr_vals)}, {len(ssim_vals)}")



            # Live plot update
            d_val = d_loss.item()
            g_val = g_loss.item()
            steps.append(step)
            d_vals.append(d_val)
            g_vals.append(g_val)

            line_d.set_data(steps, d_vals)
            line_g.set_data(steps, g_vals)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)

            # Checkpoint
            step += 1
            if step % config['training']["save_interval"] == 0:
                os.makedirs(save_path, exist_ok=True)
                torch.save({
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "step": step
                }, os.path.join(save_path, f"srgan_step_{step}.pt"))
            # Save best PSNR model
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "step": step,
                    "metric": "psnr",
                    "value": best_psnr
                }, os.path.join(save_path, f"srgan_best_psnr.pt"))

            # Save best SSIM model
            if avg_ssim > best_ssim:
                best_ssim = avg_ssim
                torch.save({
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "step": step,
                    "metric": "ssim",
                    "value": best_ssim
                }, os.path.join(save_path, f"srgan_best_ssim.pt"))


    plt.ioff()
    plt.show()
    print("Training complete. Final plot displayed.")
