# import yaml
# import torch
# from torch.utils.data import DataLoader
# from models.generator import Generator
# from models.discriminator import Discriminator
# from dataloader import SRDataset
# from trainer import train_srgan

# def load_config(path='config.yaml'):
#     with open(path, 'r') as f:
#         return yaml.safe_load(f)

# def main():
#     config = load_config()

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     generator = Generator().to(device)
#     discriminator = Discriminator().to(device)

#     dataset = SRDataset(config['data']['train_dir'], config['data']['crop_size'])
#     # print(f"[DEBUG] Found {len(dataset)} samples in {config['data']['train_dir']}")
#     # if len(dataset) == 0:
#     #     raise RuntimeError("Dataset is empty – please recheck your paths and folder structure")
#     dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)

#     train_srgan(generator, discriminator, dataloader, config, device)

# if __name__ == "__main__":
#     main()
# main.py
import yaml
import torch
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator
from dataloader import SRDataset
from trainer import train_srgan

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1) load settings
    config = load_config()
    train_dir    = config['data']['train_dir']
    crop_size    = config['data']['crop_size']
    upscale      = config['data']['upscale_factor']
    batch_size   = config['training']['batch_size']

    # 2) device~
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3) models
    generator     = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 4) dataset + debug check
    dataset = SRDataset(train_dir, crop_size, upscale)
    print(f"[DEBUG] Found {len(dataset)} samples in {train_dir}")
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty – please check your train_dir & subfolder names")

    # 5) dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    # 6) train
    train_srgan(generator, discriminator, dataloader, config, device)

if __name__ == "__main__":
    main()