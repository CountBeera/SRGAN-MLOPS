# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# import os
# import glob

# class SRDataset(Dataset):
#     def __init__(self, root_dir, crop_size=96, scale_factor=4):
#         super().__init__()
#         # root_dir should be ".../data/train"
#         hr_dir = os.path.join(root_dir, "HR", "hr_images")
#         lr_dir = os.path.join(root_dir, "LR", "lr_images")

#         # grab all .jpg in those exact folders
#         self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.jpg")))
#         self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.jpg")))

#         # sanity checks
#         if not os.path.isdir(hr_dir):
#             raise RuntimeError(f"HR directory not found: {hr_dir}")
#         if not os.path.isdir(lr_dir):
#             raise RuntimeError(f"LR directory not found: {lr_dir}")
#         if len(self.hr_paths) == 0:
#             raise RuntimeError(f"No HR images found in {hr_dir}")
#         if len(self.lr_paths) == 0:
#             raise RuntimeError(f"No LR images found in {lr_dir}")
#         if len(self.hr_paths) != len(self.lr_paths):
#             raise RuntimeError(
#                 f"Count mismatch: {len(self.hr_paths)} HR vs {len(self.lr_paths)} LR"
#             )

#         self.crop_size = crop_size
#         self.scale     = scale_factor

#         # -- ensure HR is always crop_size × crop_size
#         self.hr_transform = transforms.Compose([
#             transforms.RandomCrop((crop_size, crop_size)),
#             transforms.ToTensor()
#         ])
#         # -- ensure LR is always (crop_size/scale) × (crop_size/scale)
#         self.lr_transform = transforms.Compose([
#             transforms.Resize(
#                 (crop_size // self.scale, crop_size // self.scale),
#                 interpolation=Image.BICUBIC
#             ),
#             transforms.ToTensor()
#         ])

#     def __len__(self):
#         return len(self.hr_paths)

#     def __getitem__(self, idx):
#         hr_img = Image.open(self.hr_paths[idx]).convert("RGB")
#         lr_img = Image.open(self.lr_paths[idx]).convert("RGB")

#         hr_tensor = self.hr_transform(hr_img)      # → [3, 96, 96]
#         lr_tensor = self.lr_transform(lr_img)      # → [3, 24, 24]

#         return lr_tensor, hr_tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import os
import glob
import random

class SRDataset(Dataset):
    def __init__(self, root_dir, crop_size=96, scale_factor=4):
        super().__init__()
        self.hr_dir = os.path.join(root_dir, "HR", "hr_images")
        self.lr_dir = os.path.join(root_dir, "LR", "lr_images")

        self.hr_paths = sorted(glob.glob(os.path.join(self.hr_dir, "*.jpg")))
        self.lr_paths = sorted(glob.glob(os.path.join(self.lr_dir, "*.jpg")))

        if not self.hr_paths or not self.lr_paths:
            raise RuntimeError("No HR or LR images found.")
        if len(self.hr_paths) != len(self.lr_paths):
            raise RuntimeError(f"Mismatch: {len(self.hr_paths)} HR vs {len(self.lr_paths)} LR")

        self.crop_size = crop_size
        self.lr_crop_size = crop_size // scale_factor
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_paths[idx]).convert("RGB")
        lr_img = Image.open(self.lr_paths[idx]).convert("RGB")

        # Ensure both images are big enough
        hr_w, hr_h = hr_img.size
        lr_w, lr_h = lr_img.size

        if hr_w < self.crop_size or hr_h < self.crop_size:
            raise ValueError(f"HR image too small for crop: {hr_w}x{hr_h}")
        if lr_w < self.lr_crop_size or lr_h < self.lr_crop_size:
            raise ValueError(f"LR image too small for crop: {lr_w}x{lr_h}")

        # Sync crop: pick random top-left corner
        hr_x = random.randint(0, hr_w - self.crop_size)
        hr_y = random.randint(0, hr_h - self.crop_size)
        lr_x = hr_x // (hr_w // lr_w)
        lr_y = hr_y // (hr_h // lr_h)

        # Crop both HR and LR at aligned locations
        hr_patch = hr_img.crop((hr_x, hr_y, hr_x + self.crop_size, hr_y + self.crop_size))
        lr_patch = lr_img.crop((lr_x, lr_y, lr_x + self.lr_crop_size, lr_y + self.lr_crop_size))

        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)
