import torch
import torch.nn as nn
from torchvision.models import vgg19
import torch.nn.functional as F
# from torchvision.models import mobilenet_v2

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features[:36].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.MSELoss()

    def forward(self, generated, target):
        generated = nn.functional.interpolate(generated, size=(224, 224), mode='bilinear', align_corners=False)
        target = nn.functional.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)
        return self.criterion(self.vgg(generated), self.vgg(target))
import torch
import torch.nn as nn

# class MobileNetPerceptualLoss(nn.Module):
#     def __init__(self, layer_index: int = 13):
#         super().__init__()
#         # Load the pretrained MobileNetV2 feature extractor
#         mobilenet = mobilenet_v2(pretrained=True).features
#         # Take layers 0 through layer_index (inclusive) → gives you a 14×14 feature map by default
#         self.features = nn.Sequential(*list(mobilenet.children())[: layer_index + 1]).eval()
#         # Freeze all weights
#         for p in self.features.parameters():
#             p.requires_grad = False
#         self.criterion = nn.MSELoss()

#     def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         # Resize both to 224×224
#         generated = F.interpolate(generated, size=(224, 224), mode='bilinear', align_corners=False)
#         target    = F.interpolate(target,    size=(224, 224), mode='bilinear', align_corners=False)
#         # (optional) normalize to MobileNet’s training stats
#         # mean = torch.tensor([0.485, 0.456, 0.406], device=generated.device)[None,:,None,None]
#         # std  = torch.tensor([0.229, 0.224, 0.225], device=generated.device)[None,:,None,None]
#         # generated = (generated - mean) / std
#         # target    = (target    - mean) / std

#         # Extract features and compute MSE
#         f_gen = self.features(generated)
#         f_tgt = self.features(target)
#         return self.criterion(f_gen, f_tgt)

# # --- instantiation on your GPU ---
# # perceptual_loss_fn = MobileNetPerceptualLoss(layer_index=13).to(device)
