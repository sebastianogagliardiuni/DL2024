# -- Backbones --

import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
#import clip

# Load ResNet-50 from torchvision
def load_ResNet50():

    weights = ResNet50_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    return resnet50(weights=weights), preprocess

# Load VIT-B/16 from torchvision
def load_VITB16():

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    return vit_b_16(weights=weights), preprocess

"""
# Load CLIP from clip openai library
def load_CLIP(model_type, device):

    model, img_preprocess = clip.load(model_type, device)
    return model, img_preprocess
"""

# -- Custom Backbones + Feature extractor --

# ResNet-50
class FeaturesResNet50(torch.nn.Module):
    def __init__(self, resnet50):
        super().__init__()
        # ResNet-50 model
        self.model = resnet50

    # Forward returns last layer features
    # https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet50
    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        h_last = torch.flatten(x, 1)
        out = self.model.fc(h_last)

        return h_last, out

# Vit-B/16
class FeaturesViTB16(torch.nn.Module):
    def __init__(self, vit_b_16):
        super().__init__()
        # ResNet-50 model
        self.model = vit_b_16

    # Forward returns also last layer features
    # https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html#vit_b_16
    def forward(self, x):

        # Reshape and permute the input tensor
        x = self.model._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        h_last = self.model.encoder(x)
        # Classifier "token" as used by standard language architectures
        h_last = h_last[:, 0]
        out = self.model.heads(h_last)

        return h_last, out