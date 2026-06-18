# backend/architectures.py
import torch.nn as nn
import segmentation_models_pytorch as smp


class DeepLabV3PlusModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )

    def forward(self, x):
        return self.model(x)


class FPNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.FPN(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )

    def forward(self, x):
        return self.model(x)


class UNetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )

    def forward(self, x):
        return self.model(x)
