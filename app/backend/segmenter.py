from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "9-pranet-full" / "best_pranet-full_model.pth"


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
    ):

        super().__init__()

        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        return self.relu(x)


class RFBModified(nn.Module):
    def __init__(self, in_channel, out_channel):

        super().__init__()

        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.conv_cat = nn.Conv2d(out_channel * 3, out_channel, 3, padding=1)

    def forward(self, x):

        x0 = self.branch0(x)

        x1 = self.branch1(x)

        x2 = self.branch2(x)

        x_cat = torch.cat([x0, x1, x2], dim=1)

        return self.conv_cat(x_cat)


class Aggregation(nn.Module):
    def __init__(self, channel):

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channel * 3, channel * 2, 3, padding=1),
            nn.BatchNorm2d(channel * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 2, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 1),
        )

    def forward(self, x1, x2, x3):

        x2 = F.interpolate(x2, size=x1.shape[2:], mode="bilinear", align_corners=False)

        x3 = F.interpolate(x3, size=x1.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([x1, x2, x3], dim=1)

        return self.conv(x)


class PraNet(nn.Module):
    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            "res2net50_26w_4s", pretrained=True, features_only=True
        )

        self.rfb2 = RFBModified(512, 32)

        self.rfb3 = RFBModified(1024, 32)

        self.rfb4 = RFBModified(2048, 32)

        self.agg = Aggregation(32)

    def forward(self, x):

        feats = self.backbone(x)

        # if not hasattr(self, "_printed"):
        #    for i, f in enumerate(feats):
        #        print(i, f.shape)
        #    self._printed = True

        x2 = feats[2]
        x3 = feats[3]
        x4 = feats[4]

        x2_rfb = self.rfb2(x2)

        x3_rfb = self.rfb3(x3)

        x4_rfb = self.rfb4(x4)

        lateral_map_5 = self.agg(x2_rfb, x3_rfb, x4_rfb)

        lateral_map_5 = F.interpolate(
            lateral_map_5, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        lateral_map_4 = lateral_map_5

        lateral_map_3 = lateral_map_5

        lateral_map_2 = lateral_map_5

        return (lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2)


class PraNetSegmenter:
    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = PraNet()

        checkpoint = torch.load(MODEL_PATH, map_location=self.device)

        self.model.load_state_dict(checkpoint, strict=False)

        self.model.to(self.device)

        self.model.eval()

    def predict_mask(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        image = cv2.resize(image, (256, 256))

        x = image.astype(np.float32) / 255.0

        x = torch.from_numpy(x).permute(2, 0, 1)

        x = x.unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, _, _, pred = self.model(x)

            pred = torch.sigmoid(pred)

        pred = pred.squeeze().cpu().numpy()

        binary_mask = (pred > 0.5).astype(np.uint8)

        return binary_mask

    def create_overlay(self, rgb_img, mask, alpha=0.4):
        overlay = rgb_img.copy()

        color_mask = np.zeros_like(rgb_img)

        color_mask[:, :, 1] = mask * 255

        overlay = cv2.addWeighted(rgb_img, 1 - alpha, color_mask, alpha, 0)

        return overlay

    def predict_and_overlay(self, image):

        mask = self.predict_mask(image)

        image = cv2.resize(image, (256, 256))

        overlay = self.create_overlay(image, mask)

        return mask, overlay
