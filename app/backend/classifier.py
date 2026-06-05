from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "13-densenet-121"
    / "best_densenet121.pt"
)

CLASS_NAMES = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis"
]


class DenseNetClassifier:

    def __init__(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = timm.create_model(
            "densenet121",
            pretrained=False,
            num_classes=8
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device
        )

        checkpoint = {
            k: v
            for k, v in checkpoint.items()
            if "total_ops" not in k
            and "total_params" not in k
        }

        self.model.load_state_dict(
            checkpoint,
            strict=False
        )

        self.model.to(self.device)
        self.model.eval()

        self.tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image):

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        x = self.tfms(image)
        x = x.unsqueeze(0).to(self.device)

        with torch.no_grad():

            logits = self.model(x)

            probs = torch.softmax(
                logits,
                dim=1
            )

            confidence, idx = torch.max(
                probs,
                dim=1
            )

        return {
            "class_name":
                CLASS_NAMES[idx.item()],
            "confidence":
                confidence.item()
        }
        
    def is_polyp(self, image, threshold=0.8):
        
        POLYPS_CLASSES = [
            "dyed-lifted-polyps",
            "polyps"
        ]

        result = self.predict(image)

        return (
            result["class_name"] in POLYPS_CLASSES
            and
            result["confidence"] >= threshold
        )