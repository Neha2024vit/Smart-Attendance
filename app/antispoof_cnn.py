
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

IMAGE_SIZE = 96
DEVICE = torch.device("cpu")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "antispoof_cnn.pth"


class AntiSpoofCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


_model = None


def load_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}"
            )

        model = AntiSpoofCNN().to(DEVICE)
        state = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()
        _model = model

    return _model


def predict_antispoof(face_image):
    """
    Input: OpenCV face crop (BGR image).
    Output: dictionary with predicted class and probabilities.

    Class 0 = REAL
    Class 1 = SPOOF
    """
    if face_image is None or face_image.size == 0:
        raise ValueError("Empty face image supplied to anti-spoof model")

    image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    image = np.transpose(image, (2, 0, 1)).copy()

    tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    model = load_model()

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    real_probability = float(probabilities[0])
    spoof_probability = float(probabilities[1])

    return {
        "label": "REAL" if real_probability >= spoof_probability else "SPOOF",
        "real_probability": real_probability,
        "spoof_probability": spoof_probability,
    }
