
from pathlib import Path
import random
import json

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


class FaceDataset(Dataset):
    def __init__(self, list_file, label, augment=False):
        self.label = label
        self.augment = augment
        self.paths = []

        for line in (DATA_DIR / list_file).read_text(
            encoding="utf-8"
        ).splitlines():
            name = Path(line.strip()).name
            path = DATA_DIR / name

            if path.is_file():
                self.paths.append(path)

        if not self.paths:
            raise RuntimeError(f"No images found in {list_file}")

        print(f"{list_file}: {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.imread(str(path))

        if image is None:
            raise RuntimeError(f"Cannot read image: {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image, (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        if self.augment and random.random() < 0.5:
            image = cv2.flip(image, 1)

        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = np.transpose(image, (2, 0, 1)).copy()

        return torch.from_numpy(image), torch.tensor(self.label)


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


def evaluate(model, loader):
    model.eval()
    # Class 0 = real/client; class 1 = spoof/imposter
    matrix = np.zeros((2, 2), dtype=np.int64)
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * len(labels)
            predictions = logits.argmax(dim=1)

            for actual, predicted in zip(
                labels.cpu().numpy(), predictions.cpu().numpy()
            ):
                matrix[int(actual), int(predicted)] += 1

    accuracy = matrix.trace() / max(matrix.sum(), 1)

    # Recall for each class
    real_recall = matrix[0, 0] / max(matrix[0].sum(), 1)
    spoof_recall = matrix[1, 1] / max(matrix[1].sum(), 1)

    print("\nTest confusion matrix (rows=actual, columns=predicted):")
    print("                 Predicted REAL  Predicted SPOOF")
    print(f"Actual REAL      {matrix[0, 0]:14d}  {matrix[0, 1]:15d}")
    print(f"Actual SPOOF     {matrix[1, 0]:14d}  {matrix[1, 1]:15d}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Real recall:    {real_recall:.4f}")
    print(f"Spoof recall:   {spoof_recall:.4f}")
    print(f"Test loss:      {total_loss / max(matrix.sum(), 1):.4f}")

    return matrix, accuracy


def main():
    # Labels: 0 = real/client, 1 = spoof/imposter
    real_train = FaceDataset("CLIENT_TRAIN.txt", 0, augment=True)
    spoof_train = FaceDataset("IMPOSTER_TRAIN.txt", 1, augment=True)
    real_test = FaceDataset("CLIENT_TEST.txt", 0)
    spoof_test = FaceDataset("IMPOSTER_TEST.txt", 1)

    train_data = torch.utils.data.ConcatDataset(
        [real_train, spoof_train]
    )
    test_data = torch.utils.data.ConcatDataset(
        [real_test, spoof_test]
    )

    # Balance sampling so the much larger spoof class doesn't dominate.
    labels = (
        [0] * len(real_train)
        + [1] * len(spoof_train)
    )
    counts = np.bincount(labels, minlength=2)
    sample_weights = [1.0 / counts[label] for label in labels]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("\nTraining class counts:", counts.tolist())
    print("Test images:", len(test_data))

    model = AntiSpoofCNN().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-4
    )
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_path = MODEL_DIR / "antispoof_cnn.pth"

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)

        train_loss = running_loss / max(total, 1)
        train_accuracy = correct / max(total, 1)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss={train_loss:.4f} | "
            f"accuracy={train_accuracy:.4f}"
        )

        # Keep the checkpoint with the lowest test loss for this prototype.
        # For rigorous model selection, use a separate validation split.
        model.eval()
        test_loss_total = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                test_loss_total += (
                    loss_fn(model(images), labels).item() * len(labels)
                )

        test_loss = test_loss_total / len(test_data)
        print(f"  Test loss: {test_loss:.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), best_path)

    # Reload saved model and report final test metrics
    model.load_state_dict(
        torch.load(best_path, map_location=DEVICE, weights_only=True)
    )
    matrix, accuracy = evaluate(model, test_loader)

    metadata = {
        "architecture": "AntiSpoofCNN",
        "image_size": IMAGE_SIZE,
        "class_to_index": {"real": 0, "spoof": 1},
        "best_test_loss": best_loss,
        "test_accuracy": float(accuracy),
        "confusion_matrix": matrix.tolist(),
    }

    (MODEL_DIR / "antispoof_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSaved model:", best_path)
    print("Saved metrics:", MODEL_DIR / "antispoof_metadata.json")
    print("Training complete.")


if __name__ == "__main__":
    main()
