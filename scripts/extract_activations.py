
from pathlib import Path

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_nmf.models import MNISTMLP


# -----------------------------
# 1. Configuration
# -----------------------------

BATCH_SIZE = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# 2. Load MNIST test dataset
# -----------------------------

def load_test_data():
    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST(
        root="data/raw",
        train=False,
        download=True,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return test_loader


# -----------------------------
# 3. Extract activations
# -----------------------------

def extract_features(model, loader):
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(DEVICE)

            # Extract hidden-layer activations
            features = model.forward_features(images)

            all_features.append(
                features.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

    activations = np.concatenate(
        all_features,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    return activations, labels


# -----------------------------
# 4. Calculate statistics
# -----------------------------

def calculate_statistics(activations, activation):

    stats = {
        "shape": list(activations.shape),
        "mean": float(np.mean(activations)),
        "std": float(np.std(activations)),
        "minimum": float(np.min(activations)),
        "maximum": float(np.max(activations)),
        "zero_ratio": float(
            np.mean(activations == 0)
        ),
        "near_zero_ratio": float(
            np.mean(np.abs(activations) < 1e-3)
        ),
        "negative_ratio": float(
            np.mean(activations < 0)
        ),
    }

    return stats


# -----------------------------
# 5. Main
# -----------------------------

def main():

    print("Device:", DEVICE)

    test_loader = load_test_data()

    output_dir = Path("reports/activations")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_statistics = {}

    for activation in ["relu", "tanh"]:

        print(f"\n===== {activation.upper()} =====")

        # Load trained model
        model = MNISTMLP(
            activation=activation
        ).to(DEVICE)

        model_path = Path(
            f"reports/models/{activation}_model.pth"
        )

        model.load_state_dict(
            torch.load(
                model_path,
                map_location=DEVICE,
            )
        )

        # Extract activations
        features, labels = extract_features(
            model,
            test_loader,
        )

        # Check dimensions
        print("Activation shape:", features.shape)
        print("Labels shape:", labels.shape)

        # Calculate statistics
        stats = calculate_statistics(
            features,
            activation,
        )

        all_statistics[activation] = stats

        for key, value in stats.items():
            print(f"{key}: {value}")

        # Save activation matrix and labels
        np.savez_compressed(
            output_dir / f"{activation}_activations.npz",
            activations=features,
            labels=labels,
        )

    # Save statistics
    with open(
        output_dir / "activation_statistics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_statistics,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\nActivation extraction completed.")


if __name__ == "__main__":
    main()
