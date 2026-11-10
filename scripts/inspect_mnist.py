
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main():
    # 1. Set random seed
    torch.manual_seed(42)

    # 2. Define data transformation
    transform = transforms.ToTensor()

    # 3. Download MNIST dataset
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    # 4. Print dataset information
    print("===== MNIST Dataset Information =====")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    image, label = train_dataset[0]

    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    print(f"Pixel minimum: {image.min().item():.4f}")
    print(f"Pixel maximum: {image.max().item():.4f}")

    # 5. Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    images, labels = next(iter(train_loader))

    print("\n===== Batch Information =====")
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")

    # 6. Visualize 10 handwritten digits
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        image, label = train_dataset[i]

        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label}")
        ax.axis("off")

    fig.suptitle("MNIST Handwritten Digits")
    plt.tight_layout()

    # 7. Save figure
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_dir / "mnist_samples.png",
        dpi=150,
        bbox_inches="tight",
    )

    print("\nFigure saved to reports/figures/mnist_samples.png")

    plt.show()


if __name__ == "__main__":
    main()
