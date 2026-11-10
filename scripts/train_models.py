
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_nmf.models import MNISTMLP


# -----------------------------
# 1. Configuration
# -----------------------------

SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# 2. Reproducibility
# -----------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# 3. Load MNIST
# -----------------------------

def load_data():
    transform = transforms.ToTensor()

    data_dir = Path("data/raw")

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

    return train_dataset, test_dataset


# -----------------------------
# 4. Evaluate model
# -----------------------------

def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


# -----------------------------
# 5. Train one model
# -----------------------------

def train_model(
    activation,
    train_dataset,
    test_dataset,
    initial_state,
):
    # Reset seed to make data order reproducible
    set_seed(SEED)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = MNISTMLP(
        activation=activation
    ).to(DEVICE)

    # Use the same initial weights for both models
    model.load_state_dict(initial_state)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []

    print(f"\n===== Training {activation.upper()} =====")

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_dataset)

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
        )

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }

        history.append(epoch_result)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Test Loss: {test_loss:.4f} "
            f"Test Accuracy: {test_accuracy:.4%}"
        )

    training_time = time.time() - start_time

    # Save model weights
    model_dir = Path("reports/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        model_dir / f"{activation}_model.pth",
    )

    return {
        "activation": activation,
        "training_time_seconds": training_time,
        "history": history,
    }


# -----------------------------
# 6. Main
# -----------------------------

def main():
    set_seed(SEED)

    print("Device:", DEVICE)

    train_dataset, test_dataset = load_data()

    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))

    # Generate one common initial state
    set_seed(SEED)

    initial_model = MNISTMLP(activation="relu")

    initial_state = {
        key: value.detach().clone()
        for key, value in initial_model.state_dict().items()
    }

    results = {}

    for activation in ["relu", "tanh"]:
        results[activation] = train_model(
            activation,
            train_dataset,
            test_dataset,
            initial_state,
        )

    # Save experiment results
    metrics_dir = Path("reports/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(
        metrics_dir / "training_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\n===== Training Completed =====")

    for activation, result in results.items():
        final_accuracy = result["history"][-1]["test_accuracy"]

        print(
            f"{activation.upper()} "
            f"Final Accuracy: {final_accuracy:.4%}"
        )

        print(
            f"Training Time: "
            f"{result['training_time_seconds']:.2f}s"
        )


if __name__ == "__main__":
    main()
