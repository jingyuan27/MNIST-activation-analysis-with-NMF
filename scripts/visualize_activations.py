
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1. Load activation matrices
# -----------------------------

DATA_DIR = Path("reports/activations")
OUTPUT_DIR = Path("reports/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(activation):
    data = np.load(
        DATA_DIR / f"{activation}_activations.npz"
    )

    return data["activations"], data["labels"]


# -----------------------------
# 2. Plot activation heatmaps
# -----------------------------

def plot_heatmaps(relu, tanh):

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6)
    )

    axes[0].imshow(
        relu[:200],
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )

    axes[0].set_title("ReLU Activation Heatmap")
    axes[0].set_xlabel("Hidden Neurons")
    axes[0].set_ylabel("Test Samples")
    fig.colorbar(
        axes[0].images[0], ax=axes[0]
    )

    axes[1].imshow(
        tanh[:200],
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )

    axes[1].set_title("Tanh Activation Heatmap")
    axes[1].set_xlabel("Hidden Neurons")
    axes[1].set_ylabel("Test Samples")
    fig.colorbar(
        axes[1].images[0], ax=axes[1]
    )

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "activation_heatmaps.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# -----------------------------
# 3. Plot activation distributions
# -----------------------------

def plot_distributions(relu, tanh):

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5)
    )

    axes[0].hist(
        relu.flatten(),
        bins=100,
        density=True,
    )

    axes[0].set_title("ReLU Activation Distribution")
    axes[0].set_xlabel("Activation Value")
    axes[0].set_ylabel("Density")

    axes[1].hist(
        tanh.flatten(),
        bins=100,
        density=True,
    )

    axes[1].set_title("Tanh Activation Distribution")
    axes[1].set_xlabel("Activation Value")
    axes[1].set_ylabel("Density")

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "activation_distributions.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# -----------------------------
# 4. Calculate class-wise means
# -----------------------------

def calculate_class_means(activations, labels):

    class_means = []

    for digit in range(10):

        class_activations = activations[
            labels == digit
        ]

        mean_activation = class_activations.mean(
            axis=0
        )

        class_means.append(mean_activation)

    return np.array(class_means)


# -----------------------------
# 5. Plot class-wise means
# -----------------------------

def plot_class_means(relu, tanh, labels):

    relu_means = calculate_class_means(
        relu, labels
    )

    tanh_means = calculate_class_means(
        tanh, labels
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 6)
    )

    axes[0].imshow(
        relu_means,
        aspect="auto",
        cmap="viridis",
    )

    axes[0].set_title("ReLU Class-wise Mean Activations")
    axes[0].set_xlabel("Hidden Neurons")
    axes[0].set_ylabel("Digit Class")

    axes[1].imshow(
        tanh_means,
        aspect="auto",
        cmap="viridis",
    )

    axes[1].set_title("Tanh Class-wise Mean Activations")
    axes[1].set_xlabel("Hidden Neurons")
    axes[1].set_ylabel("Digit Class")

    axes[0].set_yticks(range(10))
    axes[1].set_yticks(range(10))

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "class_mean_activations.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# -----------------------------
# 6. Main
# -----------------------------

def main():

    relu, relu_labels = load_data("relu")
    tanh, tanh_labels = load_data("tanh")

    # Check labels are aligned
    assert np.array_equal(
        relu_labels, tanh_labels
    )

    print("ReLU shape:", relu.shape)
    print("Tanh shape:", tanh.shape)

    print(
        "ReLU labels:",
        np.unique(relu_labels, return_counts=True)
    )

    print(
        "Tanh labels:",
        np.unique(tanh_labels, return_counts=True)
    )

    plot_heatmaps(relu, tanh)

    plot_distributions(relu, tanh)

    plot_class_means(
        relu,
        tanh,
        relu_labels,
    )

    print("\nVisualization completed.")


if __name__ == "__main__":
    main()
