
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =====================================
# 1. Configuration
# =====================================

ROOT = Path(__file__).resolve().parents[1]

ACTIVATION_DIR = ROOT / "reports" / "activations"
NMF_DIR = ROOT / "reports" / "nmf"
FIG_DIR = ROOT / "reports" / "figures" / "nmf"
METRICS_DIR = ROOT / "reports" / "metrics"

FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

RANKS = [16, 32]


# =====================================
# 2. Load labels
# =====================================

with np.load(
    ACTIVATION_DIR / "relu_activations.npz"
) as data:
    labels = data["labels"]

print("Labels shape:", labels.shape)


# =====================================
# 3. Analyze class-wise activations
# =====================================

for k in RANKS:

    print(f"\n===== NMF Class Analysis: k={k} =====")

    with np.load(NMF_DIR / f"relu_nmf_k{k}.npz") as data:
        W = data["W"]

    assert W.shape[0] == len(labels), (
        "W and labels have inconsistent sample counts"
    )

    # Calculate mean activation for each digit class
    class_means = np.zeros((10, k))

    for digit in range(10):

        mask = labels == digit

        class_means[digit] = W[mask].mean(axis=0)

    # Normalize each component across digit classes
    column_max = class_means.max(axis=0, keepdims=True)

    normalized = np.divide(
        class_means,
        column_max,
        out=np.zeros_like(class_means),
        where=column_max > 0
    )

    # =================================
    # 3.1 Class-wise activation heatmap
    # =================================

    plt.figure(figsize=(14, 6))

    plt.imshow(
        normalized,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    plt.colorbar(label="Normalized Mean Activation")

    plt.xlabel("NMF Components")
    plt.ylabel("Digit Class")
    plt.yticks(range(10))

    plt.title(
        f"ReLU NMF Class-wise Activations (k={k})"
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / f"relu_nmf_class_activation_k{k}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =================================
    # 3.2 Find strongest digit class
    # =================================

    dominant_classes = np.argmax(class_means, axis=0)

    print("\nComponent -> Dominant Digit")

    for component, digit in enumerate(dominant_classes):

        print(
            f"Component {component:2d} "
            f"-> Digit {digit}"
        )

    # Save numerical results
    np.savez_compressed(
        METRICS_DIR / "nmf_class_activation_k{k}.npz",
        class_means=class_means,
        normalized=normalized,
        dominant_classes=dominant_classes
    )

print("===== Class Analysis Completed =====")
print("Figures saved to:", FIG_DIR)
