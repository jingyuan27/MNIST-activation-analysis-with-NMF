
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =====================================
# 1. Configuration
# =====================================

ROOT = Path(__file__).resolve().parents[1]

NMF_DIR = ROOT / "reports" / "nmf"
FIG_DIR = ROOT / "reports" / "figures" / "nmf"
METRICS_DIR = ROOT / "reports" / "metrics"

FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

RANKS = [16, 32]
TOP_N = 10


# =====================================
# 2. Calculate component sparsity
# =====================================

def calculate_hoyer_sparsity(H):
    """
    Calculate Hoyer sparsity for each NMF component.
    Each row of H represents one component.
    """

    n = H.shape[1]

    l1 = np.sum(np.abs(H), axis=1)
    l2 = np.sqrt(np.sum(H ** 2, axis=1))

    sparsity = np.zeros(H.shape[0])

    valid = l2 > 0

    sparsity[valid] = (
        np.sqrt(n) - l1[valid] / l2[valid]
    ) / (np.sqrt(n) - 1)

    return sparsity


# =====================================
# 3. Load and visualize components
# =====================================

all_metrics = {}

for k in RANKS:

    print(f"\n===== Visualizing NMF: k={k} =====")

    file_path = NMF_DIR / f"relu_nmf_k{k}.npz"

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    with np.load(file_path) as data:
        W = data["W"]
        H = data["H"]

    print("W shape:", W.shape)
    print("H shape:", H.shape)

    # Normalize each component for visualization.
    # This removes differences in component scale.

    row_sums = H.sum(axis=1, keepdims=True)

    H_normalized = np.divide(
        H,
        row_sums,
        out=np.zeros_like(H),
        where=row_sums > 0
    )

    # =================================
    # 3.1 Component heatmap
    # =================================

    plt.figure(figsize=(14, 7))

    plt.imshow(
        H_normalized,
        aspect="auto",
        cmap="viridis"
    )

    plt.colorbar(label="Normalized Weight")

    plt.xlabel("Hidden Neurons")
    plt.ylabel("NMF Components")
    plt.title(f"ReLU NMF Components (k={k})")

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / f"relu_nmf_components_k{k}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =================================
    # 3.2 Top neurons for first 4 components
    # =================================

    num_components = min(4, k)

    fig, axes = plt.subplots(
        num_components,
        1,
        figsize=(12, 3 * num_components)
    )

    if num_components == 1:
        axes = [axes]

    for i, ax in enumerate(axes):

        top_indices = np.argsort(
            H_normalized[i]
        )[-TOP_N:][::-1]

        top_values = H_normalized[i, top_indices]

        ax.bar(
            range(TOP_N),
            top_values
        )

        ax.set_xticks(range(TOP_N))
        ax.set_xticklabels(top_indices)

        ax.set_xlabel("Hidden Neuron Index")
        ax.set_ylabel("Normalized Weight")
        ax.set_title(f"Component {i}: Top {TOP_N} Neurons")

    plt.tight_layout()

    output_path = FIG_DIR / f"relu_nmf_components_k{k}.png"

plt.savefig(
    str(output_path),
    format="png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved:", output_path)

plt.close()

    # =================================
    # 3.3 Hoyer sparsity
    # =================================

sparsity = calculate_hoyer_sparsity(H)

all_metrics[str(k)] = {
        "mean_sparsity": float(np.mean(sparsity)),
        "std_sparsity": float(np.std(sparsity)),
        "min_sparsity": float(np.min(sparsity)),
        "max_sparsity": float(np.max(sparsity))
    }

plt.figure(figsize=(8, 5))

plt.hist(
        sparsity,
        bins=15,
        edgecolor="black"
    )

plt.xlabel("Hoyer Sparsity")
plt.ylabel("Number of Components")
plt.title(f"NMF Component Sparsity (k={k})")

plt.tight_layout()

plt.savefig(
        FIG_DIR / f"relu_nmf_sparsity_k{k}.png",
        dpi=300,
        bbox_inches="tight"
    )

plt.close()

    # =================================
    # 3.4 Export top neuron information
    # =================================

csv_path = FIG_DIR / f"relu_nmf_top_neurons_k{k}.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "component",
            "neuron",
            "normalized_weight"
        ])

        for i in range(k):

            top_indices = np.argsort(
                H_normalized[i]
            )[-TOP_N:][::-1]

            for neuron in top_indices:

                writer.writerow([
                    i,
                    int(neuron),
                    float(H_normalized[i, neuron])
                ])

print("Mean Hoyer sparsity:", np.mean(sparsity))

# =====================================
# 4. Save metrics
# =====================================

metrics_path = METRICS_DIR / "nmf_component_sparsity.json"

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(
        all_metrics,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\n===== Visualization Completed =====")
print("Figures saved to:", FIG_DIR)
print("Metrics saved to:", metrics_path)
