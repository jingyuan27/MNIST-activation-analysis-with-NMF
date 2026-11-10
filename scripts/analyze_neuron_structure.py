
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt


# =====================================
# 1. Configuration
# =====================================

DATA_DIR = Path("reports/activations")
OUTPUT_DIR = Path("reports/metrics")
FIGURE_DIR = Path("reports/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SATURATION_THRESHOLD = 0.95
STRICT_SATURATION_THRESHOLD = 0.99


# =====================================
# 2. Load activation matrices
# =====================================

def load_data(activation):

    data = np.load(
        DATA_DIR / f"{activation}_activations.npz"
    )

    return data["activations"], data["labels"]


# =====================================
# 3. Per-neuron sparsity
# =====================================

def calculate_zero_ratio(activations):

    return np.mean(
        activations == 0,
        axis=0
    )


# =====================================
# 4. Tanh saturation
# =====================================

def calculate_saturation_ratio(
    activations,
    threshold
):

    return np.mean(
        np.abs(activations) > threshold,
        axis=0
    )


# =====================================
# 5. Class selectivity (Eta squared)
# =====================================

def calculate_eta_squared(activations, labels):

    grand_mean = np.mean(
        activations,
        axis=0
    )

    between_class_sum = np.zeros(
        activations.shape[1]
    )

    for digit in range(10):

        class_data = activations[
            labels == digit
        ]

        class_mean = np.mean(
            class_data,
            axis=0
        )

        n_class = len(class_data)

        between_class_sum += (
            n_class * (class_mean - grand_mean) ** 2
        )

    total_sum = np.sum(
        (activations - grand_mean) ** 2,
        axis=0
    )

    eta_squared = np.divide(
        between_class_sum,
        total_sum,
        out=np.zeros_like(between_class_sum),
        where=total_sum > 0
    )

    return eta_squared


# =====================================
# 6. Save per-neuron metrics
# =====================================

def save_neuron_metrics(
    relu_zero,
    tanh_zero,
    tanh_sat,
    tanh_strict_sat,
    relu_eta,
    tanh_eta
):

    output_file = OUTPUT_DIR / "neuron_metrics.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "neuron",
            "relu_zero_ratio",
            "tanh_zero_ratio",
            "tanh_saturation_ratio_095",
            "tanh_saturation_ratio_099",
            "relu_eta_squared",
            "tanh_eta_squared"
        ])

        for j in range(len(relu_zero)):

            writer.writerow([
                j,
                relu_zero[j],
                tanh_zero[j],
                tanh_sat[j],
                tanh_strict_sat[j],
                relu_eta[j],
                tanh_eta[j]
            ])


# =====================================
# 7. Plot neuron-level distributions
# =====================================

def plot_neuron_metrics(
    relu_zero,
    tanh_sat,
    tanh_strict_sat,
    relu_eta,
    tanh_eta
):

    # ReLU zero ratio
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        relu_zero,
        bins=20,
        edgecolor="black"
    )

    ax.set_title("ReLU Per-neuron Zero Ratio")
    ax.set_xlabel("Zero Ratio")
    ax.set_ylabel("Number of Neurons")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "relu_neuron_sparsity.png",
        dpi=200
    )

    plt.show()
    plt.close(fig)

    # Tanh saturation ratio
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        tanh_sat,
        bins=20,
        alpha=0.7,
        label="Threshold 0.95"
    )

    ax.hist(
        tanh_strict_sat,
        bins=20,
        alpha=0.7,
        label="Threshold 0.99"
    )

    ax.set_title("Tanh Per-neuron Saturation Ratio")
    ax.set_xlabel("Saturation Ratio")
    ax.set_ylabel("Number of Neurons")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "tanh_neuron_saturation.png",
        dpi=200
    )

    plt.show()
    plt.close(fig)

    # Eta squared comparison
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        relu_eta,
        bins=20,
        alpha=0.7,
        label="ReLU"
    )

    ax.hist(
        tanh_eta,
        bins=20,
        alpha=0.7,
        label="Tanh"
    )

    ax.set_title("Per-neuron Class Selectivity")
    ax.set_xlabel("Eta Squared")
    ax.set_ylabel("Number of Neurons")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "neuron_class_selectivity.png",
        dpi=200
    )

    plt.show()
    plt.close(fig)


# =====================================
# 8. Main
# =====================================

def main():

    relu, relu_labels = load_data("relu")
    tanh, tanh_labels = load_data("tanh")

    assert np.array_equal(
        relu_labels,
        tanh_labels
    )

    print("Activation matrices loaded.")
    print("ReLU shape:", relu.shape)
    print("Tanh shape:", tanh.shape)

    # Calculate metrics
    relu_zero = calculate_zero_ratio(relu)
    tanh_zero = calculate_zero_ratio(tanh)

    tanh_sat = calculate_saturation_ratio(
        tanh,
        SATURATION_THRESHOLD
    )

    tanh_strict_sat = calculate_saturation_ratio(
        tanh,
        STRICT_SATURATION_THRESHOLD
    )

    relu_eta = calculate_eta_squared(
        relu,
        relu_labels
    )

    tanh_eta = calculate_eta_squared(
        tanh,
        tanh_labels
    )

    # Summary statistics
    summary = {

        "relu": {
            "mean_zero_ratio": float(np.mean(relu_zero)),
            "std_zero_ratio": float(np.std(relu_zero)),
            "mean_eta_squared": float(np.mean(relu_eta)),
            "max_eta_squared": float(np.max(relu_eta))
        },

        "tanh": {
            "mean_zero_ratio": float(np.mean(tanh_zero)),
            "mean_saturation_ratio_095": float(np.mean(tanh_sat)),
            "mean_saturation_ratio_099": float(np.mean(tanh_strict_sat)),
            "mean_eta_squared": float(np.mean(tanh_eta)),
            "max_eta_squared": float(np.max(tanh_eta))
        }
    }

    # Print results
    print("===== ReLU =====")
    print(json.dumps(
        summary["relu"],
        indent=4
    ))

    print("===== Tanh =====")
    print(json.dumps(
        summary["tanh"],
        indent=4
    ))

    # Save results
    save_neuron_metrics(
        relu_zero,
        tanh_zero,
        tanh_sat,
        tanh_strict_sat,
        relu_eta,
        tanh_eta
    )

    with open(
        OUTPUT_DIR / "neuron_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # Plot results
    plot_neuron_metrics(
        relu_zero,
        tanh_sat,
        tanh_strict_sat,
        relu_eta,
        tanh_eta
    )

    print("Neuron structure analysis completed.")


if __name__ == "__main__":
    main()
