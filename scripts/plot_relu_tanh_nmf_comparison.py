
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. NMF reconstruction errors
# --------------------------------------------------

ranks = [4, 8, 12, 16, 24, 32, 48, 64]

relu_errors = [
    0.5199224,
    0.4222895,
    0.3703664,
    0.3331290,
    0.2830227,
    0.2457550,
    0.1944284,
    0.1552279
]

tanh_errors = [
    0.5166458,
    0.4441593,
    0.4075516,
    0.3833076,
    0.3498141,
    0.3235556,
    0.2816786,
    0.2425840
]


# --------------------------------------------------
# 2. Hoyer sparsity of NMF H matrices
# --------------------------------------------------

sparsity_ranks = [16, 32]

relu_sparsity = [0.5625, 0.7040]
tanh_sparsity = [0.4814, 0.6601]


# --------------------------------------------------
# 3. Create comparison figure
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))


# Left: Reconstruction error
axes[0].plot(
    ranks,
    relu_errors,
    marker="o",
    linewidth=2,
    label="ReLU"
)

axes[0].plot(
    ranks,
    tanh_errors,
    marker="s",
    linewidth=2,
    label="Tanh Positive"
)

axes[0].set_xlabel("NMF Rank")
axes[0].set_ylabel("Relative Reconstruction Error")
axes[0].set_title("NMF Reconstruction Error")
axes[0].set_xticks(ranks)
axes[0].legend()
axes[0].grid(alpha=0.3)


# Right: Hoyer sparsity
x = np.arange(len(sparsity_ranks))
width = 0.35

axes[1].bar(
    x - width / 2,
    relu_sparsity,
    width,
    label="ReLU"
)

axes[1].bar(
    x + width / 2,
    tanh_sparsity,
    width,
    label="Tanh Positive"
)

axes[1].set_xlabel("NMF Rank")
axes[1].set_ylabel("Mean Hoyer Sparsity")
axes[1].set_title("NMF Component Sparsity")
axes[1].set_xticks(x)
axes[1].set_xticklabels(sparsity_ranks)
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)


# --------------------------------------------------
# 4. Save figure
# --------------------------------------------------

fig.suptitle(
    "Comparison of ReLU and Tanh Positive NMF",
    fontsize=14
)

fig.tight_layout()

output_path = OUTPUT_DIR / "relu_tanh_nmf_comparison.png"

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"Figure saved to: {output_path}")
